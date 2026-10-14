"""风控规则引擎：纯函数规则集，统一签名。

def rule_xxx(invoice, ctx, config) -> list[Finding]
ctx 提供 dedup_store / duplicate_hits / file_sha256 等上下文。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Any

from invoice_agent.models.invoice import InvoiceData
from invoice_agent.models.risk import Finding, RiskLevel

if TYPE_CHECKING:
    from invoice_agent.config import AppConfig
    from invoice_agent.services.dedup_store import DedupStore

RuleFunc = Callable[[InvoiceData, dict, "AppConfig"], list[Finding]]


def _to_date(s: str | date) -> date:
    return (
        s
        if isinstance(s, date)
        else date.fromisoformat(str(s).replace("年", "-").replace("月", "-").replace("日", ""))
    )


def rule_r001_buyer_mismatch(invoice, ctx, config) -> list[Finding]:
    """R001 抬头/税号与公司信息不符。"""
    findings = []
    company = config.company
    name_ok = invoice.buyer_name == company.name
    tax_ok = invoice.buyer_tax_id in (None, company.tax_id)
    if not name_ok or not tax_ok:
        detail = []
        if not name_ok:
            detail.append(f"抬头「{invoice.buyer_name}」≠ 公司名「{company.name}」")
        if not tax_ok:
            detail.append(f"税号「{invoice.buyer_tax_id}」≠ 公司税号「{company.tax_id}」")
        findings.append(
            Finding(
                rule_id="R001",
                rule_name="抬头/税号不符",
                level=RiskLevel.HIGH,
                message="发票购买方信息与公司信息不一致：" + "；".join(detail),
                evidence={"buyer_name": invoice.buyer_name, "buyer_tax_id": invoice.buyer_tax_id},
            )
        )
    return findings


def rule_r002_duplicate(invoice, ctx, config) -> list[Finding]:
    """R002 重复提交。"""
    hits = ctx.get("duplicate_hits") or []
    if not hits:
        return []
    return [
        Finding(
            rule_id="R002",
            rule_name="重复报销",
            level=RiskLevel.HIGH,
            message=(
                f"该发票已于 {hits[0].first_submitted_at:%Y-%m-%d %H:%M} 提交过"
                f"（状态：{hits[0].status}），疑似重复报销"
            ),
            evidence={"hits": [h.model_dump(mode="json") for h in hits]},
        )
    ]


def rule_r003_serial(invoice, ctx, config) -> list[Finding]:
    """R003 连号发票：同销售方历史记录中号码差值 ≤ N。"""
    store: DedupStore | None = ctx.get("dedup_store")
    if store is None or not invoice.seller_tax_id:
        return []
    tolerance = config.risk.serial_gap_tolerance
    try:
        current = int(invoice.invoice_number)
    except ValueError:
        return []
    recent = store.find_recent_by_seller(
        invoice.seller_tax_id, days=90, exclude_number=invoice.invoice_number
    )
    close = []
    for r in recent:
        try:
            if abs(int(r["invoice_number"]) - current) <= tolerance:
                close.append(r["invoice_number"])
        except (ValueError, TypeError):
            continue
    if not close:
        return []
    return [
        Finding(
            rule_id="R003",
            rule_name="连号发票",
            level=RiskLevel.MEDIUM,
            message=(
                f"与同销售方已提交发票号码差值 ≤ {tolerance}（库内号码：{', '.join(close[:5])}），"
                "疑似连号开具，请核实是否为同一批次拆分开票"
            ),
            evidence={"nearby_numbers": close, "current": invoice.invoice_number},
        )
    ]


def rule_r004_split(invoice, ctx, config) -> list[Finding]:
    """R004 大额拆分：同销售方窗口期内累计金额 ≥ 阈值且张数达标（含本张）。"""
    store: DedupStore | None = ctx.get("dedup_store")
    if store is None or not invoice.seller_tax_id:
        return []
    cfg = config.risk
    recent = store.find_recent_by_seller(
        invoice.seller_tax_id, days=cfg.split_window_days, exclude_number=invoice.invoice_number
    )
    total = invoice.total_amount + sum(r["total_amount"] for r in recent)
    count = len(recent) + 1
    if total < cfg.split_amount_threshold or count < cfg.split_min_count:
        return []
    return [
        Finding(
            rule_id="R004",
            rule_name="疑似大额拆分",
            level=RiskLevel.MEDIUM,
            message=(
                f"同销售方 {cfg.split_window_days} 天内累计 {count} 张、合计 ¥{total:.2f}"
                f"（阈值 ¥{cfg.split_amount_threshold:.0f}），疑似化整为零拆分开票"
            ),
            evidence={"window_total": total, "count": count, "recent": recent},
        )
    ]


def rule_r005_sensitive(invoice, ctx, config) -> list[Finding]:
    """R005 敏感商品：明细命中词表。"""
    keywords = config.risk.sensitive_keywords
    hits = [item.name for item in invoice.items if any(kw in item.name for kw in keywords)]
    if not hits:
        return []
    return [
        Finding(
            rule_id="R005",
            rule_name="敏感商品",
            level=RiskLevel.HIGH,
            message=f"明细包含敏感商品（{'、'.join(hits)}），此类商品通常不得税前扣除/报销",
            evidence={"matched_items": hits},
        )
    ]


def rule_r006_deadline(invoice, ctx, config) -> list[Finding]:
    """R006 报销时效。"""
    inv_date = _to_date(invoice.invoice_date)
    days = (date.today() - inv_date).days
    if days <= config.risk.reimburse_deadline_days:
        return []
    return [
        Finding(
            rule_id="R006",
            rule_name="超过报销时效",
            level=RiskLevel.LOW,
            message=f"开票日期距今 {days} 天，超过 {config.risk.reimburse_deadline_days} 天报销时效",
            evidence={"invoice_date": inv_date.isoformat(), "days": days},
        )
    ]


def rule_r007_date_anomaly(invoice, ctx, config) -> list[Finding]:
    """R007 日期异常：未来日期 / 周末大额餐饮。"""
    findings = []
    inv_date = _to_date(invoice.invoice_date)
    if inv_date > date.today():
        findings.append(
            Finding(
                rule_id="R007",
                rule_name="日期异常",
                level=RiskLevel.MEDIUM,
                message=f"开票日期 {inv_date} 晚于今天，疑似日期错误",
                evidence={"invoice_date": inv_date.isoformat()},
            )
        )
        return findings
    is_weekend = inv_date.weekday() >= 5
    is_catering = any("餐" in i.name for i in invoice.items)
    if is_weekend and is_catering and invoice.total_amount >= 1000:
        findings.append(
            Finding(
                rule_id="R007",
                rule_name="周末大额餐饮",
                level=RiskLevel.LOW,
                message=f"开票日 {inv_date} 为周末且餐饮金额 ¥{invoice.total_amount:.2f}，请确认是否为个人消费",
                evidence={"invoice_date": inv_date.isoformat()},
            )
        )
    return findings


def rule_r008_amount_check(invoice, ctx, config) -> list[Finding]:
    """R008 金额勾稽：价税合计 ≈ 不含税 + 税额；明细合计 ≈ 不含税金额。"""
    findings = []
    TOL = 0.05
    if (
        invoice.amount_without_tax is not None
        and invoice.tax_amount is not None
        and abs(invoice.amount_without_tax + invoice.tax_amount - invoice.total_amount) > TOL
    ):
        findings.append(
            Finding(
                rule_id="R008",
                rule_name="金额勾稽异常",
                level=RiskLevel.MEDIUM,
                message=(
                    f"价税合计 ¥{invoice.total_amount:.2f} ≠ 不含税 ¥{invoice.amount_without_tax:.2f}"
                    f" + 税额 ¥{invoice.tax_amount:.2f}，存在识别错误或票面异常"
                ),
                evidence={
                    "total": invoice.total_amount,
                    "without_tax": invoice.amount_without_tax,
                    "tax": invoice.tax_amount,
                },
            )
        )
    if invoice.amount_without_tax is not None and invoice.items:
        items_sum = sum(i.amount or 0 for i in invoice.items)
        if items_sum and abs(items_sum - invoice.amount_without_tax) > TOL:
            findings.append(
                Finding(
                    rule_id="R008",
                    rule_name="明细勾稽异常",
                    level=RiskLevel.MEDIUM,
                    message=(
                        f"明细合计 ¥{items_sum:.2f} ≠ 不含税金额 ¥{invoice.amount_without_tax:.2f}"
                    ),
                    evidence={"items_sum": items_sum, "without_tax": invoice.amount_without_tax},
                )
            )
    return findings


ALL_RULES: list[RuleFunc] = [
    rule_r001_buyer_mismatch,
    rule_r002_duplicate,
    rule_r003_serial,
    rule_r004_split,
    rule_r005_sensitive,
    rule_r006_deadline,
    rule_r007_date_anomaly,
    rule_r008_amount_check,
]


def run_risk_rules(invoice: InvoiceData, ctx: dict[str, Any], config: AppConfig) -> list[Finding]:
    """顺序执行全部规则，汇总 findings。"""
    findings: list[Finding] = []
    for rule in ALL_RULES:
        findings.extend(rule(invoice, ctx, config))
    return findings
