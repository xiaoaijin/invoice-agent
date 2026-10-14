"""自反馈优化：回答质量自评 / 识票字段自检 / 风控结论复核。

三个 critic 均为可注入 LLM 的纯函数（llm_fn 为 None 或调用异常时静默降级放行），
便于按现有 monkeypatch 模式测试。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from invoice_agent.graph.state import AgentState
from invoice_agent.models.invoice import InvoiceData
from invoice_agent.models.risk import Finding, RiskLevel

logger = logging.getLogger(__name__)

# llm_fn：接收消息列表，返回回答文本。None = LLM 不可用（降级）。
LlmFn = Callable[[list], str] | None

CRITIQUE_SYSTEM_PROMPT = """你是严格的回答质量评审员。给定用户问题、助手回答与系统已掌握的事实（发票数据、验真结论、风控发现、检索条款），
从三个维度打 0~1 分：
- sufficiency：信息是否充分回答了问题；
- relevance：是否答非所问；
- completeness：风险提示是否完整（若事实中存在"高"级风控发现而回答未提及，则 completeness ≤ 0.4）。
issues 用中文逐条列出可改进点；无不足则为空数组。
只输出 JSON：{"sufficiency": 0.x, "relevance": 0.x, "completeness": 0.x, "score": 0.x, "issues": ["..."]}
score 取三个维度的加权综合（sufficiency 0.4 / relevance 0.3 / completeness 0.3）。"""

FIELD_CHECK_SYSTEM_PROMPT = """你是发票数据质检员。给定 OCR 识别出的发票 JSON，只检查以下两类问题：
1) 关键字段缺失或可疑 null。关键字段仅指：发票类型、发票号码、开票日期、购买方/销售方名称与税号、
价税合计/合计金额、明细中的商品名称与金额。明细行的单位、规格、数量等非核心字段缺失不算问题，勿报；
2) 票面内部逻辑矛盾，仅限：开票日期晚于当前日期、金额勾稽不符（明细/合计/税额互相矛盾）、
税率与商品常识明显矛盾（如餐饮服务出现 13%）。

严禁检查以下内容（由代码级校验负责，报了也会被忽略）：
- 号码/税号/信用代码的位数与字符编码规则（8 位专票、20 位数电票、18 位信用代码均正常，含 W/X/Y 字符合法）；
- 票面没有的信息、真实性判断、与外部数据的一致性。
每条问题一句话中文描述。只输出 JSON：{"issues": ["..."]}。无问题输出 {"issues": []}。"""

RISK_REVIEW_SYSTEM_PROMPT = """你是风控复核员。给定发票 JSON 与规则引擎输出的风险发现列表（含证据），复核：
1) 误报：规则机械命中但业务上不成立 → 从 adjusted 中移除，并在 removed 列出其 rule_id；
2) 漏报：票面明显异常但规则未覆盖 → 以 rule_id="LLM-REVIEW" 补充到 added（level 按严重度"高"/"中"/"低"，
message 注明是模型复核发现、需人工确认）；
3) summary 用一句话给出复核结论。
只基于给定票面数据判断，不确定的降级为"低"并在 message 注明。
只输出 JSON：{"adjusted": [...], "removed": ["R00x"], "added": [...], "summary": "..."}
adjusted 为复核后完整的发现列表（保留项沿用原字段）。"""

LEVEL_BY_VALUE = {lv.value: lv for lv in RiskLevel}


class CritiqueResult(BaseModel):
    """回答自评结果。"""

    score: float = 1.0
    passed: bool = True
    sufficiency: float = 1.0
    relevance: float = 1.0
    completeness: float = 1.0
    issues: list[str] = Field(default_factory=list)


class RiskReview(BaseModel):
    """风控复核结果。"""

    adjusted: list[Finding] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    added: list[Finding] = Field(default_factory=list)
    summary: str = ""


def default_llm_fn() -> LlmFn:
    """生产默认实现：对话模型。放在独立工厂便于测试替换。"""
    from invoice_agent.services.llm import get_chat_model

    try:
        model = get_chat_model()
    except Exception as exc:  # noqa: BLE001 — 无 API key 等场景直接降级
        logger.warning("critic LLM 不可用，走降级：%s", exc)
        return None
    return lambda messages: model.invoke(messages).content


def _extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中提取首个 JSON 对象（容忍 ```json 包裹与前后缀文本）。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clamp(value: Any, default: float = 1.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# ---- ① 回答质量自评 ----


def critique_answer(
    answer: str, state: AgentState, llm_fn: LlmFn, pass_score: float
) -> CritiqueResult:
    """LLM 自评最终回答。任何异常 → passed=True 放行（不阻塞主流程）。"""
    if llm_fn is None:
        return CritiqueResult()
    findings = state.get("risk_findings") or []
    fact_parts: list[str] = []
    invoice = state.get("invoice")
    if invoice is not None:
        fact_parts.append(
            f"发票数据：{invoice.summary_text() if hasattr(invoice, 'summary_text') else invoice}"
        )
    if findings:
        fact_parts.append(
            "风控发现：" + "；".join(f"[{f.level.value}] {f.rule_id} {f.message}" for f in findings)
        )
    chunks = state.get("rag_chunks") or []
    if chunks:
        fact_parts.append("检索条款：" + " | ".join(c.doc_title for c in chunks[:5]))
    verification = state.get("verification")
    if verification is not None:
        fact_parts.append(f"验真结论：{verification.status}（{verification.message}）")

    user_prompt = (
        f"## 用户问题\n{state.get('user_question') or '（无，发票查验汇总）'}\n\n"
        f"## 助手回答\n{answer}\n\n"
        f"## 系统事实\n" + ("\n".join(fact_parts) or "（无）")
    )
    try:
        raw = llm_fn(
            [
                {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("回答自评 LLM 调用失败（放行）：%s", exc)
        return CritiqueResult()

    data = _extract_json(raw)
    if data is None:
        logger.warning("回答自评输出无法解析（放行）：%s", raw[:200])
        return CritiqueResult()

    sufficiency = _clamp(data.get("sufficiency"))
    relevance = _clamp(data.get("relevance"))
    completeness = _clamp(data.get("completeness"))
    # 事实中存在高级别发现而回答未提及 → completeness 封顶 0.4（防模型漏判）
    has_high = any(f.level == RiskLevel.HIGH for f in findings)
    if has_high and state.get("invoice") is not None and RiskLevel.HIGH.value not in answer:
        completeness = min(completeness, 0.4)
    score = _clamp(data.get("score"), 0.0) or round(
        sufficiency * 0.4 + relevance * 0.3 + completeness * 0.3, 4
    )
    issues = [str(i) for i in (data.get("issues") or []) if str(i).strip()]
    return CritiqueResult(
        score=score,
        passed=score >= pass_score,
        sufficiency=sufficiency,
        relevance=relevance,
        completeness=completeness,
        issues=issues,
    )


# ---- ② 识票字段自检 ----


def field_self_check(invoice: InvoiceData, llm_fn: LlmFn) -> list[str]:
    """对已通过代码级校验的发票字段做 LLM 级逻辑自检。返回问题列表（空 = 通过）；异常降级返回 []。"""
    if llm_fn is None:
        return []
    try:
        raw = llm_fn(
            [
                {"role": "system", "content": FIELD_CHECK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"当前日期：{_today()}\n发票 JSON：\n{invoice.model_dump_json(indent=2)}",
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("字段自检 LLM 调用失败（跳过）：%s", exc)
        return []
    data = _extract_json(raw)
    if data is None:
        logger.warning("字段自检输出无法解析（跳过）：%s", raw[:200])
        return []
    return [f"字段自检：{str(i).strip()}" for i in (data.get("issues") or []) if str(i).strip()]


# ---- ③ 风控结论复核 ----


def review_risk(findings: list[Finding], invoice: InvoiceData | None, llm_fn: LlmFn) -> RiskReview:
    """LLM 复核规则引擎的风控发现（误报移除/漏报补充）。异常降级原样返回；解析失败内部重试 1 次。"""
    if llm_fn is None or invoice is None:
        return RiskReview(adjusted=list(findings))
    findings_json = json.dumps(
        [
            {
                "rule_id": f.rule_id,
                "rule_name": f.rule_name,
                "level": f.level.value,
                "message": f.message,
                "evidence": f.evidence,
            }
            for f in findings
        ],
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = (
        f"发票 JSON：\n{invoice.model_dump_json(indent=2)}\n\n风险发现列表：\n{findings_json}"
    )
    messages = [
        {"role": "system", "content": RISK_REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    data: dict[str, Any] | None = None
    for attempt in range(2):  # 解析失败重试 1 次
        try:
            raw = llm_fn(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("风控复核 LLM 调用失败（原样透传）：%s", exc)
            return RiskReview(adjusted=list(findings))
        data = _extract_json(raw)
        if data is not None:
            break
        logger.warning("风控复核输出无法解析（第 %d 次）：%s", attempt + 1, raw[:200])
    if data is None:
        return RiskReview(adjusted=list(findings))

    by_id = {f.rule_id: f for f in findings}
    removed = [str(r) for r in (data.get("removed") or []) if str(r) in by_id]
    adjusted = [f for f in findings if f.rule_id not in removed]

    added: list[Finding] = []
    for item in data.get("added") or []:
        try:
            added.append(
                Finding(
                    rule_id=str(item.get("rule_id", "LLM-REVIEW")),
                    rule_name=str(item.get("rule_name", "模型复核")),
                    level=LEVEL_BY_VALUE.get(str(item.get("level", "低")), RiskLevel.LOW),
                    message=str(item.get("message", "")),
                    evidence={"review": "llm"},  # 溯源标记：模型复核产出
                )
            )
        except Exception as exc:  # noqa: BLE001 — 单条解析失败不影响其余
            logger.warning("风控复核补充项解析失败：%s（%s）", item, exc)
    adjusted.extend(added)
    return RiskReview(
        adjusted=adjusted,
        removed=removed,
        added=added,
        summary=str(data.get("summary", "")),
    )


def _today() -> str:
    from datetime import date

    return date.today().isoformat()
