"""多发票批量报销编排：逐张跑现有单票图 → 分类统计聚合 → 提交审批流程。

不改动 LangGraph 图：每张发票一个独立 thread_id 调 runner.iter_run
（识票重试/验真/查重/风控全部复用），单张失败不中断整批。

API 两步提交：preview 落待确认批次（save_pending），人工审核后 confirm
（confirm_pending_batch）才真正提交审批流并写台账——pending 不进查重台账，
confirm 唯一真相源是服务端 pending 存储，客户端无法篡改批次内容。
"""

from __future__ import annotations

import hashlib
import logging
import uuid
import warnings
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from invoice_agent.config import get_app_config
from invoice_agent.models.invoice import InvoiceData
from invoice_agent.models.reimburse import (
    BatchItem,
    CategoryTotal,
    ExpenseCategory,
    ReimburseBatch,
    SubmissionResult,
)
from invoice_agent.rules.expense_category import infer_category
from invoice_agent.runner import iter_run
from invoice_agent.services.dedup_store import DedupStore
from invoice_agent.services.expense_flow_provider import (
    ExpenseFlowProvider,
    get_expense_flow_provider,
)

ProgressCb = Callable[[int, str, str], None]  # (序号, 文件名, 当前节点)

logger = logging.getLogger(__name__)


class PendingNotFoundError(LookupError):
    """待确认批次不存在（未 preview 过或已被确认/删除）。"""


class PendingExpiredError(LookupError):
    """待确认批次已过期，需重新 preview。"""


def _default_store() -> DedupStore:
    return DedupStore(get_app_config().db_path)


def _resolve_category(invoice: InvoiceData) -> tuple[ExpenseCategory, str]:
    """模型明确输出非「其他」时直接采信，否则走关键词规则兜底。"""
    if invoice.expense_category != ExpenseCategory.OTHER:
        return invoice.expense_category, "model"
    return infer_category(invoice)


def _make_item(fname: str, sha: str, state: dict) -> BatchItem:
    """从单张的最终 state 提取批次条目；识别失败记入 error 而不抛出。"""
    invoice: InvoiceData | None = state.get("invoice")
    if invoice is None:
        return BatchItem(
            file_name=fname,
            file_sha256=sha,
            error=state.get("extract_error") or "识别失败，请人工复核",
        )
    category, source = _resolve_category(invoice)
    findings = state.get("risk_findings") or []
    risk_level = (
        max((f.level for f in findings), key=lambda lv: lv.rank).value if findings else None
    )
    return BatchItem(
        file_name=fname,
        file_sha256=sha,
        invoice=invoice,
        category=category,
        category_source=source,
        verification_status=state["verification"].status if state.get("verification") else None,
        has_duplicates=bool(state.get("duplicate_hits")),
        risk_level=risk_level,
    )


def run_batch(
    files: list[Path],
    *,
    title: str | None = None,
    progress_cb: ProgressCb | None = None,
) -> ReimburseBatch:
    """逐张识别并聚合为报销批次（不写台账，写台账只在 submit_batch）。"""
    if not files:
        raise ValueError("未提供任何发票文件")
    cfg = get_app_config()
    batch_id = uuid.uuid4().hex[:8]
    items: list[BatchItem] = []
    for i, f in enumerate(files, start=1):
        fname = f.name
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        state: dict = {}
        for event, payload, _handler in iter_run(
            str(f), None, f"batch-{batch_id}-{i}", stream_tokens=False
        ):
            if event == "node":
                if progress_cb:
                    progress_cb(i, fname, payload)
            elif event == "done":
                state = payload
        items.append(_make_item(fname, sha, state))
    return build_summary(items, batch_id=batch_id, title=title or _default_title(cfg))


def _category_order() -> list[ExpenseCategory]:
    """类别展示顺序：跟随 config.reimburse.categories，异常值兜底按枚举定义序。"""
    configured = [ExpenseCategory(c) for c in get_app_config().reimburse.categories]
    return configured + [c for c in ExpenseCategory if c not in configured]


def build_summary(
    items: list[BatchItem],
    *,
    batch_id: str | None = None,
    title: str | None = None,
    exclude_numbers: frozenset[str] | set[str] = frozenset(),
) -> ReimburseBatch:
    """按排除集合（用户勾掉的发票号）重算分类统计与合计。"""
    included = [
        it
        for it in items
        if it.invoice is not None and it.invoice.invoice_number not in exclude_numbers
    ]
    totals: dict[ExpenseCategory, CategoryTotal] = {}
    for it in included:
        t = totals.setdefault(
            it.category, CategoryTotal(category=it.category, count=0, total_amount=0.0)
        )
        t.count += 1
        t.total_amount += it.invoice.total_amount
    ordered = [totals[c] for c in _category_order() if c in totals]
    return ReimburseBatch(
        batch_id=batch_id or uuid.uuid4().hex[:8],
        title=title or _default_title(get_app_config()),
        items=items,
        category_totals=ordered,
        total_amount=sum(t.total_amount for t in ordered),
    )


def _default_title(cfg) -> str:
    return f"{cfg.reimburse.title_prefix}（{date.today().isoformat()}）"


def submit_batch(
    batch: ReimburseBatch,
    *,
    provider: ExpenseFlowProvider | None = None,
    store: DedupStore | None = None,
) -> SubmissionResult:
    """提交审批流程并落台账；provider/store 可注入（测试），缺省走工厂与默认库。"""
    provider = provider or get_expense_flow_provider()
    store = store or DedupStore(get_app_config().db_path)
    result = provider.submit(batch)
    store.save_batch(batch, result)
    return result


def save_pending(batch: ReimburseBatch, *, store: DedupStore | None = None) -> None:
    """待确认批次落库（API preview 调用）；store 可注入（测试），缺省走默认库。"""
    (store or _default_store()).save_pending_batch(batch)


def confirm_pending_batch(
    batch_id: str,
    *,
    exclude_numbers: frozenset[str] | set[str] = frozenset(),
    provider: ExpenseFlowProvider | None = None,
    store: DedupStore | None = None,
) -> tuple[ReimburseBatch, SubmissionResult]:
    """人工确认后提交待确认批次并落台账。

    - 批次内容只从服务端 pending 存储读取（防篡改），请求侧最多传排除集合；
    - 过期批次删行并抛 PendingExpiredError；
    - provider.submit 抛异常时 pending 保留（人工确认的意图不能悄悄丢弃，可重试）；
    - submit 成功但落台账失败：回写 result 后 pending 保留，再次 confirm 跳过
      provider 直接重试落台账（防重复创建审批实例）。
    """
    store = store or _default_store()
    batch = store.get_pending_batch(batch_id)
    if batch is None:
        raise PendingNotFoundError(f"待确认批次不存在：{batch_id}")
    created_at = store.get_pending_created_at(batch_id)
    expire_days = get_app_config().reimburse.pending_expire_days
    if created_at is not None and datetime.now() - created_at > timedelta(days=expire_days):
        store.delete_pending_batch(batch_id)
        raise PendingExpiredError(f"待确认批次已过期（>{expire_days} 天），请重新 preview")

    # 已回写过提交结果则复用（幂等），否则提交并立即回写锚点
    result = store.get_pending_result(batch_id)
    if result is None:
        provider = provider or get_expense_flow_provider()
        result = provider.submit(batch)
        try:
            store.mark_pending_submitted(batch_id, result)
        except Exception as exc:  # noqa: BLE001 —— 回写失败仅告警，save_batch 本就会落台账
            warnings.warn(f"回写提交结果失败（不影响本次落台账）：{exc}")

    # 先按排除集合裁剪 items 再聚合：build_summary 的 exclude 只重算合计不裁剪
    # items，直接用会让被排除的票仍经 save_batch 进台账。沿用原 batch_id 便于追溯。
    kept = [
        it
        for it in batch.items
        if it.invoice is not None and it.invoice.invoice_number not in exclude_numbers
    ]
    if not kept:
        raise ValueError("排除后没有任何可提交的发票")
    new_batch = build_summary(kept, batch_id=batch.batch_id, title=batch.title)
    try:
        store.save_batch(new_batch, result)
    except Exception:
        logger.exception("待确认批次 %s 落台账失败，pending 保留可重试 confirm", batch_id)
        raise
    store.delete_pending_batch(batch_id)
    return new_batch, result
