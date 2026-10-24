"""多发票批量报销编排：逐张跑现有单票图 → 分类统计聚合 → 提交审批流程。

不改动 LangGraph 图：每张发票一个独立 thread_id 调 runner.iter_run
（识票重试/验真/查重/风控全部复用），单张失败不中断整批。
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from datetime import date
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
