import hashlib
from datetime import date, datetime
from pathlib import Path

import pytest

import invoice_agent.services.reimburse_flow as flow_mod
from invoice_agent.models.invoice import InvoiceData, InvoiceItem, InvoiceTypeEnum
from invoice_agent.models.reimburse import (
    BatchItem,
    ExpenseCategory,
    ReimburseBatch,
    SubmissionResult,
)
from invoice_agent.models.risk import Finding, RiskLevel, VerificationResult
from invoice_agent.services.dedup_store import DedupStore
from invoice_agent.services.reimburse_flow import build_summary, run_batch, submit_batch


def make_invoice(number: str, amount: float = 113.0, **kw) -> InvoiceData:
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number=number,
        invoice_date=date(2026, 10, 1),
        buyer_name="示例科技有限公司",
        buyer_tax_id="91330100MA27XXXXXX",
        seller_name="卖方",
        seller_tax_id="91330100MA28YYYYYY",
        total_amount=amount,
        items=[InvoiceItem(name="服务费")],
    )
    base.update(kw)
    return InvoiceData(**base)


def make_state(number: str, **inv_kw) -> dict:
    """模拟单张发票跑完图后的最终 state。"""
    return {
        "invoice": make_invoice(number, **inv_kw),
        "verification": VerificationResult(
            status="PASS", provider="mock", message="ok", checked_at=datetime.now()
        ),
        "duplicate_hits": [],
        "risk_findings": [],
        "answer": "结论",
    }


def fake_iter_run(states: list[dict | Exception]):
    """按文件顺序吐出预设 state 的 iter_run 替身。"""

    def _run(file_path, question, thread_id, *, stream_tokens=True):
        idx = len(_run.calls)
        _run.calls.append(thread_id)
        outcome = states[idx]
        if isinstance(outcome, Exception):
            raise outcome
        yield "node", "recognize", None
        yield "done", outcome, None

    _run.calls = []
    return _run


def write_files(tmp_path: Path, names: list[str]) -> list[Path]:
    files = []
    for i, name in enumerate(names):
        p = tmp_path / f"{i}_{name}"
        p.write_bytes(b"fake-image-" + str(i).encode())
        files.append(p)
    return files


class TestRunBatch:
    def test_aggregates_multiple_files(self, tmp_path, monkeypatch):
        states = [make_state("10001"), make_state("10002")]
        monkeypatch.setattr(flow_mod, "iter_run", fake_iter_run(states))
        files = write_files(tmp_path, ["a.png", "b.png"])
        batch = run_batch(files)
        assert batch.batch_id
        assert [it.invoice.invoice_number for it in batch.items] == ["10001", "10002"]
        assert batch.total_amount == pytest.approx(226.0)

    def test_failed_file_does_not_interrupt(self, tmp_path, monkeypatch):
        """单张识别失败（invoice 为 None）不中断整批。"""
        failed_state = {"invoice": None, "extract_error": "票面模糊", "answer": "降级说明"}
        states = [failed_state, make_state("10002")]
        monkeypatch.setattr(flow_mod, "iter_run", fake_iter_run(states))
        files = write_files(tmp_path, ["bad.png", "ok.png"])
        batch = run_batch(files)
        assert batch.items[0].invoice is None
        assert "票面模糊" in batch.items[0].error
        assert batch.items[1].invoice is not None
        assert batch.total_amount == pytest.approx(113.0)  # 失败张不计金额

    def test_rule_fallback_when_model_returns_other(self, tmp_path, monkeypatch):
        """模型输出「其他」时关键词规则兜底。"""
        inv = make_invoice("30001", seller_name="滴滴出行", expense_category=ExpenseCategory.OTHER)
        states = [{
            "invoice": inv, "verification": None, "duplicate_hits": [], "risk_findings": [],
        }]
        monkeypatch.setattr(flow_mod, "iter_run", fake_iter_run(states))
        files = write_files(tmp_path, ["taxi.png"])
        batch = run_batch(files)
        assert batch.items[0].category == ExpenseCategory.TRANSPORT
        assert batch.items[0].category_source == "rule"

    def test_model_category_trusted(self, tmp_path, monkeypatch):
        """模型明确输出非「其他」时直接采信（category_source=model）。"""
        inv = make_invoice("40001", seller_name="某公司", expense_category=ExpenseCategory.DINING)
        states = [{
            "invoice": inv, "verification": None, "duplicate_hits": [], "risk_findings": [],
        }]
        monkeypatch.setattr(flow_mod, "iter_run", fake_iter_run(states))
        files = write_files(tmp_path, ["meal.png"])
        batch = run_batch(files)
        assert batch.items[0].category == ExpenseCategory.DINING
        assert batch.items[0].category_source == "model"

    def test_risk_and_dup_flags_collected(self, tmp_path, monkeypatch):
        finding = Finding(rule_id="R001", rule_name="抬头不符", level=RiskLevel.HIGH, message="x")
        states = [{**make_state("50001"), "risk_findings": [finding], "duplicate_hits": [object()]}]
        monkeypatch.setattr(flow_mod, "iter_run", fake_iter_run(states))
        files = write_files(tmp_path, ["risky.png"])
        batch = run_batch(files)
        assert batch.items[0].risk_level == "高"
        assert batch.items[0].has_duplicates is True

    def test_progress_cb_called(self, tmp_path, monkeypatch):
        states = [make_state("60001")]
        monkeypatch.setattr(flow_mod, "iter_run", fake_iter_run(states))
        files = write_files(tmp_path, ["p.png"])
        events = []
        run_batch(files, progress_cb=lambda i, fname, node: events.append((i, fname, node)))
        assert events and events[0] == (1, "0_p.png", "recognize")

    def test_empty_files_raises(self):
        with pytest.raises(ValueError, match="未提供任何发票文件"):
            run_batch([])


class TestBuildSummary:
    def test_exclude_recalculates_totals(self):
        items = [
            BatchItem(file_name="a", file_sha256="s1", invoice=make_invoice("10001", 100.0), category=ExpenseCategory.DINING),
            BatchItem(file_name="b", file_sha256="s2", invoice=make_invoice("10002", 200.0), category=ExpenseCategory.DINING),
            BatchItem(file_name="c", file_sha256="s3", invoice=make_invoice("10003", 400.0), category=ExpenseCategory.TRANSPORT),
        ]
        batch = build_summary(items, exclude_numbers={"10001"})
        assert batch.total_amount == pytest.approx(600.0)
        cats = {t.category: t for t in batch.category_totals}
        assert cats[ExpenseCategory.DINING].count == 1
        assert cats[ExpenseCategory.DINING].total_amount == pytest.approx(200.0)
        assert cats[ExpenseCategory.TRANSPORT].count == 1

    def test_totals_follow_config_order(self):
        items = [
            BatchItem(file_name="a", file_sha256="s", invoice=make_invoice("1", 1.0), category=ExpenseCategory.OTHER),
            BatchItem(file_name="b", file_sha256="s", invoice=make_invoice("2", 2.0), category=ExpenseCategory.TRANSPORT),
        ]
        batch = build_summary(items)
        order = [t.category for t in batch.category_totals]
        assert order == [ExpenseCategory.TRANSPORT, ExpenseCategory.OTHER]  # config.yaml 顺序

    def test_summary_markdown_matches_totals(self):
        items = [
            BatchItem(file_name="a", file_sha256="s", invoice=make_invoice("1", 100.0), category=ExpenseCategory.DINING),
        ]
        batch = build_summary(items, batch_id="b1", title="T")
        md = batch.summary_markdown()
        assert "T" in md and "¥100.00" in md


class _FakeProvider:
    name = "fake"

    def __init__(self):
        self.submitted = None

    def submit(self, batch: ReimburseBatch) -> SubmissionResult:
        self.submitted = batch
        return SubmissionResult(
            status="submitted", instance_id="fake-123", provider=self.name, message="ok"
        )


class TestSubmitBatch:
    def test_submit_calls_provider_and_store(self, tmp_path):
        items = [
            BatchItem(file_name="a", file_sha256=hashlib.sha256(b"x").hexdigest(), invoice=make_invoice("10001"), category=ExpenseCategory.DINING),
        ]
        batch = build_summary(items, batch_id="b9", title="T")
        provider = _FakeProvider()
        store = DedupStore(tmp_path / "t.db")
        result = submit_batch(batch, provider=provider, store=store)
        assert result.instance_id == "fake-123"
        assert provider.submitted is batch
        assert store.get_batch("b9")["instance_id"] == "fake-123"
        # 逐张入台账（去重即刻生效）
        hits = store.find_duplicates("10001", "91330100MA28YYYYYY", "other-sha")
        assert len(hits) == 1
        assert store.batch_items("b9")[0]["expense_category"] == "餐饮"

    def test_submit_skips_failed_items_in_ledger(self, tmp_path):
        items = [BatchItem(file_name="bad", file_sha256="s", error="识别失败")]
        batch = build_summary(items, batch_id="b10", title="T")
        store = DedupStore(tmp_path / "t.db")
        submit_batch(batch, provider=_FakeProvider(), store=store)
        assert store.batch_items("b10") == []
