from datetime import date

import pytest

from invoice_agent.models.invoice import InvoiceData, InvoiceTypeEnum
from invoice_agent.models.reimburse import BatchItem, ExpenseCategory, ReimburseBatch
from invoice_agent.services.expense_flow_provider import (
    MockExpenseFlowProvider,
    get_expense_flow_provider,
)


def make_batch(batch_id: str = "batch-abc") -> ReimburseBatch:
    invoice = InvoiceData(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_number="11112222",
        invoice_date=date(2026, 10, 1),
        buyer_name="买方",
        seller_name="卖方",
        total_amount=113.0,
    )
    return ReimburseBatch(
        batch_id=batch_id,
        title="测试报销",
        items=[
            BatchItem(file_name="a.png", file_sha256="sha", invoice=invoice, category=ExpenseCategory.OTHER)
        ],
        total_amount=113.0,
    )


class TestMockProvider:
    def test_submit_success(self):
        r = MockExpenseFlowProvider().submit(make_batch())
        assert r.status == "submitted"
        assert r.instance_id and r.instance_id.startswith("mock-")
        assert r.provider == "mock"

    def test_instance_id_deterministic(self):
        """同 batch_id 结果可复现。"""
        a = MockExpenseFlowProvider().submit(make_batch("same-id"))
        b = MockExpenseFlowProvider().submit(make_batch("same-id"))
        assert a.instance_id == b.instance_id

    def test_instance_id_differs_across_batches(self):
        a = MockExpenseFlowProvider().submit(make_batch("id-1"))
        b = MockExpenseFlowProvider().submit(make_batch("id-2"))
        assert a.instance_id != b.instance_id


class TestFactory:
    def test_default_is_mock(self):
        assert isinstance(get_expense_flow_provider(), MockExpenseFlowProvider)

    def test_unknown_provider_raises(self, monkeypatch):
        from types import SimpleNamespace

        import invoice_agent.services.expense_flow_provider as mod

        monkeypatch.setattr(mod, "get_settings", lambda: SimpleNamespace(expense_flow_provider="nonsense"))
        with pytest.raises(ValueError, match="未知报销流程提供方"):
            get_expense_flow_provider()

    def test_feishu_provider_not_implemented_yet(self, monkeypatch):
        """feishu 分支在真实 provider 落地前应报 ImportError（延迟导入触发）。"""
        from types import SimpleNamespace

        import invoice_agent.services.expense_flow_provider as mod

        monkeypatch.setattr(mod, "get_settings", lambda: SimpleNamespace(expense_flow_provider="feishu"))
        with pytest.raises(ImportError):
            get_expense_flow_provider()
