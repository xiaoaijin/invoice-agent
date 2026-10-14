from datetime import date
from pathlib import Path

import pytest

from invoice_agent.models.invoice import InvoiceData, InvoiceTypeEnum
from invoice_agent.services.dedup_store import DedupStore
from invoice_agent.services.verify_provider import MockInvoiceVerifyProvider


def make_invoice(**kw) -> InvoiceData:
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number="11112222",
        invoice_date=date(2026, 10, 1),
        buyer_name="示例科技有限公司",
        buyer_tax_id="91330100MA27XXXXXX",
        seller_name="卖方",
        seller_tax_id="91330100MA28YYYYYY",
        amount_without_tax=100.0,
        tax_amount=13.0,
        total_amount=113.0,
    )
    base.update(kw)
    return InvoiceData(**base)


@pytest.fixture
def provider(tmp_path: Path) -> MockInvoiceVerifyProvider:
    store = DedupStore(tmp_path / "test.db")
    store.set_verification("88887777", is_authentic=False, note="假票")
    store.set_verification("88886666", is_authentic=True, note="真票")
    return MockInvoiceVerifyProvider(store)


class TestMockVerify:
    def test_format_fail_bad_number(self, provider):
        r = provider.verify(make_invoice(invoice_number="123"))
        assert r.status == "FAIL"

    def test_format_fail_bad_taxid(self, provider):
        r = provider.verify(make_invoice(seller_tax_id="12345"))
        assert r.status == "FAIL"

    def test_registry_fake(self, provider):
        r = provider.verify(make_invoice(invoice_number="88887777"))
        assert r.status == "FAIL"

    def test_registry_authentic(self, provider):
        r = provider.verify(make_invoice(invoice_number="88886666"))
        assert r.status == "PASS"

    def test_hash_deterministic(self, provider):
        """同号结果可复现。"""
        a = provider.verify(make_invoice(invoice_number="12121212"))
        b = provider.verify(make_invoice(invoice_number="12121212"))
        assert a.status == b.status

    def test_provider_field(self, provider):
        assert provider.verify(make_invoice()).provider == "mock"
