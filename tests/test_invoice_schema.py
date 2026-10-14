from datetime import date

import pytest
from pydantic import ValidationError

from invoice_agent.models.invoice import InvoiceData, InvoiceItem, InvoiceTypeEnum


def make_invoice(**overrides) -> dict:
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number="28374651",
        invoice_date=date(2026, 9, 30),
        buyer_name="示例科技有限公司",
        buyer_tax_id="91330100MA27XXXXXX",
        seller_name="杭州某某商贸有限公司",
        seller_tax_id="91330100MA28YYYYYY",
        amount_without_tax=1000.0,
        tax_amount=130.0,
        total_amount=1130.0,
        items=[InvoiceItem(name="办公用纸", amount=1000.0, tax_rate="13%")],
    )
    base.update(overrides)
    return base


class TestInvoiceSchema:
    def test_normal_invoice(self):
        inv = InvoiceData(**make_invoice())
        assert inv.invoice_number == "28374651"
        assert inv.items[0].name == "办公用纸"

    def test_digital_invoice_without_code(self):
        """数电票无发票代码、20 位号码，应允许通过。"""
        inv = InvoiceData(
            **make_invoice(
                invoice_type=InvoiceTypeEnum.DIGITAL,
                invoice_code=None,
                invoice_number="25612000000123456789",
            )
        )
        assert inv.invoice_code is None
        assert len(inv.invoice_number) == 20

    def test_personal_buyer_without_tax_id(self):
        inv = InvoiceData(
            **make_invoice(
                invoice_type=InvoiceTypeEnum.NORMAL,
                buyer_name="个人",
                buyer_tax_id=None,
            )
        )
        assert inv.buyer_tax_id is None

    def test_missing_required_field_raises(self):
        data = make_invoice()
        del data["invoice_number"]
        with pytest.raises(ValidationError):
            InvoiceData(**data)

    def test_amount_mismatch_lowers_confidence_not_raise(self):
        """金额勾稽不符只降置信度，不抛错。"""
        inv = InvoiceData(**make_invoice(total_amount=9999.0))
        assert inv.field_confidence.get("total_amount", 1.0) <= 0.5

    def test_amount_match_keeps_confidence(self):
        inv = InvoiceData(**make_invoice(), field_confidence={"total_amount": 0.98})
        assert inv.field_confidence["total_amount"] == 0.98

    def test_summary_text(self):
        inv = InvoiceData(**make_invoice())
        s = inv.summary_text()
        assert "增值税专用发票" in s
        assert "办公用纸" in s
