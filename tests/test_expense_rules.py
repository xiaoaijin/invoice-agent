from datetime import date

from invoice_agent.models.invoice import InvoiceData, InvoiceItem, InvoiceTypeEnum
from invoice_agent.models.reimburse import ExpenseCategory
from invoice_agent.rules.expense_category import infer_category


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
        items=[InvoiceItem(name="服务费")],
    )
    base.update(kw)
    return InvoiceData(**base)


class TestInferCategory:
    def test_transport_by_seller(self):
        cat, source = infer_category(make_invoice(seller_name="滴滴出行科技有限公司"))
        assert cat == ExpenseCategory.TRANSPORT
        assert source == "rule"

    def test_transport_by_item(self):
        inv = make_invoice(items=[InvoiceItem(name="客运服务费"), InvoiceItem(name="停车费")])
        cat, _ = infer_category(inv)
        assert cat == ExpenseCategory.TRANSPORT

    def test_dining_by_item(self):
        inv = make_invoice(items=[InvoiceItem(name="餐费")])
        cat, source = infer_category(inv)
        assert (cat, source) == (ExpenseCategory.DINING, "rule")

    def test_lodging_by_seller(self):
        cat, _ = infer_category(make_invoice(seller_name="如家酒店集团"))
        assert cat == ExpenseCategory.LODGING

    def test_office_by_item(self):
        inv = make_invoice(items=[InvoiceItem(name="办公用品（复印纸）")])
        cat, _ = infer_category(inv)
        assert cat == ExpenseCategory.OFFICE

    def test_other_when_no_hit(self):
        inv = make_invoice(seller_name="某咨询服务有限公司", items=[InvoiceItem(name="咨询服务费")])
        cat, source = infer_category(inv)
        assert (cat, source) == (ExpenseCategory.OTHER, "default")

    def test_conflict_takes_higher_score(self):
        """交通与餐饮同时命中时取得分最高者。"""
        inv = make_invoice(
            seller_name="某酒店集团", items=[InvoiceItem(name="住宿费"), InvoiceItem(name="住宿费")]
        )
        cat, _ = infer_category(inv)
        assert cat == ExpenseCategory.LODGING

    def test_tie_breaks_by_rule_order(self):
        """同分时按表内顺序（交通优先）。"""
        inv = make_invoice(seller_name="地铁口宾馆")  # "地铁"（交通）与 "宾馆"（住宿）各 1 分
        cat, _ = infer_category(inv)
        assert cat == ExpenseCategory.TRANSPORT

    def test_remarks_considered(self):
        inv = make_invoice(remarks="出差住宿")
        cat, _ = infer_category(inv)
        assert cat == ExpenseCategory.LODGING

    def test_empty_texts_defaults(self):
        inv = make_invoice(seller_name="", remarks="", items=[])
        cat, source = infer_category(inv)
        assert (cat, source) == (ExpenseCategory.OTHER, "default")
