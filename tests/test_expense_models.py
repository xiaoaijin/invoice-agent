from datetime import date

import pytest

from invoice_agent.config import load_app_config
from invoice_agent.models.invoice import InvoiceData, InvoiceTypeEnum
from invoice_agent.models.reimburse import (
    BatchItem,
    CategoryTotal,
    ExpenseCategory,
    ReimburseBatch,
)


def make_item(number: str, amount: float, category: ExpenseCategory) -> BatchItem:
    invoice = InvoiceData(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number=number,
        invoice_date=date(2026, 10, 1),
        buyer_name="示例科技有限公司",
        seller_name="卖方",
        total_amount=amount,
        expense_category=category,
    )
    return BatchItem(
        file_name=f"{number}.png",
        file_sha256=f"sha_{number}",
        invoice=invoice,
        category=category,
        category_source="model",
    )


class TestReimburseBatch:
    def test_summary_markdown_table(self):
        batch = ReimburseBatch(
            batch_id="b1",
            title="测试批次",
            items=[make_item("10001", 100.0, ExpenseCategory.DINING)],
            category_totals=[CategoryTotal(category=ExpenseCategory.DINING, count=1, total_amount=100.0)],
            total_amount=100.0,
        )
        md = batch.summary_markdown()
        assert "测试批次" in md
        assert "餐饮" in md
        assert "¥100.00" in md

    def test_empty_batch_totals_zero(self):
        batch = ReimburseBatch(batch_id="b2", title="空批次")
        assert batch.total_amount == 0.0
        assert batch.category_totals == []

    def test_failed_item_without_invoice(self):
        item = BatchItem(file_name="bad.png", file_sha256="sha", error="识别失败")
        assert item.invoice is None
        assert item.category == ExpenseCategory.OTHER


class TestConfigConsistency:
    def test_config_categories_match_enum(self):
        """config.reimburse.categories 必须与 ExpenseCategory 枚举值一致。"""
        configured = load_app_config().reimburse.categories
        assert sorted(configured) == sorted(c.value for c in ExpenseCategory)

    def test_config_categories_valid_enum_values(self):
        for c in load_app_config().reimburse.categories:
            assert ExpenseCategory(c)  # 非法值直接抛 ValueError

    def test_settings_default_provider_mock(self):
        from invoice_agent.config import Settings

        assert Settings().expense_flow_provider == "mock"


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({"expense_category": "交通"}, ExpenseCategory.TRANSPORT),
        ({"expense_category": "其他"}, ExpenseCategory.OTHER),
    ],
)
def test_invoice_expense_category_field(kw, expected):
    """InvoiceData.expense_category 接受枚举值字符串（str Enum），缺省为其他。"""
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_number="11112222",
        invoice_date=date(2026, 10, 1),
        buyer_name="买方",
        seller_name="卖方",
        total_amount=113.0,
    )
    inv = InvoiceData(**base, **kw) if kw else InvoiceData(**base)
    assert inv.expense_category == expected


def test_invoice_expense_category_default():
    inv = InvoiceData(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_number="11112222",
        invoice_date=date(2026, 10, 1),
        buyer_name="买方",
        seller_name="卖方",
        total_amount=113.0,
    )
    assert inv.expense_category == ExpenseCategory.OTHER
