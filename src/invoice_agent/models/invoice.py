"""发票数据模型。

设计原则：字段允许 None + 置信度，缺失信息交给规则层与最终回答呈现，禁止模型编造。
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class InvoiceTypeEnum(str, Enum):
    SPECIAL = "增值税专用发票"
    NORMAL = "增值税普通发票"
    E_SPECIAL = "增值税电子专用发票"
    E_NORMAL = "增值税电子普通发票"
    DIGITAL = "数电票"
    UNKNOWN = "未知/无法识别"


class ExpenseCategory(str, Enum):
    """费用类别（recognize v3 由视觉模型输出，关键词规则在聚合层兜底）。"""

    TRANSPORT = "交通"
    DINING = "餐饮"
    LODGING = "住宿"
    OFFICE = "办公用品"
    OTHER = "其他"


class InvoiceItem(BaseModel):
    name: str
    specification: str | None = None
    quantity: float | None = None
    unit: str | None = None
    amount: float | None = None  # 不含税金额
    tax_rate: str | None = None  # 保留字符串（如 "13%"）避免浮点歧义


class InvoiceData(BaseModel):
    invoice_type: InvoiceTypeEnum
    invoice_code: str | None = None  # 数电票无发票代码，允许 None
    invoice_number: str
    invoice_date: date
    buyer_name: str
    buyer_tax_id: str | None = None  # 普票抬头给个人时可能无税号
    seller_name: str
    seller_tax_id: str | None = None
    amount_without_tax: float | None = None
    tax_amount: float | None = None
    total_amount: float  # 价税合计
    items: list[InvoiceItem] = Field(default_factory=list)
    remarks: str | None = None
    # 费用类别：v2 及以下 prompt 不输出该字段时填默认值，由规则层兜底修正
    expense_category: ExpenseCategory = ExpenseCategory.OTHER
    # 模型自评置信度，驱动"人工复核"提示
    field_confidence: dict[str, float] = Field(default_factory=dict)
    overall_confidence: float = 0.0

    @model_validator(mode="after")
    def check_totals(self) -> InvoiceData:
        """金额勾稽软校验：超出容差仅标记，不抛错（交给 R008 呈现）。"""
        TOLERANCE = 0.05
        if (
            self.amount_without_tax is not None
            and self.tax_amount is not None
            and abs(self.amount_without_tax + self.tax_amount - self.total_amount) > TOLERANCE
        ):
            self.field_confidence["total_amount"] = min(
                self.field_confidence.get("total_amount", 1.0), 0.5
            )
        return self

    def summary_text(self) -> str:
        """生成用于检索 query 拼接的发票摘要。"""
        item_names = "、".join(i.name for i in self.items) or "无明细"
        return (
            f"发票类型：{self.invoice_type.value}；"
            f"购买方：{self.buyer_name}；销售方：{self.seller_name}；"
            f"商品/服务：{item_names}；价税合计：{self.total_amount:.2f} 元"
        )
