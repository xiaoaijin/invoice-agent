"""报销批次数据模型：费用类别、分类统计、批量明细与提交结果。

ExpenseCategory 定义在 models/invoice.py（InvoiceData 的字段枚举），此处 re-export，
保持「报销相关模型从 models.reimburse 导入」的单一入口，且不产生循环导入。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from invoice_agent.models.invoice import ExpenseCategory, InvoiceData

__all__ = [
    "BatchItem",
    "CategoryTotal",
    "ExpenseCategory",
    "ReimburseBatch",
    "SubmissionResult",
]


class CategoryTotal(BaseModel):
    """单个费用类别的汇总。"""

    category: ExpenseCategory
    count: int
    total_amount: float


class SubmissionResult(BaseModel):
    """审批流程提交结果。"""

    status: str  # "submitted"（mock 恒为 submitted）
    instance_id: str | None  # 飞书审批实例 ID（mock 为派生的假 ID）
    provider: str  # "mock" / "feishu"
    message: str
    submitted_at: datetime = Field(default_factory=datetime.now)


class BatchItem(BaseModel):
    """批次内的单张发票条目（识别失败也保留条目，便于 UI 展示）。"""

    file_name: str
    file_sha256: str
    invoice: InvoiceData | None = None
    category: ExpenseCategory = ExpenseCategory.OTHER
    category_source: str = "default"  # "model"（识票模型输出）/ "rule"（关键词兜底）/ "default"
    verification_status: str | None = None  # PASS / FAIL / UNKNOWN
    has_duplicates: bool = False
    risk_level: str | None = None  # 最高风险级（高/中/低），无则 None
    error: str | None = None  # 该张识别失败原因


class ReimburseBatch(BaseModel):
    """一次报销批次：明细 + 分类统计。"""

    batch_id: str
    title: str
    items: list[BatchItem] = Field(default_factory=list)
    category_totals: list[CategoryTotal] = Field(default_factory=list)
    total_amount: float = 0.0

    def summary_markdown(self) -> str:
        """分类统计的 Markdown 表格，供 UI 展示与提交备注复用。"""
        lines = [
            f"**{self.title}**（{len(self.items)} 张，合计 ¥{self.total_amount:,.2f}）",
            "",
            "| 类别 | 张数 | 金额小计 |",
            "| --- | --- | --- |",
        ]
        for t in self.category_totals:
            lines.append(f"| {t.category.value} | {t.count} | ¥{t.total_amount:,.2f} |")
        return "\n".join(lines)
