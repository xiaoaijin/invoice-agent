"""费用类别关键词兜底分类（纯函数，离线可测）。

识票 prompt（recognize v3）会让视觉模型输出 expense_category；模型返回「其他」
或缺失时，在聚合层（services/reimburse_flow.py）用本规则修正，图结构零改动。
"""

from __future__ import annotations

from invoice_agent.models.invoice import ExpenseCategory, InvoiceData

# 关键词表：按 (类别, 关键词元组)；多个类别同时命中时取得分最高者，
# 同分按表内顺序（交通 > 餐饮 > 住宿 > 办公用品）。
RULES: list[tuple[ExpenseCategory, tuple[str, ...]]] = [
    (ExpenseCategory.TRANSPORT, (
        "出租车", "网约车", "滴滴", "铁路", "火车", "高铁", "客运", "航空", "机票", "航班",
        "停车", "加油", "充电", "过路", "过桥", "轮渡", "地铁", "公交", "出行",
    )),
    (ExpenseCategory.DINING, (
        "餐", "餐饮", "食堂", "外卖", "咖啡", "奶茶", "茶饮", "饮品", "烘焙", "面包", "快餐",
    )),
    (ExpenseCategory.LODGING, (
        "酒店", "住宿", "宾馆", "旅社", "民宿", "客栈",
    )),
    (ExpenseCategory.OFFICE, (
        "办公用品", "办公", "文具", "打印", "复印", "耗材", "纸张", "硒鼓", "墨盒",
        "会议", "快递", "邮费",
    )),
]


def infer_category(invoice: InvoiceData) -> tuple[ExpenseCategory, str]:
    """按销售方名称 + 明细品名 + 备注的关键词命中打分分类。

    返回 (类别, 来源)；有命中 → (类别, "rule")，无命中 → ("其他", "default")。
    """
    parts = [invoice.seller_name or "", invoice.remarks or ""]
    parts.extend(i.name for i in invoice.items)
    blob = " ".join(p for p in parts if p)
    if not blob:
        return ExpenseCategory.OTHER, "default"

    best: ExpenseCategory | None = None
    best_score = 0
    for category, keywords in RULES:
        score = sum(blob.count(k) for k in keywords)
        if score > best_score:
            best, best_score = category, score
    if best is None:
        return ExpenseCategory.OTHER, "default"
    return best, "rule"
