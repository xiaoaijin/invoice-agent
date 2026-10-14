"""recognize 节点：qwen-vl-max 结构化提取发票字段。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from invoice_agent.graph.state import AgentState
from invoice_agent.models.invoice import InvoiceData
from invoice_agent.services.llm import get_vision_model

SYSTEM_PROMPT = """你是增值税发票识别专家。请从图片中精确提取发票信息并输出为结构化数据。

字段说明与规则：
1. invoice_type：根据票面标题判断（增值税专用发票/普通发票/电子发票/数电票等）。
2. 数电票没有发票代码（invoice_code 填 null），发票号码为 20 位；其他票发票号码通常为 8 位。
3. 价税合计在票面右下角"价税合计（大写）…（小写）¥…"处；不含税金额在"合计"行；税额在"税额"行。三者不要混淆。
4. 普通发票抬头为"个人"时购买方税号填 null。
5. 缺失或无法辨认的字段一律填 null，**严禁猜测编造**。
6. 为每个关键字段给出 0~1 的置信度（field_confidence），整体置信度 overall_confidence 取关键字段的最小值。
7. 明细行可能跨页/多行，请全部列出。
8. 日期格式为 ISO（YYYY-MM-DD）。"""


def check_extract(invoice: InvoiceData) -> list[str]:
    """代码级校验（不靠模型）。返回错误列表，空列表 = 通过。"""
    errors: list[str] = []
    n = invoice.invoice_number.strip()
    if not (len(n) == 8 or len(n) == 20):
        errors.append(f"发票号码位数异常（{len(n)} 位，应为 8 或 20 位）")
    if invoice.total_amount <= 0:
        errors.append(f"价税合计异常（{invoice.total_amount}，应大于 0）")
    if not invoice.buyer_name:
        errors.append("缺少购买方名称")
    if not invoice.seller_name:
        errors.append("缺少销售方名称")
    return errors


def recognize(state: AgentState) -> dict:
    """调用多模态模型提取字段。失败/校验不过时返回 extract_error 驱动重试。"""
    page_images = state.get("page_images") or []
    if not page_images:
        return {}

    content = [{"type": "text", "text": "请识别这张发票并输出结构化数据。"}]
    for data_url in page_images:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    # 重试时把上次的校验错误反馈给模型
    prev_error = state.get("extract_error")
    if prev_error:
        content[0]["text"] += f"\n\n上一次提取存在以下问题，请特别注意：\n{prev_error}"

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)]
    try:
        invoice = get_vision_model().with_structured_output(InvoiceData).invoke(messages)
    except Exception as exc:  # noqa: BLE001 — 网络/解析错误统一走重试
        return {
            "extract_error": f"模型调用失败：{exc}",
            "invoice": None,
            "recognize_retry": state.get("recognize_retry", 0) + 1,
        }

    errors = check_extract(invoice)
    if errors:
        return {
            "invoice": None,
            "extract_error": "；".join(errors),
            "recognize_retry": state.get("recognize_retry", 0) + 1,
        }
    return {"invoice": invoice, "extract_error": None, "recognize_retry": 0}
