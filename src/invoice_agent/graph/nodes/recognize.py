"""recognize 节点：qwen-vl-max 结构化提取发票字段。"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from invoice_agent.config import get_app_config
from invoice_agent.feedback.critics import default_llm_fn, field_self_check
from invoice_agent.graph.state import AgentState
from invoice_agent.models.invoice import InvoiceData
from invoice_agent.prompts.registry import use
from invoice_agent.services.llm import get_vision_model


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

    content = [{"type": "text", "text": "请识别以上发票图片并输出结构化数据。"}]
    for data_url in page_images:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    prev_error = state.get("extract_error")

    # prompt 版本管理：use() 解析当前版本并登记 contextvar（监控据此落库），
    # with 块包住 LLM invoke，A/B 抽样按 session_id 粘性分桶
    with use("recognize", sticky_key=state.get("session_id")) as prompt:
        system_text = prompt.text
        if "{retry_feedback}" in system_text:
            # 参数化模板（v2+）：当前日期与重试反馈填入 system prompt，human 消息只带图片与触发语。
            # 用 replace 而非 str.format：模板内花括号示例无需转义。
            system_text = (
                system_text.replace("{current_date}", date.today().isoformat())
                .replace("{retry_feedback}", prev_error or "（无，首次提取）")
            )
        elif prev_error:
            # 旧版非参数化 prompt（v1）：重试反馈走 human 消息
            content[0]["text"] += f"\n\n上一次提取存在以下问题，请特别注意：\n{prev_error}"
        messages = [SystemMessage(content=system_text), HumanMessage(content=content)]
        try:
            invoice = get_vision_model().with_structured_output(InvoiceData).invoke(messages)
        except Exception as exc:  # noqa: BLE001 — 网络/解析错误统一走重试
            return {
                "extract_error": f"模型调用失败：{exc}",
                "invoice": None,
                "recognize_retry": state.get("recognize_retry", 0) + 1,
            }

    errors = check_extract(invoice)
    # 自反馈：代码级校验通过后再做 LLM 级字段自检（缺失/逻辑矛盾），
    # 产出更精准的重试提示，复用既有 extract_error 重试边（有界 retry<2）。
    if not errors and get_app_config().feedback.field_check.enabled:
        try:
            errors = field_self_check(invoice, default_llm_fn())
        except Exception:  # noqa: BLE001 — 自检是增强项，失败不阻塞
            errors = []
    if errors:
        return {
            "invoice": None,
            "extract_error": "；".join(errors),
            "recognize_retry": state.get("recognize_retry", 0) + 1,
        }
    return {"invoice": invoice, "extract_error": None, "recognize_retry": 0}
