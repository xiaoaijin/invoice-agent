"""retrieve_docs / finalize 节点。

finalize 汇总发票信息/验真/去重/风控/RAG 引用，生成固定结构的 Markdown 回答。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from invoice_agent.graph.state import AgentState
from invoice_agent.services.llm import get_chat_model
from invoice_agent.services.rag.qa import QA_SYSTEM_PROMPT, format_chunks_for_prompt
from invoice_agent.services.rag.retriever import get_retriever

MAX_HISTORY_MESSAGES = 8  # 送入 LLM 的历史轮数上限（含双方）

# 疑似追问的开头词：检索 query 需要拼上前一个问题补全语义
FOLLOW_UP_PREFIXES = ("那", "它", "这个", "这批", "还有", "为什么", "怎么", "如上", "上述")


def _last_user_question(state: AgentState) -> str | None:
    for entry in reversed(state.get("chat_history") or []):
        if entry.get("role") == "user":
            return entry.get("content")
    return None


def _is_follow_up(question: str) -> bool:
    """启发式：短问题或以追问词开头，视为依赖上文。"""
    q = question.strip()
    return len(q) <= 15 or q.startswith(FOLLOW_UP_PREFIXES)


def _build_query(state: AgentState) -> str:
    parts = []
    invoice = state.get("invoice")
    if invoice is not None:
        parts.append(invoice.summary_text())
    question = state.get("user_question")
    if question:
        prev = _last_user_question(state)
        if prev and prev != question and _is_follow_up(question):
            parts.append(f"用户上一问题：{prev}")
        parts.append(f"用户问题：{question}")
    if not parts:
        parts.append("发票报销相关规定")
    return "；".join(parts)


def retrieve_docs(state: AgentState) -> dict:
    try:
        retriever = get_retriever()
    except RuntimeError as exc:  # 索引未建
        return {"rag_chunks": [], "extract_error": str(exc)}
    return {"rag_chunks": retriever.retrieve(_build_query(state))}


def _fallback_answer(state: AgentState) -> str:
    """无 LLM 可用/提取失败时的兜底汇总（降级人工复核路径）。"""
    lines = ["## ⚠️ 人工复核提示\n", "发票识别未通过自动校验，以下为系统已获取的信息：\n"]
    if state.get("invoice") is not None:
        lines.append(f"- 已识别发票号码：{state['invoice'].invoice_number}")
    if state.get("extract_error"):
        lines.append(f"- 问题：{state['extract_error']}")
    lines.append("\n请人工核对发票后重试，或更换更清晰的图片。")
    return "\n".join(lines)


def _history_messages(state: AgentState) -> list:
    """最近若干轮对话 → LangChain 消息（不含当前轮）。"""
    history = state.get("chat_history") or []
    # 当前问题已在 state 中，历史里最后一条 user 若与之相同则跳过（防重放）
    recent = history[-MAX_HISTORY_MESSAGES:]
    messages: list = []
    for entry in recent:
        if entry.get("role") == "user":
            messages.append(HumanMessage(content=entry.get("content", "")))
        else:
            messages.append(AIMessage(content=entry.get("content", "")))
    return messages


def _remember(state: AgentState, question: str | None, answer: str) -> list[dict]:
    """把本轮问答追加进对话历史（返回完整新列表，checkpointer 按覆盖合并）。"""
    history = list(state.get("chat_history") or [])
    if question:
        history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history


def finalize(state: AgentState) -> dict:
    invoice = state.get("invoice")
    # 识票路径但提取失败且重试耗尽 → 降级人工复核（不走 LLM）
    if invoice is None and state.get("file_path"):
        answer = _fallback_answer(state)
        return {"answer": answer, "chat_history": _remember(state, state.get("user_question"), answer)}
    # 兜底：无票无问题
    if invoice is None and not state.get("user_question"):
        return {"answer": "请上传发票图片/PDF 或输入政策问题。"}

    context_parts = []
    if invoice is not None:
        verification = state.get("verification")
        findings = state.get("risk_findings") or []
        dups = state.get("duplicate_hits") or []
        invoice_block = f"""## 发票结构化数据
```json
{invoice.model_dump_json(indent=2)}
```

## 验真结果
- 状态：{verification.status if verification else "未执行"}
- 提供方：{verification.provider if verification else "-"}（注意：当前为 Mock 验真，未接入真实查验平台）
- 说明：{verification.message if verification else "-"}

## 重复报销检查
{("发现重复：" + "; ".join(f"{d.invoice_number}（{d.status}，首次提交 {d.first_submitted_at:%Y-%m-%d %H:%M}）" for d in dups)) if dups else "未发现重复提交记录。"}

## 风控预警
{chr(10).join(f"- [{f.level.value}] {f.rule_id} {f.rule_name}：{f.message}" for f in findings) if findings else "未触发风控规则。"}"""
        context_parts.append(invoice_block)

    chunks = state.get("rag_chunks") or []
    context_parts.append("## 参考条款（检索结果）\n\n" + format_chunks_for_prompt(chunks))

    if state.get("user_question"):
        context_parts.append(f"## 用户问题\n{state['user_question']}")

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        *_history_messages(state),
        HumanMessage(
            content="\n\n".join(context_parts)
            + "\n\n请按以下结构输出 Markdown：发票概览 / 验真结论 / 重复性 / 风险提示 / 政策依据（[来源N] 引用列表）。无发票时只回答政策问题。结合对话历史理解追问（如\"那出租车票呢\"指上一问题中的票据类型）。"
        ),
    ]
    try:
        answer = get_chat_model().invoke(messages).content
    except Exception as exc:  # noqa: BLE001
        answer = _fallback_answer(state) + f"\n\n（LLM 汇总失败：{exc}）"
    return {"answer": answer, "chat_history": _remember(state, state.get("user_question"), answer)}
