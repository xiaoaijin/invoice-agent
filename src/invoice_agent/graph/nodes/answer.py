"""retrieve_docs / finalize 节点。

finalize 职责：
1. 上下文管理：历史超预算 60% 时压缩（旧对话摘要化、最近 N 轮原样保留）
2. 组装 prompt（摘要 + 相关长期记忆 + 历史消息 + 本轮上下文），并做预算兜底截断
3. 汇总发票/验真/去重/风控/RAG 引用，生成固定结构 Markdown 回答
4. 回答后把本轮问答与工具结果写入记忆（pgvector），摘要持久化
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from invoice_agent.config import get_app_config
from invoice_agent.graph.state import AgentState
from invoice_agent.prompts.registry import use
from invoice_agent.services.context_manager import (
    compress_history,
    default_summarizer,
    estimate_tokens,
    fit_budget,
)
from invoice_agent.services.llm import get_chat_model
from invoice_agent.services.rag.qa import format_chunks_for_prompt
from invoice_agent.services.rag.retriever import get_retriever

logger = logging.getLogger(__name__)


def _build_query(state: AgentState) -> str:
    parts = []
    invoice = state.get("invoice")
    if invoice is not None:
        parts.append(invoice.summary_text())
    question = state.get("user_question")
    if question:
        prev = None
        for entry in reversed(state.get("chat_history") or []):
            if entry.get("role") == "user":
                prev = entry.get("content")
                break
        if prev and prev != question and _is_follow_up(question):
            parts.append(f"用户上一问题：{prev}")
        parts.append(f"用户问题：{question}")
    if not parts:
        parts.append("发票报销相关规定")
    return "；".join(parts)


def _is_follow_up(question: str) -> bool:
    """启发式：短问题或以追问词开头，视为依赖上文。"""
    q = question.strip()
    return len(q) <= 15 or q.startswith(
        ("那", "它", "这个", "这批", "还有", "为什么", "怎么", "如上", "上述")
    )


def retrieve_docs(state: AgentState) -> dict:
    try:
        retriever = get_retriever()
    except RuntimeError as exc:  # PG 不可用 / 索引未建
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


def _history_messages(history: list[dict]) -> list:
    """历史条目 → LangChain 消息。工具条目渲染为一行「[工具 x] 结果」。"""
    messages: list = []
    for entry in history:
        if entry.get("role") == "tool":
            messages.append(
                HumanMessage(content=f"[工具 {entry.get('tool', '?')}] {entry.get('content', '')}")
            )
        elif entry.get("role") == "user":
            messages.append(HumanMessage(content=entry.get("content", "")))
        else:
            messages.append(AIMessage(content=entry.get("content", "")))
    return messages


def _remember(
    state: AgentState, question: str | None, answer: str, base: list[dict] | None = None
) -> list[dict]:
    """本轮问答 + 工具结果写入历史（返回完整新列表，checkpointer 按覆盖合并）。"""
    history = list(base if base is not None else (state.get("chat_history") or []))
    if question:
        history.append({"role": "user", "content": question})
    # 工具调用只保留工具名和调用结果
    verification = state.get("verification")
    if verification is not None:
        history.append(
            {
                "role": "tool",
                "tool": "invoice_verify",
                "content": f"{verification.status}: {verification.message}",
            }
        )
    findings = state.get("risk_findings") or []
    if findings or state.get("invoice") is not None:
        tool_summary = (
            "; ".join(f"{f.rule_id}({f.level.value})" for f in findings) if findings else "无风险"
        )
        history.append({"role": "tool", "tool": "risk_rules", "content": tool_summary})
    if answer:
        history.append({"role": "assistant", "content": answer})
    return history


def finalize(state: AgentState) -> dict:
    invoice = state.get("invoice")
    cfg = get_app_config()

    # ---- 上下文压缩（历史超预算 60% 时触发）----
    compression = compress_history(state, default_summarizer(), cfg.context)
    history: list[dict] = compression.get("chat_history") or list(state.get("chat_history") or [])
    summary = compression.get("history_summary") or state.get("history_summary") or ""

    # 识票路径但提取失败且重试耗尽 → 降级人工复核（不走 LLM）
    if invoice is None and state.get("file_path"):
        answer = _fallback_answer(state)
        return {
            "answer": answer,
            "chat_history": _remember(state, state.get("user_question"), answer),
            "history_summary": summary,
        }
    # 兜底：无票无问题
    if invoice is None and not state.get("user_question"):
        return {
            "answer": "请上传发票图片/PDF 或输入政策问题。",
            "chat_history": history,
            "history_summary": summary,
        }

    # ---- 长期记忆（pgvector）：语义召回相关历史问答 ----
    recalled: list[dict] = []
    if state.get("user_question"):
        try:
            from invoice_agent.services.memory_store import get_memory_store

            recalled = get_memory_store().recall(
                state["user_question"], top_k=cfg.memory.recall_top_k
            )
        except Exception as exc:  # noqa: BLE001 — 记忆是增强项，失败不阻塞
            logger.warning("长期记忆召回失败（忽略）：%s", exc)

    # ---- 组装 prompt ----
    context_parts = []
    if summary:
        context_parts.append(f"## 早期对话摘要\n{summary}")
    if recalled:
        mem_lines = [f"- {m['content'][:300]}（相关度 {m['score']:.2f}）" for m in recalled]
        context_parts.append("## 相关历史记忆（语义召回）\n" + "\n".join(mem_lines))
    if invoice is not None:
        context_parts.append(_invoice_block(state, invoice))
    chunks = state.get("rag_chunks") or []
    context_parts.append("## 参考条款（检索结果）\n\n" + format_chunks_for_prompt(chunks))
    if state.get("user_question"):
        context_parts.append(f"## 用户问题\n{state['user_question']}")
    # 自反馈：自评不通过时带上改进提示重生成（有界，见 critic_answer/route_after_critique）
    critic_fb = state.get("critic_feedback")
    if critic_fb:
        context_parts.append(f"## 上一版回答的不足（请针对性改进，其余内容保持）\n{critic_fb}")

    prompt_text = "\n\n".join(context_parts) + (
        "\n\n请按以下结构输出 Markdown：发票概览 / 验真结论 / 重复性 / 风险提示 / 政策依据（[来源N] 引用列表）。"
        '无发票时只回答政策问题。结合对话历史理解追问（如"那出租车票呢"指上一问题中的票据类型）。'
    )
    # 预算兜底：完整 prompt 超上限时从头截断
    prompt_text = fit_budget(prompt_text, cfg.context.token_budget)

    # prompt 版本管理：use() 解析 qa 当前版本并登记 contextvar（监控据此落库），
    # A/B 抽样按 session_id 粘性分桶（finalize 重生成轮不换 prompt）
    with use("qa", sticky_key=state.get("session_id")) as qa_prompt:
        messages = [
            SystemMessage(content=qa_prompt.text),
            *_history_messages(history),
            HumanMessage(content=prompt_text),
        ]
        try:
            answer = get_chat_model().invoke(messages).content
        except Exception as exc:  # noqa: BLE001
            answer = _fallback_answer(state) + f"\n\n（LLM 汇总失败：{exc}）"

    # ---- 记忆写入（pgvector，尽力而为）----
    # 自评未通过时跳过：中间版答案不污染长期记忆，最终通过版的重生成轮会写入
    critique = state.get("critique")
    critique_failed = bool(critique) and not critique.get("passed", True)
    question = state.get("user_question")
    if critic_fb:
        # 重生成轮：回退上一轮 finalize 追加的条目（从最后一条 user 起），避免历史重复
        while history and history[-1].get("role") != "user":
            history.pop()
        if history and history[-1].get("role") == "user":
            history.pop()
    new_history = _remember(state, question, answer, base=history)
    if not critique_failed:
        try:
            from invoice_agent.services.memory_store import get_memory_store

            memory = get_memory_store()
            if question and answer:
                memory.record_turn(state.get("session_id", "default"), question, answer)
            if compression.get("history_summary"):
                memory.save_summary(
                    state.get("session_id", "default"), compression["history_summary"]
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆写入失败（忽略）：%s", exc)

    return {"answer": answer, "chat_history": new_history, "history_summary": summary}


def _invoice_block(state: AgentState, invoice) -> str:
    verification = state.get("verification")
    findings = state.get("risk_findings") or []
    dups = state.get("duplicate_hits") or []
    return f"""## 发票结构化数据
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


# estimate_tokens 暴露给测试/调试
__all__ = ["estimate_tokens", "finalize", "retrieve_docs"]
