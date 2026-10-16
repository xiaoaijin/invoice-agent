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
from invoice_agent.services.lesson_store import get_lesson_store
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
        # 追问判定：优先用意图漏斗结果；state 无意图字段（直测/旧线程）时退回启发式
        if state.get("intent") is not None:
            is_follow_up = (state["intent"] or {}).get("retrieval_type") == "follow_up"
        else:
            is_follow_up = _is_follow_up(question)
        if prev and prev != question and is_follow_up:
            parts.append(f"用户上一问题：{prev}")
        parts.append(f"用户问题：{question}")
    if not parts:
        parts.append("发票报销相关规定")
    query = "；".join(parts)
    # 历史教训召回：相似问题曾因检索不足被自评点名，本次检索提前补相关方向
    if question:
        try:
            lcfg = get_app_config().feedback.lessons
            lessons = get_lesson_store().recall(
                question, kind="retrieval", top_k=lcfg.recall_top_k, min_score=lcfg.min_score
            )
            if lessons:
                query += "；注意补充检索：" + "；".join(l["content"][:60] for l in lessons)
        except Exception as exc:  # noqa: BLE001 — 教训是增强项，失败不影响检索
            logger.warning("检索教训召回失败（忽略）：%s", exc)
    return query


def _is_follow_up(question: str) -> bool:
    """启发式：短问题或以追问词开头，视为依赖上文。"""
    q = question.strip()
    return len(q) <= 15 or q.startswith(
        ("那", "它", "这个", "这批", "还有", "为什么", "怎么", "如上", "上述")
    )


def _rewrite_query(query: str, state: AgentState) -> str:
    """追问时用 LLM 结合对话历史改写查询、消解指代；LLM 不可用/失败降级为原查询。

    改写已前置到首次检索（意图驱动触发）。降级时原查询本身已含上一轮问题拼接，
    消解能力不打折，仅少了 LLM 精炼。
    """
    try:
        with use("query_rewrite") as prompt:
            recent = (
                "\n".join(
                    f"{'用户' if e.get('role') == 'user' else '助手'}：{e.get('content', '')}"
                    for e in (state.get("chat_history") or [])[-4:]  # 最近 2 轮
                )
                or "（无）"
            )
            domain = (state.get("intent") or {}).get("domain", "") or "（未知）"
            system_text = prompt.text
            if "{query}" in system_text:
                # 参数化模板（v3+）：占位符填入 system prompt，human 消息仅作触发。
                # 用 replace 而非 str.format：模板内样例文本无需转义。
                system_text = (
                    system_text.replace("{chat_history}", recent)
                    .replace("{domain}", domain)
                    .replace("{query}", query)
                )
                user_content = "请输出消解指代后的完整检索查询，一行以内，不要解释。"
            else:
                # 旧版非参数化 prompt（v1/v2）：对话与查询仍走 human 消息
                user_content = (
                    f"## 最近对话\n{recent}\n\n"
                    f"## 业务领域\n{domain}\n\n"
                    f"## 待改写的检索查询\n{query}"
                )
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=user_content),
            ]
            rewritten = get_chat_model().invoke(messages).content.strip()
            return rewritten or query
    except Exception as exc:  # noqa: BLE001 — 含 PromptNotFoundError（fail-fast prompt 缺失不阻塞检索）
        logger.warning("查询改写失败（降级为原查询）：%s", exc)
        return query


def retrieve_docs(state: AgentState) -> dict:
    # 检索是增强项：索引未建（RuntimeError）或 embedding 网络故障均降级为空结果，不阻塞对话
    try:
        retriever = get_retriever()
        query = _build_query(state)
        # 意图驱动改写：仅追问触发 LLM 消解指代；新问题直接用拼接 query
        if (state.get("intent") or {}).get("retrieval_type") == "follow_up":
            query = _rewrite_query(query, state)
        return {"rag_chunks": retriever.retrieve(query)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("政策检索失败（降级为无引用回答）：%s", exc)
        return {"rag_chunks": [], "extract_error": str(exc)}


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
    chunks = state.get("rag_chunks") or []
    # prompt 版本管理：use() 解析 qa 当前版本并登记 contextvar（监控据此落库），
    # A/B 抽样按 session_id 粘性分桶（finalize 重生成轮不换 prompt）。
    # with 块包住组装与 invoke：先解析版本以决定占位符填充路径，块内禁止再 get 同名 prompt
    with use("qa", sticky_key=state.get("session_id")) as qa_prompt:
        context_parts = []
        if summary:
            context_parts.append(f"## 早期对话摘要\n{summary}")
        if recalled:
            mem_lines = [f"- {m['content'][:300]}（相关度 {m['score']:.2f}）" for m in recalled]
            context_parts.append("## 相关历史记忆（语义召回）\n" + "\n".join(mem_lines))
        if invoice is not None:
            context_parts.append(_invoice_block(state, invoice))
        if "{user_query}" in qa_prompt.text:
            # 参数化模板（v3+）：用户问题与参考条款填入 system prompt，human 消息只承载其余上下文。
            # 用 replace 而非 str.format：模板内 JSON/引用示例的花括号无需转义。
            # 条款是 human 消息的大头，预算先花在它身上（其余上下文随后还有 fit_budget 兜底）
            reference_clauses = (
                fit_budget(format_chunks_for_prompt(chunks), cfg.context.token_budget)
                or "（无）"
            )
            system_text = (
                qa_prompt.text
                .replace("{user_query}", state.get("user_question") or "（无，发票查验汇总）")
                .replace("{reference_clauses}", reference_clauses)
            )
        else:
            # 旧版非参数化 prompt（v1/v2）：问题与条款仍走 human 消息
            system_text = qa_prompt.text
            context_parts.append("## 参考条款（检索结果）\n\n" + format_chunks_for_prompt(chunks))
            if state.get("user_question"):
                context_parts.append(f"## 用户问题\n{state['user_question']}")
        # 自反馈：自评不通过时带上改进提示重生成（有界，见 critic_answer/route_after_critique）
        critic_fb = state.get("critic_feedback")
        if critic_fb:
            context_parts.append(f"## 上一版回答的不足（请针对性改进，其余内容保持）\n{critic_fb}")
        # 历史教训：过往相似问题自评发现的答案类不足，提前规避（当轮已有 critic_feedback 时跳过，避免重复）
        elif state.get("user_question"):
            try:
                lcfg = cfg.feedback.lessons
                lessons = get_lesson_store().recall(
                    state["user_question"],
                    kind="answer",
                    top_k=lcfg.recall_top_k,
                    min_score=lcfg.min_score,
                )
                if lessons:
                    lines = [f"- {l['content'][:200]}" for l in lessons]
                    context_parts.append(
                        "## 历史改进提示（过往类似问题的自评教训，请提前规避）\n" + "\n".join(lines)
                    )
            except Exception as exc:  # noqa: BLE001 — 教训是增强项，失败不阻塞回答
                logger.warning("答案教训召回失败（忽略）：%s", exc)

        # 意图区分：有发票才输出查验类章节；纯政策问答直接自然作答，不做格式化模板
        if invoice is not None:
            output_rule = (
                "请按以下结构输出 Markdown：发票概览 / 验真结论 / 重复性 / 风险提示 / "
                "政策依据（[来源N] 引用列表）。"
                '结合对话历史理解追问（如"那出租车票呢"指上一问题中的票据类型）。'
            )
        else:
            output_rule = (
                "本次未上传发票，属于纯政策问答：请直接、自然地回答用户问题本身，"
                "不要套用任何固定章节模板，禁止输出发票概览 / 验真结论 / 重复性 / 风险提示"
                "等发票查验类内容，也不要声明「未提供发票」；"
                "政策结论按 system prompt 要求标注 [来源N] 引用。"
                '结合对话历史理解追问（如"那出租车票呢"指上一问题中的票据类型）。'
            )
        prompt_text = "\n\n".join(context_parts) + "\n\n" + output_rule
        # 预算兜底：完整 prompt 超上限时从头截断
        prompt_text = fit_budget(prompt_text, cfg.context.token_budget)
        messages = [
            SystemMessage(content=system_text),
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
