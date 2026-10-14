"""上下文管理：单次 LLM 交互 token 预算与历史压缩。

策略（config.yaml [context]）：
- token_budget：单次交互 token 上限（字符估算，中文≈1 token/字、其他≈4 字符/token）
- compress_threshold：历史 token 达预算 60% 时触发压缩
- keep_recent_rounds：压缩时原样保留最近 N 轮（1 轮 = user+assistant 两条），
  更早的内容由 LLM 做一次滚动摘要；工具调用只保留工具名和调用结果，不进摘要正文。
"""

from __future__ import annotations

import logging

from invoice_agent.config import ContextConfig

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """你是对话摘要器。请把提供的历史对话压缩为一段简洁摘要，供后续对话作为背景知识。

要求：
1. 保留与发票报销相关的关键事实：已查验的发票号码/结论、已做出的判断、用户关注的问题。
2. 工具调用（[工具 xxx] 开头的条目）只保留工具名和结果，不展开过程。
3. 用简体中文，不超过 300 字，直接输出摘要正文。"""


def estimate_tokens(text: str) -> int:
    """确定性 token 估算：CJK 字符 ≈1 token/字，其他 ≈4 字符/token。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return cjk + other // 4 + 1


def history_tokens(history: list[dict]) -> int:
    return sum(estimate_tokens(str(e.get("content", ""))) for e in history)


def _render_entry(entry: dict) -> str:
    role = entry.get("role", "?")
    if role == "tool":
        return f"[工具 {entry.get('tool', '?')}] {entry.get('content', '')}"
    return f"{role}: {entry.get('content', '')}"


def _split_keep_window(history: list[dict], keep_rounds: int) -> tuple[list[dict], list[dict]]:
    """从尾部按「轮」（user+assistant 成对）切分，返回 (更早部分, 保留窗口)。"""
    kept_entries = keep_rounds * 2
    if len(history) <= kept_entries:
        return [], list(history)
    split_at = len(history) - kept_entries
    return history[:split_at], history[split_at:]


def compress_history(
    state: dict, summarize_fn, config: ContextConfig
) -> dict:
    """历史 token 超过预算 60% 时压缩：旧内容摘要、最近 N 轮原样保留。

    summarize_fn(prev_summary: str, old_entries: list[dict]) -> str
    返回 state 增量 {"chat_history", "history_summary"}；无需压缩返回 {}。
    """
    history = state.get("chat_history") or []
    threshold = config.token_budget * config.compress_threshold
    if history_tokens(history) <= threshold:
        return {}

    old, recent = _split_keep_window(history, config.keep_recent_rounds)
    if not old:
        return {}

    prev_summary = state.get("history_summary") or ""
    summary = summarize_fn(prev_summary, old)
    logger.info(
        "上下文压缩触发：历史 %d 条（~%d tokens）→ 摘要 %d 字 + 保留 %d 条",
        len(history), history_tokens(history), len(summary), len(recent),
    )
    return {"chat_history": recent, "history_summary": summary}


def build_summary_prompt(prev_summary: str, old_entries: list[dict]) -> list:
    """构造摘要 LLM 调用的消息列表。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    parts = []
    if prev_summary:
        parts.append(f"【此前摘要】\n{prev_summary}\n")
    parts.append("【待压缩历史对话】\n" + "\n".join(_render_entry(e) for e in old_entries))
    return [SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content="\n".join(parts))]


def default_summarizer():
    """生产用摘要函数（调 qwen-max）。"""
    from invoice_agent.services.llm import get_chat_model

    def summarize(prev_summary: str, old_entries: list[dict]) -> str:
        try:
            return get_chat_model().invoke(build_summary_prompt(prev_summary, old_entries)).content
        except Exception as exc:  # noqa: BLE001 — 摘要失败时降级为截断拼接
            logger.warning("历史摘要生成失败，降级为首尾拼接：%s", exc)
            text = "\n".join(_render_entry(e) for e in old_entries[-6:])
            return f"（自动摘要不可用，以下是较早对话的截断）\n{text[:600]}"

    return summarize


def fit_budget(text: str, budget: int) -> str:
    """兜底截断：组装出的完整 prompt 超预算时从头截断（保留后半段）。"""
    if estimate_tokens(text) <= budget:
        return text
    prefix = "（前文已因超长截断）\n"
    allowed = budget - estimate_tokens(prefix) - 1
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[-mid:]) <= allowed:
            lo = mid
        else:
            hi = mid - 1
    return prefix + text[-lo:]
