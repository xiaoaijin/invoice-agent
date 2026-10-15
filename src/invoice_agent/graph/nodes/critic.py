"""critic 节点：自反馈优化。

- risk_review：风控结论 LLM 复核（单次修订，不回环——规则引擎重跑无信息增益）
- critic_answer：回答质量自评，不通过时带反馈回 finalize 重生成（有界）；
  检索不足类问题只落教训库供后续召回，不再回环重检（query 改写已前置到首检、由意图驱动）
- 自评发现的问题同时落长期教训库（lesson_store），供后续运行召回优化
"""

from __future__ import annotations

import logging

from invoice_agent.config import get_app_config
from invoice_agent.feedback.critics import (
    critique_answer,
    default_llm_fn,
    review_risk,
)
from invoice_agent.graph.state import AgentState

logger = logging.getLogger(__name__)


def risk_review(state: AgentState) -> dict:
    """风控复核：误报移除 / 漏报补充（rule_id=LLM-REVIEW）。LLM 失败原样透传。"""
    cfg = get_app_config()
    if not cfg.feedback.risk_review.enabled:
        return {}
    invoice = state.get("invoice")
    if invoice is None:  # 纯 RAG / 降级路径无票面数据，无复核对象
        return {}
    findings = state.get("risk_findings") or []
    try:
        review = review_risk(findings, invoice, default_llm_fn())
    except Exception as exc:  # noqa: BLE001 — 复核是增强项，失败不阻塞
        logger.warning("风控复核异常（原样透传）：%s", exc)
        return {}
    if review.removed or review.added:
        logger.info(
            "风控复核完成：移除误报 %s，补充 %d 条 LLM-REVIEW 发现",
            review.removed,
            len(review.added),
        )
    return {"risk_findings": review.adjusted, "risk_review": review.model_dump()}


def critic_answer(state: AgentState) -> dict:
    """回答自评：不通过 → finalize 带反馈重生成（有界）。"""
    cfg = get_app_config()
    fcfg = cfg.feedback.answer
    if not fcfg.enabled:
        return {"critique": {"passed": True}, "critic_feedback": None}

    answer = state.get("answer")
    if not answer:
        return {"critique": {"passed": True}, "critic_feedback": None}
    # 降级人工复核路径 / 兜底提示不评审（无 LLM 汇总产物）
    if state.get("invoice") is None and state.get("file_path"):
        return {"critique": {"passed": True, "skipped": "fallback"}, "critic_feedback": None}
    if not state.get("file_path") and not state.get("user_question"):
        return {"critique": {"passed": True, "skipped": "noop"}, "critic_feedback": None}

    result = critique_answer(answer, state, default_llm_fn(), fcfg.pass_score)
    retry = state.get("answer_retry", 0)
    if result.passed or retry >= fcfg.max_retries:
        _record_lessons(state, result)
        dump = result.model_dump()
        if not result.passed:
            # 重试耗尽放行：标记 passed 防 route_after_critique 回环
            dump["passed"] = True
            dump["exhausted"] = True
            # 防下一轮对话误读旧值
            dump["retrieval_issues"] = []
        # 通过或耗尽：显式清空 feedback，防 checkpointer 旧值串扰下一轮对话
        return {"critique": dump, "critic_feedback": None}
    _record_lessons(state, result)
    feedback = "；".join(result.issues) or f"综合评分 {result.score:.2f} 低于阈值 {fcfg.pass_score}"
    logger.info("回答自评未通过（score=%.2f），触发重生成：%s", result.score, feedback)
    return {
        "critique": result.model_dump(),
        "answer_retry": retry + 1,
        "critic_feedback": feedback,
    }


def _record_lessons(state: AgentState, result) -> None:
    """自评发现的问题落长期教训（lesson_store），PG 不可用/开关关闭时静默跳过。"""
    if not (result.retrieval_issues or result.issues):
        return
    try:
        from invoice_agent.services.lesson_store import get_lesson_store

        store = get_lesson_store()
        question = state.get("user_question") or (
            state["invoice"].summary_text() if state.get("invoice") is not None else ""
        )
        for issue in result.retrieval_issues:
            store.record("retrieval", question, issue)
        for issue in result.issues:
            store.record("answer", question, issue)
    except Exception as exc:  # noqa: BLE001 — 教训是增强项，失败不阻塞
        logger.warning("教训写入失败（忽略）：%s", exc)
