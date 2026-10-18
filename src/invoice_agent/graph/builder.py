"""LangGraph 状态图组装。

流程：
START → intake → intent → route_entry ─┬─ 有文件 → recognize ⇄ check_extract（失败重试/降级）┐
                                       │                                                  ├→ verify → dedup → risk → risk_review ┐
                                       └─ 仅提问（闲聊短路 → finalize）→ retrieve_docs ────┘                                       │
                                                ↓                                                                                 │
                                             retrieve_docs ←──────────────────────────────────────────────────────────────────────┘
                                                ↓                              ↑
                                             finalize ⇄ critic_answer ─┬─ 通过 → END
                                                ↑                      └─ 其余不通过（≤max_retries）→ finalize 重生成
                                                └────────────────────────────┘
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from invoice_agent.graph.checkpointer import get_checkpointer
from invoice_agent.graph.nodes import (
    answer,
    critic,
    dedup,
    intake,
    intent,
    recognize,
    risk,
    route,
    verify,
)
from invoice_agent.graph.state import AgentState


@lru_cache
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake.intake)
    g.add_node("intent", intent.classify_intent)
    g.add_node("recognize", recognize.recognize)
    g.add_node("verify", verify.verify)
    g.add_node("dedup", dedup.dedup)
    g.add_node("risk", risk.risk)
    g.add_node("risk_review", critic.risk_review)
    g.add_node("retrieve_docs", answer.retrieve_docs)
    g.add_node("finalize", answer.finalize)
    g.add_node("critic_answer", critic.critic_answer)

    g.add_edge(START, "intake")
    g.add_edge("intake", "intent")
    # 意图识别后路由：有文件走识票；仅提问走检索（闲聊短路直接 finalize）
    g.add_conditional_edges(
        "intent",
        route.route_entry,
        {"recognize": "recognize", "retrieve_docs": "retrieve_docs", "finalize": "finalize"},
    )
    # 提取校验：通过→verify；失败重试→recognize（携带错误提示）；重试耗尽→finalize 降级
    g.add_conditional_edges(
        "recognize",
        route.route_after_extract,
        {"verify": "verify", "recognize": "recognize", "finalize": "finalize"},
    )
    g.add_edge("verify", "dedup")
    g.add_edge("dedup", "risk")
    g.add_edge("risk", "risk_review")
    g.add_edge("risk_review", "retrieve_docs")
    g.add_edge("retrieve_docs", "finalize")
    # 回答自评：不通过 → finalize 带反馈重生成（有界）；通过/耗尽 → END
    # （检索不足类问题只落教训库，query 改写已前置到首检、由意图驱动，不再回环重检）
    g.add_conditional_edges(
        "critic_answer",
        route.route_after_critique,
        {"finalize": "finalize", "END": END},
    )
    g.add_edge("finalize", "critic_answer")

    return g.compile(checkpointer=get_checkpointer())
