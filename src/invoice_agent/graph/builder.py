"""LangGraph 状态图组装。

流程：
START → intake → route_entry ─┬─ 有文件 → recognize → check_extract ─┬─ 通过 → verify → dedup → risk ─┐
                              │                                      ├─ 失败重试 → recognize           │
                              │                                      └─ 失败降级 ──────────────────────┤
                              └─ 仅提问 → retrieve_docs ──────────────────────────────────────────────┤
                                       ↓                                                              │
                                    retrieve_docs ←────────────────────────────────────────────────────┘
                                       ↓
                                    finalize → END
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from invoice_agent.graph.nodes import answer, dedup, intake, recognize, risk, route, verify
from invoice_agent.graph.state import AgentState


@lru_cache
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake.intake)
    g.add_node("recognize", recognize.recognize)
    g.add_node("verify", verify.verify)
    g.add_node("dedup", dedup.dedup)
    g.add_node("risk", risk.risk)
    g.add_node("retrieve_docs", answer.retrieve_docs)
    g.add_node("finalize", answer.finalize)

    g.add_edge(START, "intake")
    g.add_conditional_edges(
        "intake",
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
    g.add_edge("risk", "retrieve_docs")
    g.add_edge("retrieve_docs", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())
