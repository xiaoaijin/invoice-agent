"""checkpointer PostgresSaver 集成测试：需本地 PG（uv run --no-sync python scripts/local_pg.py）。

不触 LLM / 业务节点：用最小 TypedDict 图验证跨图实例（模拟跨进程）状态恢复。
"""

import pytest

pytestmark = pytest.mark.integration


def _tiny_graph():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        turns: int

    def bump(state: S) -> dict:
        return {"turns": state.get("turns", 0) + 1}

    g = StateGraph(S)
    g.add_node("bump", bump)
    g.add_edge(START, "bump")
    g.add_edge("bump", END)
    return g


class TestPostgresCheckpointer:
    def test_persist_across_instances(self):
        """两个图实例（模拟两个进程）共享 PG checkpointer，同 thread 状态累积。"""
        from invoice_agent.config import get_settings
        from invoice_agent.graph.checkpointer import build_postgres_checkpointer

        saver = build_postgres_checkpointer(get_settings().pg_dsn)
        cfg = {"configurable": {"thread_id": "it-checkpointer-test"}}
        _tiny_graph().compile(checkpointer=saver).invoke({}, cfg)
        state = _tiny_graph().compile(checkpointer=saver).invoke({}, cfg)
        assert state["turns"] == 2  # 第二个"进程"从 PG 恢复了第一轮状态
        saver.delete_thread("it-checkpointer-test")  # 清理，防跨次运行串扰
