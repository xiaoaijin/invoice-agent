"""图执行公共入口：CLI 与 UI（以及后续 API 层）共用的流式执行逻辑。

此前 cli.py 的 _stream_graph 与 ui/app.py 的 run_graph 是同一逻辑的两份实现
（updates 流驱动节点进度、messages 流捕获 finalize 逐 token 输出），此处收拢为
生成器核心 iter_run + 回调式薄封装 run_stream。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

NODE_LABELS = {
    "intake": "接收文件",
    "recognize": "识票提取",
    "verify": "验真",
    "dedup": "去重检查",
    "risk": "风控规则",
    "risk_review": "风控复核",
    "retrieve_docs": "政策检索",
    "finalize": "汇总回答",
    "critic_answer": "回答自评",
}


def node_latency(handler: Any, node: str) -> float | None:
    """从 handler 事件缓冲中取节点最近一次耗时（ms）。

    并发节点下回调来自 LangGraph 内部线程池，先快照再遍历避免与写入交错。
    """
    if handler is None:
        return None
    for e in reversed(list(handler.events)):
        if e.get("type") == "node_end" and e.get("node") == node:
            return e.get("latency_ms")
    return None


def iter_run(
    file_path: str | None,
    question: str | None,
    thread_id: str,
    *,
    stream_tokens: bool = True,
) -> Iterator[tuple[str, Any, Any]]:
    """流式执行图的生成器核心。

    yield 三类事件，第三个元素始终是 handler（监控关闭时为 None）：
      ("node",  node_name, handler)   节点完成（updates 流）
      ("token", piece, handler)       finalize 的逐 token 输出（messages 流）
      ("done",  final_state, handler) 整图执行完毕
    stream_tokens=False 时跳过 messages 流（CLI 调试场景不需要 token 事件）。
    """
    from invoice_agent.graph.builder import build_graph
    from invoice_agent.monitoring.setup import build_run_config

    run_type = "invoice" if file_path else "ask"
    config, handler = build_run_config(thread_id, run_type=run_type)
    graph = build_graph()
    modes = ["updates", "messages"] if stream_tokens else ["updates"]
    final: dict = {}
    for mode, payload in graph.stream(
        {"session_id": thread_id, "file_path": file_path, "user_question": question},
        config=config,
        stream_mode=modes,
    ):
        if mode == "updates":
            for node, delta in payload.items():
                final.update(delta or {})
                yield "node", node, handler
        elif mode == "messages":
            chunk, meta = payload
            if meta.get("langgraph_node") == "finalize" and chunk.content:
                yield "token", chunk.content, handler
    yield "done", final, handler


def run_stream(
    file_path: str | None,
    question: str | None,
    thread_id: str,
    *,
    progress_cb: Callable[[str, Any], None] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> tuple[dict, Any]:
    """回调式封装：节点完成回调 progress_cb(node, handler)，token 回调 on_token(piece)。

    返回 (最终聚合 state, handler)。
    """
    final: dict = {}
    handler = None
    for event, payload, h in iter_run(file_path, question, thread_id):
        handler = h
        if event == "node":
            if progress_cb:
                progress_cb(payload, handler)
        elif event == "token":
            if on_token:
                on_token(payload)
        elif event == "done":
            final = payload
    return final, handler
