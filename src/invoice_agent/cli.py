"""调试/演示 CLI。

用法：
  uv run python -m invoice_agent.cli recognize samples/01_normal_special.png   # 只识票
  uv run python -m invoice_agent.cli run samples/01_normal_special.png --question "这张发票能报销吗"
  uv run python -m invoice_agent.cli ask "餐饮发票能抵扣进项税吗"               # 纯政策问答
"""

from __future__ import annotations

import argparse
import json
import sys
import time

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


def _stream_graph(file_path: str | None, question: str | None, thread_id: str) -> dict:
    """流式执行图：节点进度打 stderr（stdout 保持只输出最终答案），返回聚合 state。"""
    from invoice_agent.graph.builder import build_graph
    from invoice_agent.monitoring.setup import build_run_config

    run_type = "invoice" if file_path else "ask"
    config, handler = build_run_config(thread_id, run_type=run_type)
    graph = build_graph()
    final: dict = {}
    started = time.perf_counter()
    for update in graph.stream(
        {"session_id": thread_id, "file_path": file_path, "user_question": question},
        config=config,
        stream_mode="updates",
    ):
        for node, delta in update.items():
            final.update(delta or {})
            latency = None
            if handler:
                for e in reversed(handler.events):
                    if e.get("type") == "node_end" and e.get("node") == node:
                        latency = e.get("latency_ms")
                        break
            msg = f"[节点] {NODE_LABELS.get(node, node)} ✓"
            if latency is not None:
                msg += f" {latency:.0f}ms"
            print(msg, file=sys.stderr)
    elapsed = time.perf_counter() - started
    info = f"[完成] 总耗时 {elapsed:.1f}s"
    if handler:
        prompt = sum(e.get("prompt_tokens") or 0 for e in handler.events if e["type"] == "llm_end")
        completion = sum(
            e.get("completion_tokens") or 0 for e in handler.events if e["type"] == "llm_end"
        )
        cost = sum(e.get("cost_yuan") or 0 for e in handler.events if e["type"] == "llm_end")
        info += f" ｜ tokens {prompt}+{completion} ｜ 成本 ¥{cost:.4f}"
    print(info, file=sys.stderr)
    return final


def cmd_recognize(args: argparse.Namespace) -> int:
    state = _stream_graph(args.file, None, "cli")
    invoice = state.get("invoice")
    if invoice is None:
        print(f"识别失败：{state.get('extract_error')}", file=sys.stderr)
        return 1
    print(json.dumps(invoice.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    state = _stream_graph(args.file, args.question, "cli")
    print(state.get("answer") or "(无输出)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    state = _stream_graph(None, args.question, "cli")
    print(state.get("answer") or "(无输出)")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """多轮对话 REPL：/invoice <文件> 查验发票，/new 重置会话，/quit 退出。"""
    thread_id = f"cli-chat-{args.session}"
    print(
        f"多轮对话模式（会话 {args.session}）。/invoice <文件> 查验发票；/new 新会话；/quit 退出。"
    )
    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question == "/quit":
            return 0
        if question == "/new":
            thread_id = f"cli-chat-{args.session}-{int(time.time())}"
            print(f"（已切换新会话：{thread_id}）")
            continue
        file_path = None
        if question.startswith("/invoice "):
            file_path, question = (
                question[len("/invoice ") :].strip(),
                "这张发票能否报销？有什么风险？",
            )
        state = _stream_graph(file_path, question, thread_id)
        print(f"\n助手> {state.get('answer') or '(无输出)'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="invoice-agent", description="发票助手 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("recognize", help="只识别发票并输出 JSON")
    p_rec.add_argument("file", help="发票图片/PDF 路径")
    p_rec.set_defaults(func=cmd_recognize)

    p_run = sub.add_parser("run", help="完整链路：识票→验真→风控→RAG→汇总回答")
    p_run.add_argument("file", help="发票图片/PDF 路径")
    p_run.add_argument("--question", default="这张发票能否报销？有什么风险？")
    p_run.set_defaults(func=cmd_run)

    p_ask = sub.add_parser("ask", help="纯政策问答（RAG）")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="多轮对话模式（支持追问与发票查验）")
    p_chat.add_argument("--session", default="default", help="会话名（同一会话保留记忆）")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
