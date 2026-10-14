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


def cmd_recognize(args: argparse.Namespace) -> int:
    from invoice_agent.graph.builder import build_graph

    graph = build_graph()
    state = graph.invoke(
        {
            "session_id": "cli",
            "file_path": args.file,
            "user_question": None,
        },
        config={"configurable": {"thread_id": "cli"}},
    )
    invoice = state.get("invoice")
    if invoice is None:
        print(f"识别失败：{state.get('extract_error')}", file=sys.stderr)
        return 1
    print(json.dumps(invoice.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from invoice_agent.graph.builder import build_graph

    graph = build_graph()
    state = graph.invoke(
        {
            "session_id": "cli",
            "file_path": args.file,
            "user_question": args.question,
        },
        config={"configurable": {"thread_id": "cli"}},
    )
    print(state.get("answer") or "(无输出)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from invoice_agent.graph.builder import build_graph

    graph = build_graph()
    state = graph.invoke(
        {
            "session_id": "cli",
            "file_path": None,
            "user_question": args.question,
        },
        config={"configurable": {"thread_id": "cli"}},
    )
    print(state.get("answer") or "(无输出)")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """多轮对话 REPL：/invoice <文件> 查验发票，/new 重置会话，/quit 退出。"""
    from invoice_agent.graph.builder import build_graph

    graph = build_graph()
    thread_id = f"cli-chat-{args.session}"
    print(f"多轮对话模式（会话 {args.session}）。/invoice <文件> 查验发票；/new 新会话；/quit 退出。")
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
            thread_id = f"cli-chat-{args.session}-{int(__import__('time').time())}"
            print(f"（已切换新会话：{thread_id}）")
            continue
        file_path = None
        if question.startswith("/invoice "):
            file_path, question = question[len("/invoice "):].strip(), "这张发票能否报销？有什么风险？"
        state = graph.invoke(
            {"session_id": thread_id, "file_path": file_path, "user_question": question},
            config={"configurable": {"thread_id": thread_id}},
        )
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
