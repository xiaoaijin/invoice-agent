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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
