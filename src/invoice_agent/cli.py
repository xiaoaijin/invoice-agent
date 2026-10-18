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

from invoice_agent.runner import NODE_LABELS, node_latency, run_stream


def _stream_graph(file_path: str | None, question: str | None, thread_id: str) -> dict:
    """流式执行图：节点进度打 stderr（stdout 保持只输出最终答案），返回聚合 state。"""

    def _progress(node: str, handler) -> None:
        latency = node_latency(handler, node)
        msg = f"[节点] {NODE_LABELS.get(node, node)} ✓"
        if latency is not None:
            msg += f" {latency:.0f}ms"
        print(msg, file=sys.stderr)

    started = time.perf_counter()
    final, handler = run_stream(file_path, question, thread_id, progress_cb=_progress)
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


# ---- prompt 版本管理 ----


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def cmd_prompts_list(args: argparse.Namespace) -> int:
    from invoice_agent.prompts.registry import get_registry

    rows = [
        [r["name"], r["active"], r["active_hash"], str(r["versions"]),
         str(r["weights"]), r["description"]]
        for r in get_registry().list_prompts()
    ]
    _print_table(["name", "active", "hash", "versions", "weights", "description"], rows)
    return 0


def cmd_prompts_show(args: argparse.Namespace) -> int:
    from invoice_agent.prompts.registry import get_registry

    meta = get_registry().get(args.name, version=args.version)
    print(f"# {meta.name} @ {meta.version}（hash {meta.hash}）\n")
    print(meta.text)
    return 0


def cmd_prompts_set_active(args: argparse.Namespace) -> int:
    from invoice_agent.prompts.registry import get_registry

    get_registry().set_active(args.name, args.version)
    meta = get_registry().get(args.name)
    print(f"已切换 {args.name} 激活版本 → {args.version}（hash {meta.hash}）")
    return 0


def cmd_prompts_usage(args: argparse.Namespace) -> int:
    from invoice_agent.config import get_app_config
    from invoice_agent.monitoring.store import MonitorStore

    store = MonitorStore(get_app_config().monitoring_db_path)
    rows = [
        [
            r["prompt_name"] or "?",
            r["prompt_version"] or "?",
            str(r["calls"]),
            f"{r['success_rate']:.2f}" if r["success_rate"] is not None else "-",
            f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] is not None else "-",
            f"¥{r['cost_yuan']:.4f}",
        ]
        for r in store.prompt_usage()
    ]
    if not rows:
        print("（暂无 prompt 调用记录）")
        return 0
    _print_table(["name", "version", "calls", "success_rate", "avg_latency", "cost"], rows)
    return 0


def cmd_prompts_report(args: argparse.Namespace) -> int:
    """按版本输出 A/B 对比报告（调用指标 + 人工标注聚合）。"""
    from invoice_agent.config import get_app_config
    from invoice_agent.monitoring.store import MonitorStore

    rows = MonitorStore(get_app_config().monitoring_db_path).version_report(args.name)
    if not rows:
        print(f"（prompt {args.name} 暂无调用记录）")
        return 0
    _print_table(
        ["version", "calls", "success_rate", "avg_latency", "cost", "annotated", "avg_rating"],
        [
            [
                r["prompt_version"] or "?",
                str(r["calls"]),
                f"{r['success_rate']:.2f}" if r["success_rate"] is not None else "-",
                f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] is not None else "-",
                f"¥{r['cost_yuan']:.4f}",
                str(r["annotated"]),
                f"{r['avg_rating']:.2f}" if r["avg_rating"] is not None else "-",
            ]
            for r in rows
        ],
    )
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


def cmd_serve(args: argparse.Namespace) -> int:
    """启动 HTTP API（uvicorn）。须单 worker 部署：checkpointer 已可跨进程（postgres 后端），
    但监控/台账/会话历史三个 SQLite 文件库与 BM25 进程内索引仍不兼容多 worker。"""
    import uvicorn

    uvicorn.run(
        "invoice_agent.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
    )
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

    p_chat = sub.add_parser("chat", help="多轮对话模式（支持追问与发票查验）")
    p_chat.add_argument(
        "--session",
        default="default",
        help="会话名（同一会话保留记忆；postgres checkpointer 下跨进程/重启也恢复）",
    )
    p_chat.set_defaults(func=cmd_chat)

    p_prompts = sub.add_parser("prompts", help="prompt 版本管理（查看/切换/用量）")
    p_prompts_sub = p_prompts.add_subparsers(dest="action", required=True)
    p_prompts_sub.add_parser("list", help="列出全部 prompt 及激活版本").set_defaults(
        func=cmd_prompts_list
    )
    p_show = p_prompts_sub.add_parser("show", help="查看 prompt 全文（默认激活版本）")
    p_show.add_argument("name")
    p_show.add_argument("--version", default=None, help="指定版本号")
    p_show.set_defaults(func=cmd_prompts_show)
    p_active = p_prompts_sub.add_parser("set-active", help="切换激活版本（回写 YAML）")
    p_active.add_argument("name")
    p_active.add_argument("version")
    p_active.set_defaults(func=cmd_prompts_set_active)
    p_prompts_sub.add_parser("usage", help="按 prompt 名+版本查看调用量/成功率/成本").set_defaults(
        func=cmd_prompts_usage
    )
    p_report = p_prompts_sub.add_parser("report", help="按版本输出 A/B 对比报告（含人工标注）")
    p_report.add_argument("name", help="prompt 名")
    p_report.set_defaults(func=cmd_prompts_report)

    p_serve = sub.add_parser("serve", help="启动 HTTP API（FastAPI/uvicorn，单 worker）")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
