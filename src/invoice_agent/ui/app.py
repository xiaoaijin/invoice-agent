"""发票助手 Agent — Streamlit 界面。

运行：uv run streamlit run src/invoice_agent/ui/app.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import date
from pathlib import Path

import streamlit as st

# streamlit run 以脚本方式执行，需手动把 src 加入 path
PACKAGE_DIR = Path(__file__).resolve().parents[1]  # src/invoice_agent
SRC_DIR = PACKAGE_DIR.parent
SAMPLES_DIR = PACKAGE_DIR / "samples"
sys.path.insert(0, str(SRC_DIR))

from invoice_agent.config import get_app_config
from invoice_agent.monitoring.setup import setup_logging
from invoice_agent.services.dedup_store import DedupStore

setup_logging()

st.set_page_config(page_title="发票助手 Agent", page_icon="🧾", layout="wide")

MAX_DISPLAY_HISTORY = 50

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

CSS = """
<style>
    /* 隐藏 Streamlit 原生 chrome */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {display: none;}
    .block-container {padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1080px;}

    /* 顶部横幅 */
    .ia-hero {
        display: flex; align-items: center; gap: 16px;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 60%, #9333ea 100%);
        border-radius: 16px; padding: 20px 28px; color: #fff; margin-bottom: 8px;
    }
    .ia-hero-icon {font-size: 40px; line-height: 1;}
    .ia-hero-title {font-size: 24px; font-weight: 700; letter-spacing: .5px;}
    .ia-hero-sub {font-size: 13px; opacity: .85; margin-top: 4px;}

    /* 空状态欢迎卡 */
    .ia-empty {text-align: center; padding: 36px 0 8px;}
    .ia-empty-emoji {font-size: 52px;}
    .ia-empty-title {font-size: 22px; font-weight: 700; margin-top: 8px; color: #1e1b4b;}
    .ia-empty-sub {font-size: 14px; color: #6b7280; margin-top: 6px;}

    /* 侧边栏 */
    section[data-testid="stSidebar"] {background: #f8f9fe; border-right: 1px solid #e9eaf6;}
    section[data-testid="stSidebar"] hr {margin: .6rem 0;}

    /* 上传区：柔和虚线框 */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px; border: 1.5px dashed #c7cbe2; background: #fafbff;
    }
    [data-testid="stFileUploaderDropzone"]:hover {border-color: #6366f1;}

    /* 按钮 */
    .stButton > button {border-radius: 10px; font-weight: 500;}
    button[kind="primary"] {box-shadow: 0 2px 10px rgba(79, 70, 229, .28);}

    /* 指标卡 */
    [data-testid="stMetric"] {
        background: #fafbff; border: 1px solid #eef0f8;
        border-radius: 12px; padding: 12px 14px;
    }

    /* 气泡与状态容器 */
    [data-testid="stChatMessage"] {padding: .5rem 0;}
    div[data-testid="stStatusWidget"] {border-radius: 12px;}
</style>
"""


def ensure_session() -> str:
    """会话 thread_id：同一浏览器会话共享（含发票查验与追问），「新对话」重置。"""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    return st.session_state.thread_id


def run_graph(
    file_path: str | None,
    question: str | None,
    thread_id: str,
    progress_cb=None,
) -> tuple[dict, object | None]:
    """流式执行图：每完成一个节点回调 progress_cb，返回聚合的最终 state 与监控 handler。"""
    from invoice_agent.graph.builder import build_graph
    from invoice_agent.monitoring.setup import build_run_config

    config, handler = build_run_config(thread_id, run_type="invoice" if file_path else "ask")
    graph = build_graph()
    final: dict = {}
    for update in graph.stream(
        {"session_id": thread_id, "file_path": file_path, "user_question": question},
        config=config,
        stream_mode="updates",
    ):
        for node, delta in update.items():
            final.update(delta or {})
            if progress_cb:
                progress_cb(node, handler)
    return final, handler


def _node_latency(handler, node: str) -> float | None:
    """从 handler 事件缓冲中取节点最近一次耗时（ms）。"""
    if handler is None:
        return None
    for e in reversed(handler.events):
        if e.get("type") == "node_end" and e.get("node") == node:
            return e.get("latency_ms")
    return None


def _progress_writer(node: str, handler) -> None:
    """run_graph 的实时进度回调：在 st.status 容器内逐行打勾。"""
    label = NODE_LABELS.get(node, node)
    latency = _node_latency(handler, node)
    line = f"✅ {label}" if latency is None else f"✅ {label}（{latency:.0f}ms）"
    st.write(line)


def render_invoice_card(invoice) -> None:
    st.markdown("##### 📋 发票概览")
    c1, c2, c3 = st.columns(3)
    c1.metric("发票类型", invoice.invoice_type.value)
    c1.metric("发票号码", invoice.invoice_number)
    c2.metric("开票日期", str(invoice.invoice_date))
    c2.metric("价税合计", f"¥{invoice.total_amount:,.2f}")
    c3.metric(
        "不含税金额", f"¥{invoice.amount_without_tax:,.2f}" if invoice.amount_without_tax else "—"
    )
    c3.metric("税额", f"¥{invoice.tax_amount:,.2f}" if invoice.tax_amount else "—")
    st.write(f"**购买方**：{invoice.buyer_name}（税号 {invoice.buyer_tax_id or '—'}）")
    st.write(f"**销售方**：{invoice.seller_name}（税号 {invoice.seller_tax_id or '—'}）")
    if invoice.items:
        st.dataframe(
            [
                {
                    "名称": i.name,
                    "规格": i.specification or "",
                    "数量": i.quantity or "",
                    "金额": i.amount or "",
                    "税率": i.tax_rate or "",
                }
                for i in invoice.items
            ],
            use_container_width=True,
            hide_index=True,
        )
    low_conf = {k: v for k, v in invoice.field_confidence.items() if v < 0.8}
    if low_conf:
        st.warning(f"以下字段置信度较低，请人工核对：{low_conf}")


def render_assistant_turn(idx: int, entry: dict) -> None:
    """渲染一条助手回复：发票查验结果卡片，或纯问答的回答。"""
    answer = entry.get("answer")

    if entry.get("kind") == "invoice":
        if entry.get("invoice") is None:
            st.error(f"❌ 发票识别失败：{answer or '未知错误'}")
            return
        render_invoice_card(entry["invoice"])

        st.markdown("##### 🔍 验真结果")
        ver = entry.get("verification")
        if ver is not None:
            if ver.status == "PASS":
                st.success(f"✔ 查验通过（{ver.provider}）— {ver.message}")
            elif ver.status == "FAIL":
                st.error(f"✘ 查验不符（{ver.provider}）— {ver.message}")
            else:
                st.warning(f"? 查验异常（{ver.provider}）— {ver.message}")
            st.caption("⚠️ 当前为 Mock 验真，结果不代表真实真伪")

        st.markdown("##### 🔁 查重结果")
        dups = entry.get("dups") or []
        if dups:
            st.warning(f"检测到 {len(dups)} 条疑似重复报销记录")
            for d in dups:
                st.markdown(
                    f"- 发票号 `{d.invoice_number}`（{d.invoice_date}，¥{d.total_amount:,.2f}）"
                    f"已于 {d.first_submitted_at:%Y-%m-%d %H:%M} 提交，状态：**{d.status}**"
                )
        else:
            st.success("台账中无重复记录")

        st.markdown("##### 🛡 风控预警")
        findings = entry.get("findings") or []
        if findings:
            for f in findings:
                icon = {"高": "🔴", "中": "🟡", "低": "🔵"}[f.level.value]
                with st.container(border=True):
                    st.markdown(f"{icon} **[{f.level.value}] {f.rule_id} {f.rule_name}**")
                    st.write(f.message)
        else:
            st.success("未触发风控规则")

    if answer:
        if entry.get("kind") != "invoice":
            st.markdown("##### 💬 回答")
        else:
            st.markdown("##### 📝 综合结论")
        st.markdown(answer)
    if entry.get("rag_chunks"):
        with st.expander("查看检索到的政策条款"):
            for c in entry["rag_chunks"]:
                article = c.article_no or ""
                st.markdown(f"**《{c.source}》{c.doc_title} {article}**（score={c.score}）")
                st.text(c.text[:300])

    if entry.get("kind") == "invoice" and entry.get("invoice") is not None:
        st.divider()
        inv = entry["invoice"]
        if entry.get("submitted"):
            st.success("✔ 已写入报销台账，重复报销检测即刻生效")
        elif st.button("✅ 确认提交报销（写入台账）", key=f"submit_{idx}", type="primary"):
            import hashlib

            sha = hashlib.sha256(Path(entry["file_path"]).read_bytes()).hexdigest()
            DedupStore(get_app_config().db_path).save_reimbursement(
                invoice_number=inv.invoice_number,
                invoice_code=inv.invoice_code,
                seller_tax_id=inv.seller_tax_id,
                buyer_tax_id=inv.buyer_tax_id,
                total_amount=inv.total_amount,
                invoice_date=date.fromisoformat(str(inv.invoice_date)),
                file_sha256=sha,
            )
            entry["submitted"] = True
            st.rerun()


def render_composer() -> None:
    """统一输入区：上传发票（可选）+ 文字（可选），支持三种组合。"""
    # 发送后清空输入框：须在下一次 text_area 实例化前消费
    if st.session_state.pop("_clear_composer", False):
        st.session_state.pop("composer_text", None)
    # 示例问题按钮排队填充输入框
    if prompt := st.session_state.pop("queued_prompt", None):
        st.session_state["composer_text"] = prompt

    pending = st.session_state.get("pending_file")
    uploaded = st.file_uploader(
        "上传发票（可选，图片或 PDF）", type=["png", "jpg", "jpeg", "pdf"], key="_uploader"
    )
    if pending and uploaded is None:
        c1, c2 = st.columns([5, 1])
        c1.info(f"📎 将使用示例发票：**{pending['name']}**")
        if c2.button("移除", key="remove_pending"):
            st.session_state.pop("pending_file", None)
            st.rerun()

    st.text_area(
        "问题（可选）",
        key="composer_text",
        height=80,
        placeholder="可只上传发票、只提问，或两者同时。例如：这张发票能报销吗？餐饮部分能抵扣吗？",
    )

    has_file = pending is not None or uploaded is not None
    text = (st.session_state.get("composer_text") or "").strip()
    if has_file and text:
        mode = "🔎 识票查验 + 💬 政策解答"
    elif has_file:
        mode = "🔎 识票查验（识票 · 验真 · 查重 · 风控）"
    elif text:
        mode = "💬 政策问答"
    else:
        mode = "请上传发票或输入问题"

    c_btn, c_mode = st.columns([1, 3])
    send = c_btn.button("🚀 发送", type="primary", use_container_width=True,
                        disabled=not (has_file or text))
    c_mode.caption(f"将执行：{mode}")
    st.session_state["_send"] = send


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="ia-empty">
            <div class="ia-empty-emoji">👋</div>
            <div class="ia-empty-title">你好，我是发票助手</div>
            <div class="ia-empty-sub">
                上传发票可自动识票、验真、查重、风控；直接提问可进行政策问答；也可以同时进行。
                发送后可在对话中继续追问。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    examples = [
        "电子发票丢失了还能报销吗？",
        "餐饮发票的进项税可以抵扣吗？",
        "增值税专用发票和普通发票报销有什么区别？",
    ]
    cols = st.columns(3)
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["queued_prompt"] = ex
            st.rerun()


def render_conversation_tab(cfg) -> None:
    display: list[dict] = st.session_state.setdefault("display", [])

    render_composer()

    # 发送处理
    if st.session_state.pop("_send", False):
        pending = st.session_state.get("pending_file")
        payload = None
        if pending:
            payload = (pending["bytes"], pending["name"])
        elif (uploaded := st.session_state.get("_uploader")) is not None:
            payload = (uploaded.getvalue(), uploaded.name)

        text = (st.session_state.get("composer_text") or "").strip()
        if payload is None and not text:
            st.warning("请上传发票或输入问题")
        else:
            fpath = None
            fname = None
            if payload:
                data, fname = payload
                updir = Path(cfg.db_path).parent / "uploads"
                updir.mkdir(parents=True, exist_ok=True)
                fpath = updir / f"{uuid.uuid4().hex[:8]}_{fname}"
                fpath.write_bytes(data)

            display.append({"role": "user", "file_name": fname, "text": text or None})
            thread_id = st.session_state.thread_id
            with (
                st.chat_message("assistant", avatar="🧾"),
                st.status("处理中…", expanded=True) as status,
            ):
                state, handler = run_graph(
                    str(fpath) if fpath else None,
                    text or None,
                    thread_id,
                    progress_cb=_progress_writer,
                )
                done = sum(1 for e in handler.events if e["type"] == "node_end") if handler else 0
                status.update(label=f"完成（{done} 个节点）", state="complete", expanded=False)

            entry: dict = {
                "role": "assistant",
                "answer": state.get("answer"),
                "rag_chunks": state.get("rag_chunks") or [],
            }
            if fpath:
                entry.update(
                    {
                        "kind": "invoice",
                        "invoice": state.get("invoice"),
                        "verification": state.get("verification"),
                        "dups": state.get("duplicate_hits") or [],
                        "findings": state.get("risk_findings") or [],
                        "file_path": str(fpath),
                        "submitted": False,
                    }
                )
            else:
                entry["kind"] = "qa"
            display.append(entry)

            st.session_state["_clear_composer"] = True
            st.session_state.pop("pending_file", None)
            st.rerun()

    # 时间线
    if not display:
        render_empty_state()
        return
    for i, e in enumerate(display[-MAX_DISPLAY_HISTORY:]):
        if e["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💼"):
                if e.get("file_name"):
                    st.markdown(f"📎 上传了发票 `{e['file_name']}`")
                if e.get("text"):
                    st.markdown(e["text"])
        else:
            with st.chat_message("assistant", avatar="🧾"):
                render_assistant_turn(i, e)


def render_monitor_tab() -> None:
    """监控面板：近况指标、历史 run 明细、告警高亮（只读自 MonitorStore）。"""
    from invoice_agent.monitoring.store import MonitorStore

    cfg = get_app_config()
    if not cfg.monitoring.enabled:
        st.info("监控未启用（config.yaml: monitoring.enabled=false）")
        return
    store = MonitorStore(cfg.monitoring_db_path)
    window = cfg.monitoring.alert.success_rate_window
    stats = store.stats(window)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"近 {window} 次 run", stats["run_count"])
    c2.metric(
        "成功率",
        f"{stats['success_rate']:.0%}" if stats["success_rate"] is not None else "—",
        delta="低于阈值"
        if stats["success_rate"] is not None
        and stats["success_rate"] < cfg.monitoring.alert.min_success_rate
        else None,
    )
    c3.metric(
        "平均耗时", f"{stats['avg_latency_ms'] / 1000:.1f}s" if stats["avg_latency_ms"] else "—"
    )
    c4.metric("Token 用量", f"{stats['prompt_tokens'] + stats['completion_tokens']:,}")
    c5.metric("累计成本", f"¥{stats['cost_yuan']:.4f}")

    st.subheader("🔔 告警")
    alerts = store.recent_alerts(30)
    if not alerts:
        st.success("无告警")
    for a in alerts:
        detail = f"{a['created_at'][:19]} ｜ {a['detail']}"
        if a["severity"] == "ERROR":
            st.error(f"🔴 [{a['alert_type']}] {detail}", icon="🚨")
        else:
            st.warning(f"🟡 [{a['alert_type']}] {detail}")

    st.subheader("🕘 最近运行")
    runs = store.recent_runs(20)
    if not runs:
        st.caption("暂无运行记录")
        return
    st.dataframe(
        [
            {
                "时间": r["started_at"][:19] if r["started_at"] else "",
                "类型": r["run_type"] or "",
                "状态": r["status"] or "",
                "耗时(s)": round((r["total_latency_ms"] or 0) / 1000, 1),
                "重试": r["retry_count"] or 0,
                "节点数": r["node_count"] or 0,
                "tokens": (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0),
                "成本(元)": round(r["cost_yuan"] or 0, 4),
            }
            for r in runs
        ],
        use_container_width=True,
        hide_index=True,
    )
    options = [r["run_id"] for r in runs]
    picked = st.selectbox("查看节点时间线", options, format_func=lambda rid: f"run {rid[:8]}…")
    detail = store.run_detail(picked)
    if detail["spans"]:
        st.dataframe(
            [
                {
                    "节点": NODE_LABELS.get(s["node"], s["node"]),
                    "顺序": s["seq"],
                    "耗时(ms)": round(s["latency_ms"] or 0),
                    "状态": s["status"],
                    "重试次数": s["retry_of"],
                    "错误": s["error"] or "",
                }
                for s in detail["spans"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_prompts_tab() -> None:
    """prompt 版本面板：版本清单、全文查看、A/B 调用分布（只读，切换走 CLI/文件）。"""
    from invoice_agent.prompts.registry import get_registry

    registry = get_registry()
    prompts = registry.list_prompts()
    if not prompts:
        st.caption("prompts 目录为空")
        return

    st.subheader("📚 版本清单")
    st.dataframe(
        [
            {
                "名称": p["name"],
                "说明": p["description"],
                "激活版本": p["active"],
                "文本hash": p["active_hash"],
                "版本数": p["versions"],
                "A/B权重": ", ".join(f"{v}={w}" for v, w in p["weights"].items()),
            }
            for p in prompts
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("版本切换走 CLI（`prompts set-active`）或直接编辑 YAML，此处只读。")

    st.subheader("📄 全文查看")
    name = st.selectbox("选择 prompt", [p["name"] for p in prompts])
    versions = registry.versions(name)
    labels = {
        v.version: f"{v.version}（权重 {max(1, v.weight)}）{('：' + v.description) if v.description else ''}"
        for v in versions
    }
    version = st.radio("版本", [v.version for v in versions], format_func=lambda v: labels[v])
    meta = registry.get(name, version=version)
    st.code(meta.text, language="text")
    st.caption(f"hash：{meta.hash}")

    st.subheader("⚖️ A/B 调用分布")
    try:
        from invoice_agent.monitoring.store import MonitorStore

        store = MonitorStore(get_app_config().monitoring_db_path)
        usage = store.prompt_usage()
    except Exception as exc:  # noqa: BLE001 — 监控降级时提示即可
        st.caption(f"监控不可用（{exc}）")
        return
    rows = [u for u in usage if u["prompt_name"] == name]
    if not rows:
        st.caption("该 prompt 暂无调用记录")
        return
    st.dataframe(
        [
            {
                "版本": u["prompt_version"] or "?",
                "调用量": u["calls"],
                "成功率": f"{u['success_rate']:.0%}" if u["success_rate"] is not None else "—",
                "平均耗时(ms)": round(u["avg_latency_ms"] or 0),
                "成本(元)": round(u["cost_yuan"] or 0, 4),
            }
            for u in rows
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧾 发票助手")
        st.caption("识票 · 验真 · 查重 · 风控 · 政策问答")
        if st.button("🆕 新对话", use_container_width=True, type="primary"):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.display = []
            st.rerun()

        st.divider()
        st.caption("📚 示例发票（点击后直接发送即可查验）")
        for p in sorted(SAMPLES_DIR.glob("*.png"))[:8]:
            if st.button(p.name, key=f"sample_{p.name}", use_container_width=True):
                st.session_state["pending_file"] = {"name": p.name, "bytes": p.read_bytes()}
                st.rerun()

        st.divider()
        st.caption("⚠️ 验真当前为 Mock，未接真实查验平台")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ia-hero">
            <div class="ia-hero-icon">🧾</div>
            <div>
                <div class="ia-hero-title">发票助手</div>
                <div class="ia-hero-sub">识票 · 验真 · 查重 · 风控 · 政策问答 — 一次输入，全部搞定</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ensure_session()
    cfg = get_app_config()
    render_sidebar()

    tab_chat, tab_monitor, tab_prompts = st.tabs(["💬 助手对话", "📈 监控面板", "📜 Prompt 版本"])
    with tab_chat:
        render_conversation_tab(cfg)
    with tab_monitor:
        render_monitor_tab()
    with tab_prompts:
        render_prompts_tab()


main()
