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
from invoice_agent.runner import NODE_LABELS, node_latency, run_stream
from invoice_agent.services.conversation_store import ConversationStore
from invoice_agent.services.dedup_store import DedupStore
from invoice_agent.ui.reimburse import render_reimburse_tab

setup_logging()

st.set_page_config(page_title="发票助手 Agent", page_icon="🧾", layout="wide")

MAX_DISPLAY_HISTORY = 50

CSS = """
<style>
    /* 隐藏 Streamlit 原生 chrome（顶栏仅做透明，不能 display:none——
       侧边栏收起后的展开箭头挂在顶栏里，藏掉顶栏就再打不开了） */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent; height: 2.2rem; pointer-events: none;
    }
    /* 展开箭头要可点击：单独恢复事件 + 白底圆角便于看清 */
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {
        pointer-events: auto; background: #fff; border: 1px solid #e5e7eb;
        border-radius: 8px; box-shadow: 0 1px 4px rgba(17, 24, 39, .08);
    }
    .block-container {
        padding-top: 1.4rem; padding-bottom: 0; max-width: 1200px;
        min-height: calc(100vh - 50px); display: flex; flex-direction: column;
    }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stTabs"]) {flex: 1;}

    /* 消息时间线：助手无头像全宽，用户右侧灰色气泡（千问式） */
    [data-testid="stChatMessage"] {padding: .4rem 0;}
    /* 头像容器为消息框首个子 div（无 testid），整体隐藏 */
    [data-testid="stChatMessage"] > div:not([data-testid="stChatMessageContent"]):first-child {
        display: none;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        margin-left: 0; flex: 1 1 auto; width: 100%;
    }
    .ia-user-row {display: flex; justify-content: flex-end; padding: .35rem 0;}
    .ia-user-bubble {
        background: #f3f4f6; border-radius: 18px; padding: 10px 16px;
        max-width: 76%; color: #1f2937; font-size: 15px; line-height: 1.6;
        word-break: break-word; text-align: left;
    }
    .ia-user-bubble code {background: #e8eaef; border-radius: 6px; padding: 1px 6px;}

    /* 空状态欢迎卡 */
    .ia-empty {text-align: center; padding: 48px 0 8px;}
    .ia-empty-emoji {font-size: 52px;}
    .ia-empty-title {font-size: 22px; font-weight: 700; margin-top: 8px; color: #1e1b4b;}
    .ia-empty-sub {font-size: 14px; color: #6b7280; margin-top: 6px;}

    /* 侧边栏（千问式：浅色底、无框会话列表） */
    section[data-testid="stSidebar"] {background: #fafafa; border-right: 1px solid #ececf1;}
    section[data-testid="stSidebar"] hr {margin: .6rem 0;}
    section[data-testid="stSidebar"] .stButton > button {
        border: none; box-shadow: none; background: transparent; color: #374151;
        font-weight: 400; padding: 7px 12px; border-radius: 10px;
        justify-content: flex-start; text-align: left;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {background: #f0f1f5;}
    section[data-testid="stSidebar"] .stButton > button > p {
        width: 100%; text-align: left;
    }
    /* 「新对话」：浅灰圆角大按钮 */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: #f1f2f6; border: 1px solid #e5e7eb; color: #111827;
        font-weight: 600; justify-content: center; box-shadow: none;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {background: #e8eaef;}
    /* 侧边栏底部用户 */
    .ia-avatar {
        width: 34px; height: 34px; background: #e5e7eb; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; font-size: 18px;
    }

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

    /* 状态容器 */
    div[data-testid="stStatusWidget"] {border-radius: 12px;}

    /* 对话 tab 底部输入区（千问式）：粘性置底，➕ 在输入框最左侧
       sticky 必须挂在 stLayoutWrapper 上——sticky 元素只能在父盒子内移动，
       输入条自己的父盒与它等高，挂上去无法滑动 */
    [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> .st-key-ia-input-bar) {
        position: sticky; bottom: 0; z-index: 100; margin-top: auto;
        background: #fff;
    }
    .st-key-ia-input-bar {
        padding: 10px 2px 4px; margin-top: 12px;
    }
    .st-key-ia-input-bar [data-testid="stChatInput"] {
        background: #fff; border: 1px solid #e5e7eb;
        border-radius: 22px; box-shadow: 0 6px 24px rgba(17, 24, 39, .07);
    }
    .st-key-ia-input-bar [data-testid="stChatInput"] textarea {
        font-size: 15px; padding-top: 12px;
    }
    .st-key-ia-input-bar [data-testid="stPopoverButton"],
    .st-key-ia-input-bar [data-testid="stPopover"] > button {
        border: none !important; box-shadow: none !important;
        background: transparent !important; width: auto !important;
        font-size: 22px; padding: 4px 6px;
    }
    /* 主内容区自身作为滚动容器，输入条的 sticky 才能钉在视口底部 */
    section[data-testid="stMain"] {height: 100vh; max-height: 100vh; overflow-y: auto;}

    /* 内容不足一屏时也把输入区压到视口底部（仅对话 tab 面板）。
       注意不能改面板自身的 display——Streamlit 靠 emotion 类的 display:none 隐藏
       非激活面板，高优先级规则会把它顶掉；直接给面板的子块设最小高度即可 */
    [data-testid="stTabPanel"]:nth-child(2) > [data-testid="stVerticalBlock"] {
        min-height: calc(100vh - 150px);
    }
    /* 免责声明（输入卡片下方居中小字） */
    .st-key-ia-input-bar::after {
        content: "内容由 AI 生成，请注意核实";
        display: block; text-align: center; color: #9ca3af;
        font-size: 12px; padding: 2px 0 4px;
    }
</style>
"""


def ensure_session() -> str:
    """会话 thread_id：同一浏览器会话共享（含发票查验与追问），「新对话」重置。"""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    return st.session_state.thread_id


def _conv_store() -> ConversationStore:
    """对话历史库：与业务库同目录（data/chat_history.db），可随时删除。"""
    return ConversationStore(Path(get_app_config().db_path).parent / "chat_history.db")


def _conv_title(display: list[dict]) -> str:
    """会话标题：取首条用户输入（千问式）。"""
    for e in display:
        if e["role"] == "user":
            if e.get("text"):
                return e["text"][:18]
            if e.get("file_name"):
                return f"发票查验：{e['file_name'][:14]}"
    return "新对话"


def save_conversation() -> None:
    """当前对话落库（每轮结束调用，近期对话列表与消息恢复的数据源）。"""
    display = st.session_state.get("display") or []
    if not display:
        return
    _conv_store().save(st.session_state.thread_id, _conv_title(display), display)


def _make_progress_writer():
    """进度回调工厂：单个 placeholder 内按 NODE_LABELS 固定顺序重绘打勾行。

    verify ∥ dedup 并行后节点完成顺序不稳定，不能按事件到达顺序逐行追加。
    """

    done: dict[str, float | None] = {}
    ph = st.empty()

    def writer(node: str, handler) -> None:
        done[node] = node_latency(handler, node)
        lines = []
        for n in NODE_LABELS:
            if n not in done:
                continue
            lat = done[n]
            lines.append(f"✅ {NODE_LABELS[n]}" + (f"（{lat:.0f}ms）" if lat is not None else ""))
        ph.markdown("\n\n".join(lines))

    return writer


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


SUGGESTIONS = {
    "invoice": ["这张发票能报销吗？", "有哪些合规风险需要注意？", "政策依据是什么？"],
    "qa": ["能详细解释一下吗？", "能举个报销的例子吗？", "报销时需要哪些材料？"],
}


def render_user_message(fname: str | None, text: str | None) -> None:
    """用户消息：右对齐浅灰圆角气泡（千问式），替代 chat_message 气泡。"""
    import html as _html

    rows = []
    if fname:
        rows.append(f"📎 <code>{_html.escape(fname)}</code>")
    if text:
        rows.append(_html.escape(text).replace("\n", "<br>"))
    st.markdown(
        f'<div class="ia-user-row"><div class="ia-user-bubble">{"".join(rows)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_toolbar(idx: int, entry: dict, is_last: bool) -> None:
    """回答下方操作栏（千问式）：复制 / 点赞 / 点踩 / 重新生成（仅最后一条）。"""
    cols = st.columns([1, 0.6, 0.6, 1.4, 3])
    if cols[0].button("📋 复制原文", key=f"copy_{idx}", use_container_width=True):
        st.session_state[f"_show_copy_{idx}"] = True
    fb = st.session_state.get(f"_fb_{idx}")
    if cols[1].button("👍", key=f"up_{idx}", use_container_width=True,
                      type="primary" if fb == 1 else "secondary"):
        st.session_state[f"_fb_{idx}"] = 1
        st.toast("感谢反馈！")
    if cols[2].button("👎", key=f"down_{idx}", use_container_width=True,
                      type="primary" if fb == -1 else "secondary"):
        st.session_state[f"_fb_{idx}"] = -1
        st.toast("感谢反馈，我们会持续改进。")
    if is_last and cols[3].button("🔄 重新生成", key=f"regen_{idx}", use_container_width=True):
        st.session_state["_regen_idx"] = idx
        st.rerun()
    if st.session_state.get(f"_show_copy_{idx}"):
        st.code(entry.get("answer") or "（无文本回答）", language="markdown")


def _render_followups(entry: dict) -> None:
    """最新回答下方的追问建议 chips，点击直接发送。"""
    cols = st.columns(3)
    for col, s in zip(cols, SUGGESTIONS[entry.get("kind", "qa")]):
        if col.button(s, key=f"fu_{s}", use_container_width=True):
            st.session_state["queued_prompt"] = s
            st.rerun()


def render_assistant_turn(idx: int, entry: dict, is_last: bool = False) -> None:
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

    _render_toolbar(idx, entry, is_last)


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


def render_conversation_tab() -> None:
    """对话时间线：用户/助手消息气泡（千问/豆包式，输入框固定在页面底部）。"""
    display: list[dict] = st.session_state.setdefault("display", [])
    if not display:
        render_empty_state()
        return
    entries = display[-MAX_DISPLAY_HISTORY:]
    last_assist = max((i for i, e in enumerate(entries) if e["role"] == "assistant"), default=None)
    for i, e in enumerate(entries):
        if e["role"] == "user":
            render_user_message(e.get("file_name"), e.get("text"))
        else:
            with st.chat_message("assistant", avatar="🧾"):
                render_assistant_turn(i, e, is_last=(i == last_assist))
    if last_assist is not None:
        _render_followups(entries[last_assist])


def run_turn(
    cfg, chat_slot, file_payload: tuple | None, text: str | None, append_user: bool = True
) -> None:
    """执行一轮对话：气泡内流式跑图，完成后写入 display 并刷新。

    chat_slot 为对话 tab 的容器（可重复进入），实时过程渲染在其中，
    与历史时间线保持同位。append_user=False 用于「重新生成」：用户气泡已在时间线中，
    不重复追加。
    """
    display: list[dict] = st.session_state.setdefault("display", [])
    fpath = None
    fname = None
    if file_payload:
        data, fname = file_payload
        up = cfg.upload
        suffix = Path(fname).suffix.lower()
        if suffix not in up.allowed_extensions:
            st.error(f"不支持的文件类型：{suffix}（支持 {up.allowed_extensions}）")
            return
        if len(data) > up.max_size_mb * 1024 * 1024:
            st.error(f"文件超过大小上限：{up.max_size_mb}MB")
            return
        updir = Path(cfg.db_path).parent / "uploads"
        updir.mkdir(parents=True, exist_ok=True)
        fpath = updir / f"{uuid.uuid4().hex[:8]}_{fname}"
        fpath.write_bytes(data)

    if append_user:
        display.append({"role": "user", "file_name": fname, "text": text or None})
    thread_id = st.session_state.thread_id

    if append_user:
        with chat_slot:
            render_user_message(fname, text)

    with chat_slot, st.chat_message("assistant", avatar="🧾"):
        with st.status("处理中…", expanded=True) as status:
            # 流式回答：首个 finalize token 到达时才建占位符（保证排在进度行下方）
            streamed: list[str] = []
            holder: dict = {}

            def on_token(piece) -> None:
                if isinstance(piece, list):  # 多模态 content 取文本片段
                    piece = "".join(p.get("text", "") for p in piece if isinstance(p, dict))
                if not piece:
                    return
                streamed.append(piece)
                if "ph" not in holder:
                    holder["ph"] = st.empty()
                holder["ph"].markdown("".join(streamed))

            state, handler = run_stream(
                str(fpath) if fpath else None,
                text or None,
                thread_id,
                progress_cb=_make_progress_writer(),
                on_token=on_token,
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
        render_assistant_turn(len(display), entry)

    display.append(entry)
    save_conversation()
    st.session_state.pop("pending_file", None)
    if fname:  # 上传的同名文件视为已消费，不再重复入队
        st.session_state["_consumed_upload"] = fname
    st.rerun()


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
    if detail["llm_calls"]:
        st.caption("LLM 调用明细")
        st.dataframe(
            [
                {
                    "节点": NODE_LABELS.get(c["node"], c["node"]),
                    "模型": c["model"] or "",
                    "输入 tokens": c["prompt_tokens"] or 0,
                    "输出 tokens": c["completion_tokens"] or 0,
                    "耗时(ms)": round(c["latency_ms"] or 0),
                    "成本(元)": round(c["cost_yuan"] or 0, 4),
                    "prompt": (
                        f"{c['prompt_name']}@{c['prompt_version']}"
                        if c["prompt_name"]
                        else ""
                    ),
                    "成功": "✓" if c["success"] else "✗",
                    "错误": c["error"] or "",
                }
                for c in detail["llm_calls"]
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

    st.subheader("⚖️ A/B 评估")
    try:
        from invoice_agent.monitoring.store import MonitorStore

        store = MonitorStore(get_app_config().monitoring_db_path)
        usage = store.prompt_usage()
    except Exception as exc:  # noqa: BLE001 — 监控降级时提示即可
        st.caption(f"监控不可用（{exc}）")
        return
    from invoice_agent.ui.prompt_eval import render_annotation_section, render_report_section

    tab_usage, tab_annot, tab_report = st.tabs(["调用分布", "🏷 人工标注", "📊 对比报告"])
    with tab_usage:
        rows = [u for u in usage if u["prompt_name"] == name]
        if not rows:
            st.caption("该 prompt 暂无调用记录")
        else:
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
    with tab_annot:
        render_annotation_section(store, name)
    with tab_report:
        render_report_section(store, name)


def render_input_bar() -> tuple[bool, str | None]:
    """对话 tab 底部输入区：➕ 附件在输入框最左侧，同一行；待发送条在上方。

    st.chat_input 在容器内为 inline 模式，随 tab 显示/隐藏；外层容器粘性置底。
    返回 (send_only, typed)。
    """
    pending = st.session_state.get("pending_file")
    send_only = False
    with st.container(key="ia-input-bar"):
        if pending:
            c_chip, c_send, c_remove = st.columns([5, 1, 1])
            c_chip.info(f"📎 待发送：**{pending['name']}**（可输入问题一起发出）")
            send_only = c_send.button("🚀 发送", type="primary", key="send_pending",
                                      use_container_width=True)
            if c_remove.button("移除", key="remove_pending", use_container_width=True):
                st.session_state.pop("pending_file", None)
                st.session_state["_consumed_upload"] = pending["name"]
                st.rerun()
        c_plus, c_input = st.columns([1, 11], gap="small", vertical_alignment="center")
        with c_plus.popover("➕"):
            st.caption("上传发票（图片/PDF）或选择示例")
            up = st.file_uploader("发票文件", type=["png", "jpg", "jpeg", "pdf"],
                                  key="_uploader", label_visibility="collapsed")
            # 上传后进入「待发送」，随下一条消息一起发出；发送过一次不再重复入队
            if up is not None and up.name != st.session_state.get("_consumed_upload"):
                st.session_state["pending_file"] = {"name": up.name, "bytes": up.getvalue()}
            for p in sorted(SAMPLES_DIR.glob("*.png"))[:4]:
                if st.button(p.name, key=f"sample_{p.name}", use_container_width=True):
                    st.session_state["pending_file"] = {"name": p.name, "bytes": p.read_bytes()}
                    st.rerun()
        with c_input:
            typed = st.chat_input("直接提问，也可以上传图片或文件…")
    return send_only, typed


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧾 发票助手")
        if st.button("＋ 新对话", use_container_width=True, type="primary"):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.display = []
            st.rerun()

        st.caption("🕘 近期对话")
        current = st.session_state.thread_id
        for conv in _conv_store().list_conversations(15):
            marker = "●  " if conv["thread_id"] == current else ""
            clicked = st.button(f"{marker}{conv['title']}", key=f"conv_{conv['thread_id']}",
                                use_container_width=True)
            if clicked and conv["thread_id"] != current:
                st.session_state.thread_id = conv["thread_id"]
                st.session_state.display = _conv_store().get_entries(conv["thread_id"])
                st.rerun()

        # 底部用户信息
        st.divider()
        c_avatar, c_name = st.columns([1, 4])
        c_avatar.markdown('<div class="ia-avatar">👤</div>', unsafe_allow_html=True)
        c_name.markdown("**本地用户**")
        st.caption("⚠️ 验真当前为 Mock，未接真实查验平台")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_session()
    cfg = get_app_config()
    render_sidebar()

    tab_chat, tab_reimburse, tab_monitor, tab_prompts = st.tabs(
        ["💬 助手对话", "🧾 报销提交", "📈 监控面板", "📜 Prompt 版本"]
    )

    # 重新生成：须在时间线渲染前处理——本 run 只实时渲染新气泡，避免元素 key 重复；
    # run_turn 结尾的 st.rerun 会跳过本次剩余渲染
    display = st.session_state.setdefault("display", [])
    regen_i = st.session_state.pop("_regen_idx", None)
    if regen_i is not None and display:
        base = len(display) - min(MAX_DISPLAY_HISTORY, len(display))
        a_idx = base + regen_i
        if 0 <= a_idx < len(display) and display[a_idx].get("role") == "assistant":
            old = display.pop(a_idx)
            fpath = old.get("file_path")
            payload = (Path(fpath).read_bytes(), Path(fpath).name) if fpath and Path(fpath).exists() else None
            text = next((e.get("text") for e in reversed(display[:a_idx]) if e["role"] == "user"), None)
            with tab_chat:
                run_turn(cfg, tab_chat, payload, text, append_user=False)

    with tab_chat:
        render_conversation_tab()
        # 本轮实时气泡的落点：必须在输入条之前建好——run_turn 在输入条渲染之后
        # 才被触发，若直接用 tab_chat，流式元素会追加到输入条的 DOM 之后，
        # 显示在输入框下方，直到 st.rerun 才归位
        live_slot = st.container()
        send_only, typed = render_input_bar()
    with tab_reimburse:
        render_reimburse_tab(cfg)
    with tab_monitor:
        render_monitor_tab()
    with tab_prompts:
        render_prompts_tab()

    queued = st.session_state.pop("queued_prompt", None)
    text = (typed or queued or "").strip()
    pending_now = st.session_state.get("pending_file")
    payload = (pending_now["bytes"], pending_now["name"]) if pending_now else None

    if send_only:
        with tab_chat:
            run_turn(cfg, live_slot, payload, None)  # 仅发票：识票查验流程
    elif text:
        with tab_chat:
            run_turn(cfg, live_slot, payload, text)


main()
