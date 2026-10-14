"""发票助手 Agent — Streamlit 界面。

运行：uv run streamlit run src/invoice_agent/ui/app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

# streamlit run 以脚本方式执行，需手动把 src 加入 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from invoice_agent.config import get_app_config
from invoice_agent.services.dedup_store import DedupStore

st.set_page_config(page_title="发票助手 Agent", page_icon="🧾", layout="wide")


def run_graph(file_path: str | None, question: str | None) -> dict:
    from invoice_agent.graph.builder import build_graph

    graph = build_graph()
    return graph.invoke(
        {"session_id": "streamlit", "file_path": file_path, "user_question": question},
        config={"configurable": {"thread_id": "streamlit"}},
    )


def render_invoice_card(invoice) -> None:
    st.markdown("### 📋 发票概览")
    c1, c2, c3 = st.columns(3)
    c1.metric("发票类型", invoice.invoice_type.value)
    c1.metric("发票号码", invoice.invoice_number)
    c2.metric("开票日期", str(invoice.invoice_date))
    c2.metric("价税合计", f"¥{invoice.total_amount:,.2f}")
    c3.metric("不含税金额", f"¥{invoice.amount_without_tax:,.2f}" if invoice.amount_without_tax else "—")
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


def main() -> None:
    st.title("🧾 发票助手 Agent")
    st.caption("识票 · 验真 · 风控 · 政策问答（验真当前为 Mock，未接真实查验平台）")

    cfg = get_app_config()
    tab_invoice, tab_ask = st.tabs(["📤 发票查验", "💬 政策问答"])

    with tab_invoice:
        uploaded = st.file_uploader("上传发票（图片或 PDF）", type=["png", "jpg", "jpeg", "pdf"])
        question = st.text_input(
            "附加问题（可选）", placeholder="例如：这张发票能报销吗？餐饮部分能抵扣吗？"
        )
        if st.button("开始查验", type="primary", disabled=uploaded is None):
            if uploaded is None:
                st.error("请先上传发票文件")
            else:
                tmp = Path(cfg.db_path).parent / "uploads"
                tmp.mkdir(parents=True, exist_ok=True)
                fpath = tmp / uploaded.name
                fpath.write_bytes(uploaded.getvalue())
                with st.spinner("识别与校验中…"):
                    state = run_graph(str(fpath), question or None)
                st.session_state["last_state"] = {
                    "invoice": state.get("invoice"),
                    "verification": state.get("verification"),
                    "dups": state.get("duplicate_hits") or [],
                    "findings": state.get("risk_findings") or [],
                    "answer": state.get("answer"),
                    "file_path": str(fpath),
                    "invoice_dump": state.get("invoice").model_dump(mode="json") if state.get("invoice") else None,
                }

        state = st.session_state.get("last_state")
        if state:
            if state["invoice"] is None:
                st.error(f"发票识别失败：{state.get('answer', '未知错误')}")
            else:
                render_invoice_card(state["invoice"])

                ver = state["verification"]
                st.markdown("### 🔍 验真结果")
                if ver is not None:
                    if ver.status == "PASS":
                        st.success(f"✔ 查验通过（{ver.provider}）— {ver.message}")
                    elif ver.status == "FAIL":
                        st.error(f"✘ 查验不符（{ver.provider}）— {ver.message}")
                    else:
                        st.warning(f"? 查验异常（{ver.provider}）— {ver.message}")
                    st.caption("⚠️ 当前为 Mock 验真，结果不代表真实真伪")

                st.markdown("### 🛡 风控预警")
                if state["findings"]:
                    for f in state["findings"]:
                        icon = {"高": "🔴", "中": "🟡", "低": "🔵"}[f.level.value]
                        with st.container(border=True):
                            st.markdown(f"{icon} **[{f.level.value}] {f.rule_id} {f.rule_name}**")
                            st.write(f.message)
                else:
                    st.success("未触发风控规则")

                if state["answer"]:
                    st.markdown("### 📝 综合结论")
                    st.markdown(state["answer"])

                st.divider()
                if st.button("✅ 确认提交报销（写入台账）", type="primary"):
                    inv = state["invoice"]
                    import hashlib

                    sha = hashlib.sha256(Path(state["file_path"]).read_bytes()).hexdigest()
                    DedupStore(cfg.db_path).save_reimbursement(
                        invoice_number=inv.invoice_number,
                        invoice_code=inv.invoice_code,
                        seller_tax_id=inv.seller_tax_id,
                        buyer_tax_id=inv.buyer_tax_id,
                        total_amount=inv.total_amount,
                        invoice_date=date.fromisoformat(str(inv.invoice_date)),
                        file_sha256=sha,
                    )
                    st.success("已写入报销台账，重复报销检测即刻生效")

    with tab_ask:
        q = st.text_area("输入政策问题", placeholder="例如：差旅补贴标准是多少？购物卡发票能不能报销？")
        if st.button("提问", type="primary", disabled=not q.strip()) and q.strip():
            with st.spinner("检索政策条款并生成回答…"):
                state = run_graph(None, q.strip())
            st.markdown(state.get("answer") or "（无回答）")
            if state.get("rag_chunks"):
                with st.expander("查看检索到的条款"):
                    for c in state["rag_chunks"]:
                        article = c.article_no or ""
                        st.markdown(f"**《{c.source}》{c.doc_title} {article}**（score={c.score}）")
                        st.text(c.text[:300])


main()
