"""报销提交 tab：多文件上传 → 逐张识别 → 分类统计预览（可剔除）→ 确认提交审批。

向导式流程（有草稿态与确认门），独立于对话 tab 的会话状态机：
- reimburse_preview：识别完成后的 ReimburseBatch 草稿
- reimburse_result：提交后的 SubmissionResult
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from invoice_agent.config import AppConfig
from invoice_agent.models.reimburse import ReimburseBatch
from invoice_agent.runner import NODE_LABELS
from invoice_agent.services.reimburse_flow import build_summary, run_batch, submit_batch


def _included(batch: ReimburseBatch, excluded: set[str]) -> int:
    return sum(
        1
        for it in batch.items
        if it.invoice is not None and it.invoice.invoice_number not in excluded
    )


def render_reimburse_tab(cfg: AppConfig) -> None:
    st.session_state.setdefault("reimburse_preview", None)
    st.session_state.setdefault("reimburse_result", None)
    if st.session_state.reimburse_result is not None:
        _render_result()
    elif st.session_state.reimburse_preview is not None:
        _render_preview(cfg)
    else:
        _render_upload(cfg)


def _reset() -> None:
    st.session_state.reimburse_preview = None
    st.session_state.reimburse_result = None


def _render_upload(cfg: AppConfig) -> None:
    st.markdown("上传多张发票（图片/PDF），自动识票、分类统计，确认后提交报销审批。")
    st.caption("⚠️ 飞书审批当前为 Mock 提交（模拟成功，返回模拟实例 ID）")
    title = st.text_input("报销单标题（可选，默认按日期生成）")
    ups = st.file_uploader(
        "发票文件（可多选）",
        type=[ext.lstrip(".") for ext in cfg.upload.allowed_extensions],
        accept_multiple_files=True,
    )
    if not ups:
        return
    valid = []
    for up in ups:
        suffix = Path(up.name).suffix.lower()
        if suffix not in cfg.upload.allowed_extensions:
            st.error(f"不支持的文件类型：{up.name}（{suffix}）")
            continue
        if len(up.getvalue()) > cfg.upload.max_size_mb * 1024 * 1024:
            st.error(f"文件超过大小上限 {cfg.upload.max_size_mb}MB：{up.name}")
            continue
        valid.append(up)
    if not valid:
        return
    st.info(f"已选择 {len(valid)} 个文件")
    if not st.button("🔍 开始识别", type="primary"):
        return
    updir = Path(cfg.db_path).parent / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    paths = []
    for up in valid:
        p = updir / f"{uuid.uuid4().hex[:8]}_{up.name}"
        p.write_bytes(up.getvalue())
        paths.append(p)

    progress = st.empty()
    with st.status(f"批量识别中（0/{len(paths)}）…", expanded=True) as status:

        def cb(i: int, fname: str, node: str) -> None:
            progress.markdown(f"**{i}/{len(paths)}** `{fname}`：{NODE_LABELS.get(node, node)}")

        batch = run_batch(paths, title=title or None, progress_cb=cb)
        status.update(label=f"识别完成（{len(paths)} 张）", state="complete", expanded=False)
    st.session_state.reimburse_preview = batch
    st.rerun()


def _render_preview(cfg: AppConfig) -> None:
    batch: ReimburseBatch = st.session_state.reimburse_preview
    st.markdown("##### 📋 分类统计预览")
    st.caption("展开每张发票核对信息，不需要报销的勾选「不纳入本次报销」；合计将实时重算。")

    excluded: set[str] = set()
    for idx, item in enumerate(batch.items):
        inv = item.invoice
        if inv is None:
            with st.container(border=True):
                st.error(f"❌ {item.file_name}：{item.error or '识别失败，请人工复核'}")
            continue
        header = f"📄 {item.file_name}｜{item.category.value}｜¥{inv.total_amount:,.2f}"
        with st.expander(header):
            flags = []
            if item.verification_status and item.verification_status != "PASS":
                flags.append(f"验真 {item.verification_status}")
            if item.has_duplicates:
                flags.append("疑似重复报销")
            if item.risk_level:
                flags.append(f"风险预警（{item.risk_level}）")
            if flags:
                st.warning("⚠️ " + "；".join(flags))
            st.write(
                f"发票号 `{inv.invoice_number}`｜开票日期 {inv.invoice_date}｜"
                f"销售方 {inv.seller_name}｜类别来源 {item.category_source}"
            )
            if st.checkbox("不纳入本次报销", key=f"reimburse_exclude_{idx}"):
                excluded.add(inv.invoice_number)

    current = build_summary(
        batch.items, batch_id=batch.batch_id, title=batch.title, exclude_numbers=excluded
    )
    c1, c2 = st.columns(2)
    c1.metric("提交张数", _included(batch, excluded))
    c2.metric("合计金额", f"¥{current.total_amount:,.2f}")
    if current.category_totals:
        st.dataframe(
            [
                {
                    "类别": t.category.value,
                    "张数": t.count,
                    "金额小计": f"¥{t.total_amount:,.2f}",
                }
                for t in current.category_totals
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    b1, b2, _ = st.columns([1.2, 1, 4])
    n = _included(batch, excluded)
    if b1.button("✅ 确认提交报销", type="primary", disabled=(n == 0)):
        with st.spinner("提交审批流程中…"):
            result = submit_batch(current)
        st.session_state.reimburse_result = result
        st.session_state.reimburse_preview = None
        st.rerun()
    if b2.button("清空重来"):
        _reset()
        st.rerun()


def _render_result() -> None:
    r = st.session_state.reimburse_result
    st.success(f"✔ 报销单已提交（{r.provider}）")
    c1, c2 = st.columns(2)
    c1.metric("审批实例 ID", r.instance_id or "—")
    c2.metric("状态", r.status)
    st.write(r.message)
    st.caption("⚠️ 当前为 Mock 提交，实例 ID 非真实飞书审批单")
    if st.button("🧾 再提一批"):
        _reset()
        st.rerun()
