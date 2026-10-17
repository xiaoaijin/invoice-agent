"""Prompt A/B 评估：人工标注工作区与版本对比报告（「Prompt 版本」tab 内嵌使用）。

标注粒度为一次 LLM 调用（llm_calls.id）：A/B 各版本的每次真实输出都可单独打分
（1~5）并备注，保存后进入 version_report 聚合，形成调用指标 + 人工分的对比报告。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from invoice_agent.monitoring.store import MonitorStore

# 展示用列（llm_calls 原始列翻译成中文，disabled 不可编辑）
_DISPLAY_COLS = ["时间", "版本", "节点", "耗时(ms)", "成功", "评分", "备注"]
_HIDE_COLS = ["llm_call_id"]  # 随 df 携带、不进 data_editor 的定位键


def _load_calls_df(store: MonitorStore, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """拉取近期调用并转成编辑器 DataFrame；返回 (展示 df, 保存前原始 df)。"""
    rows = store.recent_calls_for_prompt(name)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_DISPLAY_COLS + _HIDE_COLS), pd.DataFrame(
            columns=_DISPLAY_COLS + _HIDE_COLS
        )
    df["时间"] = df["started_at"].fillna("").astype(str).str[:19]
    df["版本"] = df["prompt_version"].fillna("?")
    df["节点"] = df["node"].fillna("")
    df["耗时(ms)"] = (df["latency_ms"] // 1).astype("Int64")
    df["成功"] = df["success"].map({1: "✓", 0: "✗"}).fillna("?")
    df["评分"] = df["rating"].astype("Int64")  # NaN 保持空
    df["备注"] = df["note"].fillna("")
    display = df[_DISPLAY_COLS + _HIDE_COLS]
    return display, display.copy()


def render_annotation_section(store: MonitorStore, name: str) -> None:
    """人工标注工作区：近期调用逐条评分 + 备注，一次保存。"""
    st.caption(
        "对每条真实输出打分（1=差 → 5=好），已标注的行会带出旧值，修改后统一保存覆盖。"
    )
    display, original = _load_calls_df(store, name)
    if display.drop(columns=_HIDE_COLS).empty:
        st.caption(f"prompt {name} 暂无调用记录，先在对话里跑几轮再回来标注")
        return

    edited = st.data_editor(
        display.drop(columns=_HIDE_COLS),
        use_container_width=True,
        hide_index=True,
        key=f"annot_editor_{name}",
        column_config={
            "评分": st.column_config.SelectboxColumn(
                "评分", options=[1, 2, 3, 4, 5], required=False
            ),
            "备注": st.column_config.TextColumn("备注"),
        },
        disabled=["时间", "版本", "节点", "耗时(ms)", "成功"],
    )
    annotated = int(original["评分"].notna().sum())
    st.caption(f"标注进度：{annotated} / {len(original)} 条已评分")

    if not st.button("💾 保存标注", key=f"annot_save_{name}", type="primary"):
        return
    # 只提交有变化的行（新评分或改动），llm_call_id 在两侧顺序一致
    saved = 0
    for i in range(len(edited)):
        new_rating, raw_note = edited.iloc[i]["评分"], edited.iloc[i]["备注"]
        if pd.isna(new_rating):
            continue  # 未评分的行不写库
        new_note = "" if pd.isna(raw_note) else str(raw_note).strip()
        old = original.iloc[i]
        old_rating = old["评分"]
        if new_rating == old_rating and new_note == str(old["备注"]):
            continue
        store.upsert_annotation(
            int(old["llm_call_id"]),
            int(new_rating),
            new_note or None,
            run_id=None if pd.isna(old["run_id"]) else str(old["run_id"]),
            prompt_name=name,
            prompt_version=old["版本"],
        )
        saved += 1
    if saved:
        st.toast(f"已保存 {saved} 条标注")
        st.rerun()
    else:
        st.info("没有需要保存的修改")


def render_report_section(store: MonitorStore, name: str) -> None:
    """对比报告：调用指标 + 人工分按版本聚合，附 Markdown 下载。"""
    rows = store.version_report(name)
    if not rows:
        st.caption("暂无调用记录，无法生成对比报告")
        return
    st.dataframe(
        [
            {
                "版本": r["prompt_version"],
                "调用量": r["calls"],
                "成功率": f"{r['success_rate']:.0%}" if r["success_rate"] is not None else "—",
                "平均耗时(ms)": round(r["avg_latency_ms"]) if r["avg_latency_ms"] is not None else "—",
                "成本(元)": round(r["cost_yuan"] or 0, 4),
                "标注数": r["annotated"],
                "平均人工分": f"{r['avg_rating']:.2f}" if r["avg_rating"] is not None else "—",
            }
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    rated = [r for r in rows if r["avg_rating"] is not None and r["annotated"] > 0]
    if rated:
        best = max(rated, key=lambda r: (r["avg_rating"], r["annotated"]))
        st.success(
            f"🏷 按人工分暂由 **{best['prompt_version']}** 领先（{best['avg_rating']:.2f} 分，"
            f"{best['annotated']} 条标注）——样本少时仅作参考"
        )
    else:
        st.caption("还没有人工标注，先在「人工标注」页打分，报告会自动纳入平均分")
    st.download_button(
        "⬇️ 下载 Markdown 报告",
        data=_report_markdown(name, rows),
        file_name=f"prompt_ab_report_{name}.md",
        mime="text/markdown",
        key=f"report_dl_{name}",
    )


def _report_markdown(name: str, rows: list[dict]) -> str:
    """把 version_report 行渲染成 Markdown 报告文本。"""
    lines = [f"# Prompt A/B 对比报告：{name}", ""]
    lines += [
        "| 版本 | 调用量 | 成功率 | 平均耗时(ms) | 成本(元) | 标注数 | 平均人工分 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {ver} | {calls} | {rate} | {lat} | {cost} | {annot} | {rating} |".format(
                ver=r["prompt_version"],
                calls=r["calls"],
                rate=f"{r['success_rate']:.0%}" if r["success_rate"] is not None else "—",
                lat=round(r["avg_latency_ms"]) if r["avg_latency_ms"] is not None else "—",
                cost=f"{r['cost_yuan'] or 0:.4f}",
                annot=r["annotated"],
                rating=f"{r['avg_rating']:.2f}" if r["avg_rating"] is not None else "—",
            )
        )
    rated = [r for r in rows if r["avg_rating"] is not None and r["annotated"] > 0]
    if rated:
        best = max(rated, key=lambda r: (r["avg_rating"], r["annotated"]))
        lines += [
            "",
            (
                f"结论：按人工分 **{best['prompt_version']}** 领先"
                f"（{best['avg_rating']:.2f} 分，{best['annotated']} 条标注；样本少时仅作参考）。"
            ),
        ]
    return "\n".join(lines)
