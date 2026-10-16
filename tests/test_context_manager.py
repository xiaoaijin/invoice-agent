"""上下文管理测试：token 估算、60% 压缩触发、保留窗口、工具紧凑保留、预算截断。"""

from invoice_agent.config import ContextConfig
from invoice_agent.services.context_manager import (
    _render_entry,
    _split_keep_window,
    build_summary_prompt,
    compress_history,
    estimate_tokens,
    fit_budget,
)

CONFIG = ContextConfig(token_budget=8000, compress_threshold=0.6, keep_recent_rounds=10)


def make_history(rounds: int, filler: str = "这句话大约二十个汉字字符左右吧") -> list[dict]:
    history = []
    for i in range(rounds):
        history.append({"role": "user", "content": f"问题{i}：{filler}"})
        history.append({"role": "assistant", "content": f"回答{i}：{filler}"})
    return history


class TestEstimateTokens:
    def test_cjk_one_token_per_char(self):
        assert estimate_tokens("发票报销") == 4 + 1

    def test_ascii_quarter(self):
        # 8 个 ASCII 字符 ≈ 2 tokens + 1
        assert estimate_tokens("abcdefgh") == 2 + 1

    def test_mixed(self):
        tokens = estimate_tokens("发票invoice报销")
        assert tokens == 4 + 7 // 4 + 1  # 4 CJK + 7 ASCII


class TestSplitKeepWindow:
    def test_within_window(self):
        old, recent = _split_keep_window(make_history(5), keep_rounds=10)
        assert old == []
        assert len(recent) == 10

    def test_split_at_round_boundary(self):
        history = make_history(15)
        old, recent = _split_keep_window(history, keep_rounds=10)
        assert len(old) == 10  # 5 轮 × 2
        assert len(recent) == 20  # 10 轮 × 2
        assert recent[0]["content"].startswith("问题5")


class TestCompressHistory:
    def test_no_compress_below_threshold(self):
        """历史 token 未达 60% 预算 → 不压缩。"""
        history = make_history(10)  # ~20 条 × ~30 tokens ≈ 600 < 4800
        state = {"chat_history": history}
        assert compress_history(state, lambda p, e: "摘要", CONFIG) == {}

    def test_compress_triggered_at_60_percent(self):
        """历史超预算 60% → 压缩：旧内容进摘要，最近 10 轮保留。"""
        history = make_history(300)  # ~300 chars/条 × 600 条 ≈ 18 万 tokens
        summarized = {}

        def fake_summarizer(prev, old_entries):
            summarized["prev"] = prev
            summarized["old"] = old_entries
            return "这是历史摘要"

        state = {"chat_history": history}
        result = compress_history(state, fake_summarizer, CONFIG)
        assert result["chat_history"] == history[-20:]  # 最近 10 轮原样保留
        assert result["history_summary"] == "这是历史摘要"
        assert len(summarized["old"]) == len(history) - 20

    def test_rolling_summary_includes_prev(self):
        """已有摘要时再次压缩，旧摘要应传给摘要器。"""
        history = make_history(300)
        calls = {}

        def fake_summarizer(prev, old_entries):
            calls["prev"] = prev
            return "新摘要"

        state = {"chat_history": history, "history_summary": "旧摘要"}
        result = compress_history(state, fake_summarizer, CONFIG)
        assert calls["prev"] == "旧摘要"
        assert result["history_summary"] == "新摘要"

    def test_tool_entries_in_old_history(self):
        """被压缩的工具调用条目会传给摘要器（摘要器负责只留工具名+结果）。"""
        history = [
            {"role": "user", "content": "查一下这张发票"},
            {"role": "tool", "tool": "invoice_verify", "content": "PASS: 查验通过"},
            {"role": "tool", "tool": "risk_rules", "content": "R005(高)"},
            {"role": "assistant", "content": "结论……"},
        ] + make_history(300)
        captured = {}

        def fake_summarizer(prev, old_entries):
            captured["entries"] = old_entries
            return "摘要"

        compress_history({"chat_history": history}, fake_summarizer, CONFIG)
        tool_entries = [e for e in captured["entries"] if e.get("role") == "tool"]
        assert any(e["tool"] == "invoice_verify" for e in tool_entries)

    def test_threshold_custom(self):
        """阈值可配置：小预算下少量历史即触发。"""
        cfg = ContextConfig(token_budget=100, compress_threshold=0.6, keep_recent_rounds=2)
        history = make_history(10)
        result = compress_history({"chat_history": history}, lambda p, e: "S", cfg)
        assert result["history_summary"] == "S"
        assert result["chat_history"] == history[-4:]


class TestToolRendering:
    def test_render_tool_entry(self):
        assert "[工具 invoice_verify] PASS" == _render_entry(
            {"role": "tool", "tool": "invoice_verify", "content": "PASS"}
        )


class TestFitBudget:
    def test_within_budget_unchanged(self):
        text = "短文本"
        assert fit_budget(text, 8000) == text

    def test_truncated_to_budget(self):
        text = "字符" * 10000  # ~20000 tokens
        result = fit_budget(text, 8000)
        assert estimate_tokens(result) <= 8000
        assert result.endswith(text[-10:])  # 保留后半段
        assert result.startswith("（前文已因超长截断")


class TestSummaryPromptParams:
    def test_placeholder_template_filled_into_system(self):
        """summarize v2 参数化模板：此前摘要与对话填入 system prompt（真实 registry 文本）。"""
        messages = build_summary_prompt(
            "此前讨论过发票查验",
            [
                {"role": "user", "content": "餐饮发票能抵扣吗"},
                {"role": "assistant", "content": "不能抵扣"},
            ],
            None,  # 从真实 registry 取 summarize 当前版本
        )
        system_text = messages[0].content
        human_text = messages[-1].content
        assert "{prev_summary}" not in system_text
        assert "{conversation}" not in system_text
        assert "此前讨论过发票查验" in system_text
        assert "餐饮发票能抵扣吗" in system_text
        assert human_text  # user 消息非空（触发摘要）

    def test_legacy_prompt_keeps_human_payload(self):
        """v1 非参数化文本：摘要与对话仍走 human 消息。"""
        legacy = "你是对话摘要器。"
        messages = build_summary_prompt(
            "旧摘要", [{"role": "user", "content": "问题"}], legacy
        )
        assert messages[0].content == legacy
        assert "旧摘要" in messages[-1].content
