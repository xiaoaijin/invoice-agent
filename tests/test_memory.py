"""多轮对话记忆测试：验证 checkpointer 跨 invoke 保留 chat_history、追问语义、历史注入。"""

from datetime import date
from pathlib import Path

from invoice_agent.graph import builder as builder_mod
from invoice_agent.graph.nodes import answer as answer_mod
from invoice_agent.models.invoice import InvoiceData, InvoiceTypeEnum

CAPTURED: dict = {}


def make_invoice(**kw) -> InvoiceData:
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number="60000001",
        invoice_date=date(2026, 10, 1),
        buyer_name="示例科技有限公司",
        buyer_tax_id="91330100MA27XXXXXX",
        seller_name="卖方公司",
        seller_tax_id="91330100MA28YYYYYY",
        amount_without_tax=100.0,
        tax_amount=13.0,
        total_amount=113.0,
        items=[],
    )
    base.update(kw)
    return InvoiceData(**base)


def _patch_env(monkeypatch, tmp_path: Path, llm_answer: str = "OK") -> None:
    """打桩：临时 SQLite + finalize 不调 LLM 但走真实记忆逻辑。"""
    from invoice_agent.config import AppConfig, CompanyConfig, PathsConfig, RiskConfig
    from invoice_agent.graph.nodes import dedup, intent, risk, verify

    fake_cfg = AppConfig(
        company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
        risk=RiskConfig(),
        paths=PathsConfig(
            db=str(tmp_path / "m.db"),
            knowledge_dir=str(tmp_path / "knowledge"),
        ),
    )
    for mod in (dedup, intent, risk, verify):
        monkeypatch.setattr(mod, "get_app_config", lambda: fake_cfg)
    # 意图 LLM 兜底禁用：多轮追问句式已由正则/规则层覆盖，不应触达真实 LLM
    monkeypatch.setattr(intent, "default_llm_fn", lambda: None)
    monkeypatch.setattr(answer_mod, "retrieve_docs", lambda state: {"rag_chunks": []})

    def fake_finalize(state):
        CAPTURED["history_seen"] = state.get("chat_history")
        answer = f"{llm_answer}::{state.get('user_question')}"
        return {
            "answer": answer,
            "chat_history": answer_mod._remember(state, state.get("user_question"), answer),
        }

    monkeypatch.setattr(answer_mod, "finalize", fake_finalize)


def _fresh_graph():
    builder_mod.build_graph.cache_clear()
    return builder_mod.build_graph()


def _invoke(graph, question: str, thread_id: str):
    return graph.invoke(
        {"session_id": thread_id, "file_path": None, "user_question": question},
        config={"configurable": {"thread_id": thread_id}},
    )


class TestMemory:
    def test_history_accumulates_across_invokes(self, tmp_path, monkeypatch):
        _patch_env(monkeypatch, tmp_path)
        g = _fresh_graph()
        s1 = _invoke(g, "餐饮发票能抵扣吗", "sess-a")
        assert len(s1["chat_history"]) == 2  # 1 轮问答

        s2 = _invoke(g, "那购物卡呢", "sess-a")
        assert len(s2["chat_history"]) == 4  # 2 轮问答
        assert s2["chat_history"][0] == {"role": "user", "content": "餐饮发票能抵扣吗"}
        # finalize 能看到上一轮完整问答
        assert CAPTURED["history_seen"] == s1["chat_history"]

    def test_separate_threads_isolated(self, tmp_path, monkeypatch):
        _patch_env(monkeypatch, tmp_path)
        g = _fresh_graph()
        _invoke(g, "问题A", "sess-1")
        s2 = _invoke(g, "问题B", "sess-2")
        # 不同 thread_id 记忆隔离
        assert len(s2["chat_history"]) == 2
        assert s2["chat_history"][0]["content"] == "问题B"

    def test_history_capped_at_limit(self, tmp_path, monkeypatch):
        _patch_env(monkeypatch, tmp_path)
        g = _fresh_graph()
        for i in range(6):
            _invoke(g, f"第{i}个问题", "sess-c")
        s = _invoke(g, "最后问题", "sess-c")
        assert len(s["chat_history"]) == 14  # 7 轮 × 2
        # finalize 看到的是本轮之前的历史（6 轮 × 2），当前轮在 _remember 中追加
        assert len(CAPTURED["history_seen"]) == 12

    def test_follow_up_query_includes_prev_question(self, tmp_path, monkeypatch):
        """追问（短问题/「那」开头）检索 query 应拼上上一问题。"""
        state = {
            "chat_history": [
                {"role": "user", "content": "餐饮发票能抵扣吗"},
                {"role": "assistant", "content": "不能抵扣……"},
            ],
            "user_question": "那出租车票呢",
        }
        q = answer_mod._build_query(state)
        assert "餐饮发票能抵扣吗" in q
        assert "那出租车票呢" in q

    def test_independent_query_no_prev(self):
        """独立长问题不拼接上一问题。"""
        state = {
            "chat_history": [{"role": "user", "content": "餐饮发票能抵扣吗"}],
            "user_question": "公司差旅费报销标准具体是多少，请详细说明各项补贴的限额",
        }
        q = answer_mod._build_query(state)
        assert "餐饮发票能抵扣吗" not in q

    def test_query_with_invoice_summary(self):
        state = {"invoice": make_invoice(), "user_question": "能报销吗"}
        q = answer_mod._build_query(state)
        assert "增值税专用发票" in q
        assert "能报销吗" in q

    def test_intent_follow_up_overrides_heuristic(self):
        """意图字段存在时优先于启发式：follow_up → 拼上一轮。"""
        state = {
            "intent": {"retrieval_type": "follow_up", "domain": "deduction", "source": "llm"},
            "chat_history": [{"role": "user", "content": "餐饮发票能抵扣吗"}],
            "user_question": "出租车票呢",
        }
        q = answer_mod._build_query(state)
        assert "餐饮发票能抵扣吗" in q

    def test_intent_new_question_overrides_short_heuristic(self):
        """意图判定新问题（即使短问题，启发式本会判追问）→ 不拼上一轮。"""
        state = {
            "intent": {"retrieval_type": "new_question", "domain": "policy", "source": "llm"},
            "chat_history": [{"role": "user", "content": "餐饮发票能抵扣吗"}],
            "user_question": "货车通行费呢",
        }
        q = answer_mod._build_query(state)
        assert "餐饮发票能抵扣吗" not in q

    def test_history_messages_format(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "tool": "invoice_verify", "content": "PASS: ok"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        msgs = answer_mod._history_messages(history)
        assert len(msgs) == 5
        assert msgs[0].content == "q1" and type(msgs[0]).__name__ == "HumanMessage"
        assert msgs[1].content == "a1" and type(msgs[1]).__name__ == "AIMessage"
        assert "[工具 invoice_verify] PASS: ok" in msgs[2].content
