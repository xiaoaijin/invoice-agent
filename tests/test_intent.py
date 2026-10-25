"""意图识别漏斗单测：关键词/正则/规则层零 LLM（断言 llm_fn 未被调用），LLM 兜底注入假 llm_fn。"""

import pytest

from invoice_agent.config import (
    AppConfig,
    CompanyConfig,
    IntentConfig,
    PathsConfig,
    RiskConfig,
)
from invoice_agent.graph.nodes import intent as intent_mod
from invoice_agent.graph.nodes.intent import IntentResult, _classify, classify_intent
from invoice_agent.graph.nodes.route import route_entry

HISTORY = [
    {"role": "user", "content": "餐饮发票能抵扣吗"},
    {"role": "assistant", "content": "不能抵扣……"},
]


def _cfg(**kw) -> IntentConfig:
    return IntentConfig(**kw)


@pytest.fixture
def no_llm(monkeypatch):
    """把漏斗的 LLM 兜底替换为会失败可见的桩（被调用即抛错），用于断言零 LLM 路径。"""
    calls = {"n": 0}

    def boom(messages):
        calls["n"] += 1
        raise RuntimeError("不应触发 LLM 兜底")

    monkeypatch.setattr(intent_mod, "default_llm_fn", lambda: boom)
    return calls


class TestKeywordLayer:
    def test_chat_keyword_hits_without_llm(self, no_llm):
        r = _classify("你好", [], _cfg())
        assert r.retrieval_type == "chat"
        assert r.source == "keyword"
        assert r.domain == "chat"
        assert no_llm["n"] == 0

    def test_chat_keyword_with_domain_tag(self, no_llm):
        r = _classify("谢谢，报销流程真复杂", HISTORY, _cfg())
        assert r.retrieval_type == "chat"
        assert r.domain == "reimburse"
        assert no_llm["n"] == 0

    def test_domain_tagged_on_question(self, no_llm):
        r = _classify("发票抬头怎么写", [], _cfg())
        assert r.domain == "header"
        assert r.retrieval_type == "new_question"
        assert no_llm["n"] == 0


class TestNoRetrievalKeywordLayer:
    def test_keyword_hits_with_history(self, no_llm):
        r = _classify("刚才的结论再说一遍", HISTORY, _cfg())
        assert r.retrieval_type == "no_retrieval"
        assert r.source == "keyword"
        assert no_llm["n"] == 0

    def test_keyword_hits_first_turn(self, no_llm):
        """无上文命中也判 no_retrieval（与 chat 层不要求上文的现状一致）。"""
        r = _classify("再说一遍", [], _cfg())
        assert r.retrieval_type == "no_retrieval"
        assert r.source == "keyword"
        assert no_llm["n"] == 0

    def test_keyword_with_domain_tag(self, no_llm):
        r = _classify("重复一遍刚才的报销流程", HISTORY, _cfg())
        assert r.retrieval_type == "no_retrieval"
        assert r.domain == "reimburse"
        assert no_llm["n"] == 0

    def test_no_false_positive_on_real_question(self, no_llm):
        """未命中关键词的真实问题不被误伤，且不被追问正则抢走（「再说」开头但含新诉求）。"""
        r = _classify("再说一下报销流程", HISTORY, _cfg())
        assert r.retrieval_type == "follow_up"
        assert no_llm["n"] == 0

    def test_layer_order_beats_follow_up_regex(self, no_llm):
        """层序护栏：「再说一遍」命中追问正则 ^(再|重新|补充)(说…) 前缀，但免检索层在前。"""
        r = _classify("再说一遍", HISTORY, _cfg())
        assert r.retrieval_type == "no_retrieval"
        assert no_llm["n"] == 0


class TestRegexLayer:
    def test_prefix_follow_up_with_history(self, no_llm):
        r = _classify("那出租车票呢", HISTORY, _cfg())
        assert r.retrieval_type == "follow_up"
        assert r.source == "regex"
        assert no_llm["n"] == 0

    def test_ne_suffix_with_history(self, no_llm):
        r = _classify("出租车票呢？", HISTORY, _cfg())
        assert r.retrieval_type == "follow_up"
        assert no_llm["n"] == 0

    def test_first_turn_no_follow_up(self, no_llm):
        """首轮无指代对象：追问句式直接按新问题处理（规则短路，零 LLM）。"""
        r = _classify("那出租车票呢", [], _cfg())
        assert r.retrieval_type == "new_question"
        assert r.source == "rule"
        assert no_llm["n"] == 0


class TestRuleShortCircuit:
    def test_empty_history_new_question(self, no_llm):
        r = _classify("餐饮发票能抵扣吗", [], _cfg())
        assert r.retrieval_type == "new_question"
        assert r.source == "rule"
        assert no_llm["n"] == 0

    def test_long_question_with_domain(self, no_llm):
        """长问题（≥20 字）且领域已命中 → 独立新问题，不调 LLM。"""
        q = "公司员工出差取得的餐饮发票进项税额能否抵扣，具体政策依据和报销标准是什么"
        assert len(q) >= 20
        r = _classify(q, HISTORY, _cfg())
        assert r.retrieval_type == "new_question"
        assert r.source == "rule"
        assert no_llm["n"] == 0


class TestLlmFallback:
    def test_llm_called_and_parsed(self, monkeypatch):
        monkeypatch.setattr(
            intent_mod,
            "default_llm_fn",
            lambda: lambda m: '{"retrieval_type": "follow_up", "domain": "deduction"}',
        )
        r = _classify("餐饮发票全额抵扣吗", HISTORY, _cfg())
        assert r.retrieval_type == "follow_up"
        assert r.domain == "deduction"
        assert r.source == "llm"

    def test_llm_no_retrieval_parsed(self, monkeypatch):
        monkeypatch.setattr(
            intent_mod,
            "default_llm_fn",
            lambda: lambda m: '{"retrieval_type": "no_retrieval", "domain": "reimburse"}',
        )
        r = _classify("把要点总结一下", HISTORY, _cfg())
        assert r.retrieval_type == "no_retrieval"
        assert r.domain == "reimburse"
        assert r.source == "llm"

    def test_placeholder_template_filled_into_system(self, monkeypatch):
        """v2 参数化模板：对话历史与当前输入在 system prompt 中被真实值替换。"""
        captured = {}

        def fake_llm(messages):
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return '{"retrieval_type": "new_question", "domain": "policy"}'

        monkeypatch.setattr(intent_mod, "default_llm_fn", lambda: fake_llm)
        _classify("餐饮发票全额抵扣吗", HISTORY, _cfg())
        system_text = captured["system"]
        assert "{chat_history}" not in system_text
        assert "{user_input}" not in system_text
        assert "餐饮发票全额抵扣吗" in system_text
        assert captured["user"]  # user 消息非空（触发分类）

    def test_garbage_degrades_to_default(self, monkeypatch):
        monkeypatch.setattr(intent_mod, "default_llm_fn", lambda: lambda m: "乱输出不是 JSON")
        r = _classify("餐饮发票全额抵扣吗", HISTORY, _cfg())
        assert r.retrieval_type == "new_question"
        assert r.source == "default"

    def test_invalid_enum_degrades_to_default(self, monkeypatch):
        monkeypatch.setattr(
            intent_mod,
            "default_llm_fn",
            lambda: lambda m: '{"retrieval_type": "unknown", "domain": "x"}',
        )
        r = _classify("餐饮发票全额抵扣吗", HISTORY, _cfg())
        assert r.retrieval_type == "new_question"
        assert r.source == "default"

    def test_llm_exception_degrades(self, monkeypatch):
        def boom(messages):
            raise RuntimeError("网络错误")

        monkeypatch.setattr(intent_mod, "default_llm_fn", lambda: boom)
        r = _classify("餐饮发票全额抵扣吗", HISTORY, _cfg())
        assert r.retrieval_type == "new_question"
        assert r.source == "default"

    def test_none_llm_fn_falls_to_rule(self, monkeypatch):
        monkeypatch.setattr(intent_mod, "default_llm_fn", lambda: None)
        r = _classify("餐饮发票全额抵扣吗", HISTORY, _cfg())
        assert r.retrieval_type == "new_question"
        assert r.source == "rule"

    def test_llm_fallback_disabled(self, monkeypatch):
        calls = {"n": 0}

        def llm(messages):
            calls["n"] += 1
            return "{}"

        monkeypatch.setattr(intent_mod, "default_llm_fn", lambda: llm)
        r = _classify("餐饮发票全额抵扣吗", HISTORY, _cfg(llm_fallback=False))
        assert calls["n"] == 0
        assert r.retrieval_type == "new_question"
        assert r.source == "rule"


class TestClassifyIntentNode:
    def _fake_cfg(self, **intent_kw) -> AppConfig:
        return AppConfig(
            company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
            risk=RiskConfig(),
            intent=IntentConfig(**intent_kw),
            paths=PathsConfig(db="/tmp/x.db", knowledge_dir="/tmp/k"),
        )

    def test_no_question_returns_empty(self, monkeypatch):
        monkeypatch.setattr(intent_mod, "get_app_config", lambda: self._fake_cfg())
        assert classify_intent({}) == {}
        assert classify_intent({"user_question": None}) == {}

    def test_disabled_outputs_default_intent(self, monkeypatch):
        monkeypatch.setattr(intent_mod, "get_app_config", lambda: self._fake_cfg(enabled=False))
        out = classify_intent({"user_question": "你好"})
        assert out["intent"]["retrieval_type"] == "new_question"
        assert out["intent"]["source"] == "default"

    def test_enabled_classifies(self, monkeypatch, no_llm):
        monkeypatch.setattr(intent_mod, "get_app_config", lambda: self._fake_cfg())
        out = classify_intent({"user_question": "你好", "chat_history": []})
        assert out["intent"]["retrieval_type"] == "chat"
        assert isinstance(out["intent"], dict)


class TestRouteEntryChatShortCircuit:
    def test_chat_routes_to_finalize(self):
        state = {"user_question": "你好", "intent": {"retrieval_type": "chat"}}
        assert route_entry(state) == "finalize"

    def test_no_retrieval_routes_to_finalize(self):
        state = {"user_question": "再说一遍", "intent": {"retrieval_type": "no_retrieval"}}
        assert route_entry(state) == "finalize"

    def test_file_takes_precedence_over_no_retrieval(self):
        state = {"file_path": "x.png", "intent": {"retrieval_type": "no_retrieval"}}
        assert route_entry(state) == "recognize"

    def test_follow_up_routes_to_retrieve(self):
        state = {"user_question": "那它呢", "intent": {"retrieval_type": "follow_up"}}
        assert route_entry(state) == "retrieve_docs"

    def test_no_intent_field_compat(self):
        """旧 checkpointer 线程无 intent 字段 → 视为新问题走检索。"""
        assert route_entry({"user_question": "问题"}) == "retrieve_docs"

    def test_file_takes_precedence_over_chat(self):
        state = {"file_path": "x.png", "intent": {"retrieval_type": "chat"}}
        assert route_entry(state) == "recognize"

    def test_no_question_no_intent(self):
        assert route_entry({}) == "finalize"


def test_intent_result_defaults():
    r = IntentResult()
    assert r.retrieval_type == "new_question"
    assert r.domain == "policy"
    assert r.source == "default"
