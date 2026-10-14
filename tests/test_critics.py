"""自反馈 critic 单测：假 llm_fn 注入，不联网；图级用例验证自评回环。"""

from pathlib import Path

import pytest

from invoice_agent.config import (
    AnswerFeedbackConfig,
    AppConfig,
    CompanyConfig,
    FeedbackConfig,
    PathsConfig,
    RiskConfig,
)
from invoice_agent.feedback.critics import (
    CritiqueResult,
    RiskReview,
    critique_answer,
    field_self_check,
    review_risk,
)
from invoice_agent.graph import builder as builder_mod
from invoice_agent.graph.state import AgentState
from invoice_agent.models.risk import Finding, RiskLevel
from test_e2e_graph import make_invoice


def _state(**kw) -> AgentState:
    return kw


# ---- ① critique_answer ----


class TestCritiqueAnswer:
    def _llm_returning(self, payload: str):
        return lambda messages: payload

    def test_passing_answer(self):
        payload = '{"sufficiency": 0.9, "relevance": 0.9, "completeness": 0.9, "score": 0.9, "issues": []}'
        r = critique_answer("回答", _state(), self._llm_returning(payload), pass_score=0.7)
        assert isinstance(r, CritiqueResult)
        assert r.passed and r.score == pytest.approx(0.9)

    def test_failing_answer_has_issues(self):
        payload = '{"sufficiency": 0.4, "relevance": 0.5, "completeness": 0.3, "score": 0.4, "issues": ["未提及高风险"]}'
        r = critique_answer("回答", _state(), self._llm_returning(payload), pass_score=0.7)
        assert not r.passed
        assert r.issues == ["未提及高风险"]

    def test_json_wrapped_in_markdown(self):
        payload = '好的，以下是评审结果：\n```json\n{"sufficiency": 1, "relevance": 1, "completeness": 1, "score": 1, "issues": []}\n```'
        r = critique_answer("回答", _state(), self._llm_returning(payload), pass_score=0.7)
        assert r.passed

    def test_high_finding_unmentioned_caps_completeness(self):
        findings = [Finding(rule_id="R001", rule_name="x", level=RiskLevel.HIGH, message="假发票")]
        payload = (
            '{"sufficiency": 1, "relevance": 1, "completeness": 0.9, "score": 1, "issues": []}'
        )
        r = critique_answer(
            "回答里没提风险",
            _state(risk_findings=findings, invoice=make_invoice()),
            self._llm_returning(payload),
            pass_score=0.7,
        )
        assert r.completeness <= 0.4

    def test_garbage_output_degrades_to_pass(self):
        r = critique_answer("回答", _state(), self._llm_returning("这不是 JSON"), pass_score=0.7)
        assert r.passed

    def test_llm_exception_degrades_to_pass(self):
        def boom(messages):
            raise RuntimeError("网络错误")

        r = critique_answer("回答", _state(), boom, pass_score=0.7)
        assert r.passed

    def test_none_llm_fn_degrades_to_pass(self):
        r = critique_answer("回答", _state(), None, pass_score=0.7)
        assert r.passed


# ---- ② field_self_check ----


class TestFieldSelfCheck:
    def test_issues_prefixed(self):
        payload = '{"issues": ["开票日期晚于当前日期"]}'
        r = field_self_check(make_invoice(), lambda m: payload)
        assert r == ["字段自检：开票日期晚于当前日期"]

    def test_clean(self):
        r = field_self_check(make_invoice(), lambda m: '{"issues": []}')
        assert r == []

    def test_garbage_degrades_to_empty(self):
        assert field_self_check(make_invoice(), lambda m: "乱输出") == []

    def test_exception_degrades_to_empty(self):
        def boom(messages):
            raise RuntimeError("超时")

        assert field_self_check(make_invoice(), boom) == []

    def test_none_degrades_to_empty(self):
        assert field_self_check(make_invoice(), None) == []


# ---- ③ review_risk ----


def _finding(rid: str = "R001") -> Finding:
    return Finding(rule_id=rid, rule_name="测试规则", level=RiskLevel.MEDIUM, message="命中")


class TestReviewRisk:
    def test_adjustment(self):
        payload = (
            '{"adjusted": [{"rule_id": "R002", "rule_name": "保留", "level": "低", "message": "ok"}],'
            ' "removed": ["R001"],'
            ' "added": [{"rule_id": "LLM-REVIEW", "rule_name": "模型复核", "level": "中", "message": "备注异常，需人工确认"}],'
            ' "summary": "一条误报已移除"}'
        )
        r = review_risk([_finding("R001"), _finding("R002")], make_invoice(), lambda m: payload)
        assert isinstance(r, RiskReview)
        assert r.removed == ["R001"]
        ids = [f.rule_id for f in r.adjusted]
        assert "R001" not in ids and "R002" in ids and "LLM-REVIEW" in ids
        assert all(f.evidence.get("review") == "llm" for f in r.added)

    def test_removed_unknown_id_ignored(self):
        payload = '{"adjusted": [], "removed": ["R999"], "added": [], "summary": ""}'
        r = review_risk([_finding("R001")], make_invoice(), lambda m: payload)
        assert [f.rule_id for f in r.adjusted] == ["R001"]

    def test_exception_returns_original(self):
        def boom(messages):
            raise RuntimeError("超时")

        original = [_finding("R001")]
        r = review_risk(original, make_invoice(), boom)
        assert r.adjusted == original

    def test_no_invoice_returns_original(self):
        original = [_finding("R001")]
        r = review_risk(original, None, lambda m: "{}")
        assert r.adjusted == original

    def test_parse_failure_retries_then_original(self):
        calls = {"n": 0}

        def llm(messages):
            calls["n"] += 1
            return "还是不是 JSON"

        original = [_finding("R001")]
        r = review_risk(original, make_invoice(), llm)
        assert calls["n"] == 2  # 解析失败重试 1 次
        assert r.adjusted == original

    def test_none_llm_fn(self):
        original = [_finding("R001")]
        assert review_risk(original, make_invoice(), None).adjusted == original


# ---- 图级：自评回环 ----


def _build_graph_with(monkeypatch, tmp_path: Path, finalize_fn, critique_fn):
    """构建图并替换外部依赖（仿 test_e2e_graph 模式）。"""
    import invoice_agent.graph.nodes.answer as answer_mod
    from invoice_agent.graph.nodes import critic, intake, recognize

    fake_cfg = AppConfig(
        company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
        risk=RiskConfig(),
        feedback=FeedbackConfig(answer=AnswerFeedbackConfig(max_retries=1, pass_score=0.7)),
        paths=PathsConfig(db=str(tmp_path / "g.db"), knowledge_dir=str(tmp_path / "k")),
    )
    from invoice_agent.graph.nodes import dedup, risk, verify

    for mod in (dedup, risk, verify):
        monkeypatch.setattr(mod, "get_app_config", lambda: fake_cfg)
    monkeypatch.setattr(critic, "get_app_config", lambda: fake_cfg)

    monkeypatch.setattr(intake, "intake", lambda state: {"page_images": ["x"]})
    monkeypatch.setattr(
        recognize,
        "recognize",
        lambda state: {"invoice": make_invoice(), "extract_error": None, "recognize_retry": 0},
    )
    monkeypatch.setattr(answer_mod, "retrieve_docs", lambda state: {"rag_chunks": []})
    monkeypatch.setattr(answer_mod, "finalize", finalize_fn)
    monkeypatch.setattr(critic, "critic_answer", critique_fn)

    builder_mod.build_graph.cache_clear()
    return builder_mod.build_graph()


class TestCritiqueLoop:
    def test_fail_once_then_pass(self, tmp_path, monkeypatch):
        """首次自评不通过 → finalize 带反馈重生成一次 → 二次自评通过 → END。"""

        finalize_calls = {"n": 0}

        def finalize(state):
            finalize_calls["n"] += 1
            fb = state.get("critic_feedback")
            return {"answer": f"answer-v{finalize_calls['n']} fb={fb}"}

        critique_calls = {"n": 0}

        def critic_answer(state):
            critique_calls["n"] += 1
            if critique_calls["n"] == 1:
                return {
                    "critique": {"score": 0.4, "passed": False, "issues": ["信息不足"]},
                    "answer_retry": 1,
                    "critic_feedback": "信息不足",
                }
            return {"critique": {"score": 0.9, "passed": True}, "critic_feedback": None}

        g = _build_graph_with(monkeypatch, tmp_path, finalize, critic_answer)
        state = g.invoke(
            {"session_id": "c1", "file_path": "x.png", "user_question": "能报销吗"},
            config={"configurable": {"thread_id": "c1"}},
        )
        assert finalize_calls["n"] == 2
        assert critique_calls["n"] == 2
        # 第二版回答基于第一版的反馈生成
        assert state["answer"] == "answer-v2 fb=信息不足"
        assert state["critique"]["passed"] is True

    def test_retry_exhausted_terminates(self, tmp_path, monkeypatch):
        """重试耗尽后放行（critique 标记 exhausted），不再回环。"""
        finalize_calls = {"n": 0}

        def finalize(state):
            finalize_calls["n"] += 1
            return {"answer": f"v{finalize_calls['n']}"}

        def critic_answer(state):
            retry = state.get("answer_retry", 0)
            if retry < 1:
                return {
                    "critique": {"score": 0.3, "passed": False, "issues": ["差"]},
                    "answer_retry": retry + 1,
                    "critic_feedback": "差",
                }
            # 耗尽放行：passed 置 True（与真实实现 critic.py 一致）
            return {
                "critique": {"score": 0.3, "passed": True, "exhausted": True},
                "critic_feedback": None,
            }

        g = _build_graph_with(monkeypatch, tmp_path, finalize, critic_answer)
        state = g.invoke(
            {"session_id": "c2", "file_path": "x.png"},
            config={"configurable": {"thread_id": "c2"}},
        )
        assert finalize_calls["n"] == 2  # 1 次原始 + 1 次重生成，然后终止
        assert state["critique"]["exhausted"] is True

    def test_risk_review_patches_findings(self, tmp_path, monkeypatch):
        """风控复核节点输出的修订 findings 对 finalize 可见。"""
        from invoice_agent.graph.nodes import critic

        monkeypatch.setattr(
            critic,
            "risk_review",
            lambda state: {
                "risk_findings": [
                    Finding(
                        rule_id="LLM-REVIEW",
                        rule_name="模型复核",
                        level=RiskLevel.LOW,
                        message="复核发现",
                    ),
                ],
                "risk_review": {"removed": [], "added": [], "summary": "ok"},
            },
        )
        seen = {}

        def finalize(state):
            seen["findings"] = state.get("risk_findings")
            return {"answer": "OK"}

        def critic_answer(state):
            return {"critique": {"passed": True}, "critic_feedback": None}

        g = _build_graph_with(monkeypatch, tmp_path, finalize, critic_answer)
        g.invoke(
            {"session_id": "c3", "file_path": "x.png"},
            config={"configurable": {"thread_id": "c3"}},
        )
        assert [f.rule_id for f in seen["findings"]] == ["LLM-REVIEW"]
