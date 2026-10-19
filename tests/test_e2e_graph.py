"""图编排 e2e 测试：monkeypatch 掉 LLM 调用，只验证 state 流转与重试边。"""

from datetime import date
from pathlib import Path

import pytest

from invoice_agent.graph import builder as builder_mod
from invoice_agent.graph.nodes import recognize as recognize_mod
from invoice_agent.models.invoice import InvoiceData, InvoiceTypeEnum


def make_invoice(**kw) -> InvoiceData:
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number="50000001",
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


@pytest.fixture(autouse=True)
def _skip_critics(monkeypatch):
    """自反馈/意图兜底节点会调 LLM，e2e 流转测试统一短路（各自逻辑见专测文件）。"""
    from invoice_agent.graph.nodes import critic, intent

    monkeypatch.setattr(
        critic,
        "critic_answer",
        lambda state: {"critique": {"passed": True}, "critic_feedback": None},
    )
    monkeypatch.setattr(critic, "risk_review", lambda state: {})
    monkeypatch.setattr(intent, "default_llm_fn", lambda: None)


def _patch_config(monkeypatch, tmp_path: Path) -> None:
    """把各节点引用的 get_app_config 指向临时 SQLite 路径（真实 AppConfig 结构）。"""
    from invoice_agent.config import AppConfig, CompanyConfig, PathsConfig, RiskConfig

    fake_cfg = AppConfig(
        company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
        risk=RiskConfig(),
        paths=PathsConfig(
            db=str(tmp_path / "g.db"),
            knowledge_dir=str(tmp_path / "knowledge"),
        ),
    )
    from invoice_agent.graph.nodes import dedup, intent, risk, verify

    for mod in (dedup, intent, risk, verify):
        monkeypatch.setattr(mod, "get_app_config", lambda: fake_cfg)


@pytest.fixture
def graph(tmp_path: Path, monkeypatch):
    """构建图并替换外部依赖：LLM、SQLite、检索。"""
    import invoice_agent.graph.nodes.answer as answer_mod
    from invoice_agent.graph import nodes

    # 1. intake 不真实读文件（省去生成图片），直接注入 page_images
    monkeypatch.setattr(
        nodes.intake, "intake", lambda state: {"page_images": ["data:image/png;base64,xx"]}
    )

    # 2. recognize 返回固定发票
    monkeypatch.setattr(
        nodes.recognize,
        "recognize",
        lambda state: {"invoice": make_invoice(), "extract_error": None, "recognize_retry": 0},
    )

    # 3. SQLite 指向临时目录
    _patch_config(monkeypatch, tmp_path)

    # 4. 检索返回空（不联网）
    monkeypatch.setattr(answer_mod, "retrieve_docs", lambda state: {"rag_chunks": []})

    # 5. finalize 不调 LLM，输出确定性文本
    monkeypatch.setattr(
        nodes.answer,
        "finalize",
        lambda state: {"answer": f"OK invoice={state['invoice'].invoice_number}"},
    )

    return _fresh_graph()


def _fresh_graph():
    """绕过 builder 的 lru_cache，构建全新图实例。"""
    builder_mod.build_graph.cache_clear()
    return builder_mod.build_graph()


class TestGraphFlow:
    def test_happy_path(self, graph):
        state = graph.invoke(
            {"session_id": "t1", "file_path": "x.png", "user_question": "能报销吗"},
            config={"configurable": {"thread_id": "t1"}},
        )
        assert state["invoice"].invoice_number == "50000001"
        assert state["verification"] is not None
        assert state["verification"].provider == "mock"
        assert state["answer"] == "OK invoice=50000001"
        assert isinstance(state["risk_findings"], list)

    def test_verify_dedup_parallel_join(self, graph):
        """verify 与 dedup 并行（join 边保证两者都完成后才进 risk）。

        注意：LangGraph 1.2 的 updates 流对同 superstep 的并行任务按任务分帧，
        不能断言「同一 payload 同时含两节点」；join 语义由 R002 回归测试行为级保障。
        """
        visited: list[str] = []
        for payload in graph.stream(
            {"session_id": "t-par", "file_path": "x.png", "user_question": "能报销吗"},
            config={"configurable": {"thread_id": "t-par"}},
            stream_mode="updates",
        ):
            visited.extend(payload.keys())
        assert visited.index("risk") > max(visited.index("verify"), visited.index("dedup"))

    def test_r002_survives_parallel(self, graph, tmp_path):
        """并发改造后 risk 仍能看到 dedup 的 duplicate_hits：R002 不被静默吞掉。"""
        from invoice_agent.services.dedup_store import DedupStore

        # seed 一张已提交的同号假票（sha 不同 → 非同一文件重复上传，走 R002 重复报销）
        DedupStore(tmp_path / "g.db").save_reimbursement(
            "50000001", "044032000111", "91330100MA28YYYYYY",
            "91330100MA27XXXXXX", 113.0, date(2026, 10, 1), file_sha256="seeded",
        )
        state = graph.invoke(
            {"session_id": "t-r002", "file_path": "x.png", "user_question": "能报销吗"},
            config={"configurable": {"thread_id": "t-r002"}},
        )
        rule_ids = {f.rule_id for f in state["risk_findings"]}
        assert "R002" in rule_ids

    def test_retry_then_success(self, tmp_path, monkeypatch):
        """提取失败 1 次后重试成功，应走重试边。"""
        from invoice_agent.graph import nodes

        monkeypatch.setattr(nodes.intake, "intake", lambda state: {"page_images": ["x"]})
        calls = {"n": 0}

        def flaky_recognize(state):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"invoice": None, "extract_error": "发票号码位数异常", "recognize_retry": 1}
            return {"invoice": make_invoice(), "extract_error": None, "recognize_retry": 0}

        monkeypatch.setattr(nodes.recognize, "recognize", flaky_recognize)
        _patch_config(monkeypatch, tmp_path)
        import invoice_agent.graph.nodes.answer as answer_mod

        monkeypatch.setattr(answer_mod, "retrieve_docs", lambda state: {"rag_chunks": []})
        monkeypatch.setattr(
            nodes.answer,
            "finalize",
            lambda state: {"answer": f"retries={state.get('recognize_retry', 0)}"},
        )

        g = _fresh_graph()
        state = g.invoke(
            {"session_id": "t2", "file_path": "x.png"},
            config={"configurable": {"thread_id": "t2"}},
        )
        assert calls["n"] == 2
        assert state["answer"] == "retries=0"

    def test_retry_exhausted_degrades(self, tmp_path, monkeypatch):
        """重试耗尽 → finalize 降级（invoice 为 None 时输出人工复核提示）。"""
        from invoice_agent.graph import nodes

        monkeypatch.setattr(nodes.intake, "intake", lambda state: {"page_images": ["x"]})

        def always_fail(state):
            return {"invoice": None, "extract_error": "无法识别", "recognize_retry": 99}

        monkeypatch.setattr(nodes.recognize, "recognize", always_fail)
        _patch_config(monkeypatch, tmp_path)
        # 降级路径 finalize 不调 LLM，无需 patch
        g = _fresh_graph()
        state = g.invoke(
            {"session_id": "t3", "file_path": "x.png"},
            config={"configurable": {"thread_id": "t3"}},
        )
        assert "人工复核" in state["answer"]
        # 走了降级边，不应有验真结果
        assert state.get("verification") is None

    def test_pure_rag_branch(self, tmp_path, monkeypatch):
        """无文件仅提问 → retrieve_docs 分支。"""
        import invoice_agent.graph.nodes.answer as answer_mod
        from invoice_agent.graph import nodes

        monkeypatch.setattr(answer_mod, "retrieve_docs", lambda state: {"rag_chunks": []})
        monkeypatch.setattr(
            nodes.answer,
            "finalize",
            lambda state: {"answer": "RAG-ONLY"},
        )
        g = _fresh_graph()
        state = g.invoke(
            {"session_id": "t4", "file_path": None, "user_question": "餐饮票能抵扣吗"},
            config={"configurable": {"thread_id": "t4"}},
        )
        assert state["answer"] == "RAG-ONLY"
        assert state.get("invoice") is None

    def test_chat_short_circuits_retrieval(self, tmp_path, monkeypatch):
        """无文件闲聊 → 意图判 chat → 跳过检索直接 finalize。"""
        import invoice_agent.graph.nodes.answer as answer_mod
        from invoice_agent.graph import nodes

        retrieve_calls = {"n": 0}

        def retrieve_docs(state):
            retrieve_calls["n"] += 1
            return {"rag_chunks": []}

        monkeypatch.setattr(answer_mod, "retrieve_docs", retrieve_docs)
        monkeypatch.setattr(nodes.answer, "finalize", lambda state: {"answer": "你好，我是报销助手"})

        g = _fresh_graph()
        state = g.invoke(
            {"session_id": "t5", "file_path": None, "user_question": "你好"},
            config={"configurable": {"thread_id": "t5"}},
        )
        assert retrieve_calls["n"] == 0  # 闲聊短路，检索节点未执行
        assert state["answer"] == "你好，我是报销助手"
        assert state["intent"]["retrieval_type"] == "chat"


class TestFinalizePromptParams:
    def test_qa_placeholder_filled_into_system(self, monkeypatch):
        """qa v3 参数化模板：用户问题与参考条款填入 system prompt，human 消息不再重复。"""
        from langchain_core.messages import AIMessage

        import invoice_agent.graph.nodes.answer as answer_mod

        captured = {}

        class FakeModel:
            def invoke(self, messages):
                captured["messages"] = messages
                return AIMessage(content="测试回答")

        monkeypatch.setattr(answer_mod, "get_chat_model", lambda: FakeModel())
        state = {
            "session_id": "t-params",
            "user_question": "餐饮发票能抵扣吗",
            "rag_chunks": [],
            "chat_history": [],
        }
        answer_mod.finalize(state)
        messages = captured["messages"]
        system_text = messages[0].content
        human_text = messages[-1].content
        assert "{user_query}" not in system_text
        assert "{reference_clauses}" not in system_text
        assert "餐饮发票能抵扣吗" in system_text
        assert "（未检索到相关条款）" in system_text
        assert "## 用户问题" not in human_text  # 问题已入 system，human 不重复
        assert "## 参考条款" not in human_text

    def test_query_rewrite_placeholder_filled_into_system(self, monkeypatch):
        """query_rewrite v3 参数化模板：对话/领域/查询填入 system prompt。"""
        from langchain_core.messages import AIMessage

        import invoice_agent.graph.nodes.answer as answer_mod

        captured = {}

        class FakeModel:
            def invoke(self, messages):
                captured["messages"] = messages
                return AIMessage(content="出租车发票 进项税额 抵扣 政策")

        monkeypatch.setattr(answer_mod, "get_chat_model", lambda: FakeModel())
        state = {
            "chat_history": [
                {"role": "user", "content": "增值税专用发票的认证期限是多久"},
                {"role": "assistant", "content": "……"},
            ],
            "intent": {"domain": "deduction"},
        }
        q = answer_mod._rewrite_query("那出租车票呢", state)
        assert q == "出租车发票 进项税额 抵扣 政策"
        system_text = captured["messages"][0].content
        assert "{chat_history}" not in system_text
        assert "{domain}" not in system_text
        assert "{query}" not in system_text
        assert "那出租车票呢" in system_text
        assert "增值税专用发票的认证期限是多久" in system_text

    def test_recognize_placeholder_filled_into_system(self, monkeypatch):
        """recognize v2 参数化模板：当前日期与重试反馈填入 system prompt。"""
        import invoice_agent.graph.nodes.recognize as recognize_mod

        captured = {}

        class FakeStructured:
            def invoke(self, messages):
                captured["messages"] = messages
                return make_invoice()

        class FakeVision:
            def with_structured_output(self, schema):
                return FakeStructured()

        monkeypatch.setattr(recognize_mod, "get_vision_model", lambda: FakeVision())
        monkeypatch.setattr(recognize_mod, "default_llm_fn", lambda: None)  # 跳过 LLM 字段自检
        state = {
            "session_id": "t-recog",
            "page_images": ["data:image/png;base64,xx"],
            "extract_error": "发票号码位数异常",
        }
        result = recognize_mod.recognize(state)
        assert result["invoice"] is not None
        system_text = captured["messages"][0].content
        assert "{current_date}" not in system_text
        assert "{retry_feedback}" not in system_text
        assert "发票号码位数异常" in system_text  # 重试反馈进 system


class TestCheckExtract:
    def test_valid(self):
        assert recognize_mod.check_extract(make_invoice()) == []

    def test_bad_number_length(self):
        errors = recognize_mod.check_extract(make_invoice(invoice_number="12345"))
        assert any("位数" in e for e in errors)

    def test_bad_amount(self):
        errors = recognize_mod.check_extract(make_invoice(total_amount=-5))
        assert any("价税合计" in e for e in errors)

    def test_missing_buyer(self):
        errors = recognize_mod.check_extract(make_invoice(buyer_name=""))
        assert any("购买方" in e for e in errors)
