"""监控模块单测：SQLite 读写、成本计算、回调采集、阈值告警、日志幂等。"""

import logging
import typing
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from invoice_agent.config import AlertConfig, MonitoringConfig
from invoice_agent.monitoring.cost import calc_cost
from invoice_agent.monitoring.handler import TraceCallbackHandler
from invoice_agent.monitoring.setup import setup_logging
from invoice_agent.monitoring.store import MonitorStore

# ---- calc_cost ----


class TestCalcCost:
    PRICING: typing.ClassVar[dict] = {"qwen-max": {"input": 0.0024, "output": 0.0096}}

    def test_hit(self):
        # 1000 in / 1000 out → 0.0024 + 0.0096
        assert calc_cost("qwen-max", 1000, 1000, self.PRICING) == pytest.approx(0.012)

    def test_model_miss_is_zero(self):
        assert calc_cost("unknown", 1000, 1000, self.PRICING) == 0.0

    def test_missing_tokens_is_zero(self):
        assert calc_cost("qwen-max", None, 100, self.PRICING) == 0.0


# ---- MonitorStore ----


class TestMonitorStore:
    def test_write_and_readback(self, tmp_path: Path):
        store = MonitorStore(tmp_path / "m.db")
        store.start_run("r1", "t1", "invoice")
        store.write_span("r1", "recognize", 1, "2026-01-01T00:00:00", 123.0, "success", 0)
        store.write_span("r1", "recognize", 2, "2026-01-01T00:00:01", 50.0, "success", 1)
        store.write_llm_call("r1", "recognize", "qwen-vl-max", 100, 50, 150, 900.0, 0.001, True)
        store.end_run("r1", "success", 5000.0, 100, 50, 0.001)
        store.write_alert("r1", "slow_node", "WARN", "慢")

        runs = store.recent_runs(10)
        assert len(runs) == 1
        assert runs[0]["status"] == "success"
        assert runs[0]["prompt_tokens"] == 100
        assert runs[0]["node_count"] == 2
        assert runs[0]["retry_count"] == 1  # recognize 第二次执行 retry_of=1

        detail = store.run_detail("r1")
        assert [s["node"] for s in detail["spans"]] == ["recognize", "recognize"]
        # llm_calls 明细随 run_detail 一起返回（含 prompt 追溯三列）
        assert len(detail["llm_calls"]) == 1
        call = detail["llm_calls"][0]
        assert call["model"] == "qwen-vl-max"
        assert call["total_tokens"] == 150

        stats = store.stats(20)
        assert stats["run_count"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["completion_tokens"] == 50

        alerts = store.recent_alerts(10)
        assert alerts[0]["alert_type"] == "slow_node"

    def test_run_detail_includes_prompt_trace(self, tmp_path: Path):
        store = MonitorStore(tmp_path / "m.db")
        store.start_run("r1", "t1", "ask")
        store.write_llm_call(
            "r1",
            "finalize",
            "qwen3.8-max",
            100,
            50,
            150,
            900.0,
            0.001,
            True,
            prompt_name="qa",
            prompt_version="v2",
            prompt_hash="abc123",
        )
        call = store.run_detail("r1")["llm_calls"][0]
        assert call["prompt_name"] == "qa"
        assert call["prompt_version"] == "v2"
        assert call["prompt_hash"] == "abc123"

    def test_write_failure_does_not_raise(self, tmp_path: Path):
        store = MonitorStore(tmp_path / "m.db")
        # 破坏 schema 后写库应静默失败（吞异常不反噬主流程）
        import sqlite3

        conn = sqlite3.connect(store.db_path)
        conn.execute("DROP TABLE runs")
        conn.commit()
        conn.close()
        store.start_run("r1", "t1", "invoice")  # 不应抛异常
        store.write_alert("r1", "failure", "ERROR", "x")

    def test_readback_after_reconnect(self, tmp_path: Path):
        """短连接模式：新实例（模拟新进程）能读回数据。"""
        MonitorStore(tmp_path / "m.db").start_run("r1", "t1", "ask")
        runs = MonitorStore(tmp_path / "m.db").recent_runs(5)
        assert runs[0]["run_id"] == "r1"


# ---- TraceCallbackHandler ----


def _chat_result(model: str = "qwen-max", prompt: int = 100, completion: int = 50) -> ChatResult:
    msg = AIMessage(
        content="ok",
        response_metadata={
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        },
    )
    return ChatResult(generations=[ChatGeneration(message=msg)], llm_output={"model_name": model})


def _simulate_run(
    handler: TraceCallbackHandler,
    fail_node: bool = False,
    serialized=None,
    chat_result: ChatResult | None = None,
):
    """直接调回调方法模拟一次 run：root → recognize(含 LLM) → root end。"""
    root = uuid.uuid4()
    node = uuid.uuid4()
    llm = uuid.uuid4()
    handler.on_chain_start(None, {}, run_id=root, parent_run_id=None, name="LangGraph")
    handler.on_chain_start(None, {}, run_id=node, parent_run_id=root, name="recognize")
    handler.on_chat_model_start(serialized, [], run_id=llm, parent_run_id=node)
    if fail_node:
        handler.on_chain_error(RuntimeError("节点炸了"), run_id=node, parent_run_id=root)
    else:
        handler.on_chat_model_end(chat_result or _chat_result(), run_id=llm, parent_run_id=node)
        handler.on_chain_end({}, run_id=node, parent_run_id=root, name="recognize")
    handler.on_chain_end({}, run_id=root, parent_run_id=None, name="LangGraph")


class TestTraceCallbackHandler:
    def _handler(self, tmp_path: Path, **alert_kw) -> TraceCallbackHandler:
        cfg = MonitoringConfig(
            pricing={"qwen-max": {"input": 0.0024, "output": 0.0096}},
            alert=AlertConfig(**alert_kw),
        )
        return TraceCallbackHandler(MonitorStore(tmp_path / "m.db"), cfg, "t1", "invoice")

    def test_model_name_from_serialized_kwargs(self, tmp_path: Path):
        """ChatOpenAI 序列化后模型名在 kwargs.model_name（非 model），取错键会导致成本记 0。

        用 llm_output 为空的 ChatResult 模拟流式响应（llm_output 无 model_name 时
        以开始回调从 serialized 提取的为准）。
        """
        handler = self._handler(tmp_path)
        _simulate_run(
            handler,
            serialized={"kwargs": {"model_name": "qwen3.8-max"}},
            chat_result=ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))], llm_output={}
            ),
        )
        with handler.store._conn() as conn:
            row = dict(conn.execute("SELECT * FROM llm_calls").fetchone())
        assert row["model"] == "qwen3.8-max"

    def test_full_run_persists(self, tmp_path: Path):
        handler = self._handler(tmp_path)
        _simulate_run(handler)
        store = handler.store

        runs = store.recent_runs(5)
        assert runs[0]["status"] == "success"
        assert runs[0]["prompt_tokens"] == 100
        assert runs[0]["completion_tokens"] == 50
        assert runs[0]["total_latency_ms"] >= 0

        spans = store.run_detail(handler.run_id)["spans"]
        assert len(spans) == 1
        assert spans[0]["node"] == "recognize"
        assert spans[0]["status"] == "success"
        assert spans[0]["latency_ms"] >= 0

        # LLM 记录：tokens + 成本（1000/1000 价目下 100+50 → 0.00024+0.00048）
        with store._conn() as conn:
            row = dict(conn.execute("SELECT * FROM llm_calls").fetchone())
        assert row["model"] == "qwen-max"
        assert row["prompt_tokens"] == 100
        assert row["cost_yuan"] == pytest.approx(0.00072)

        # 事件缓冲供 UI 实时读取
        types = [e["type"] for e in handler.events]
        assert "run_start" in types and "node_start" in types
        assert "node_end" in types and "llm_end" in types and "run_end" in types

    def test_node_failure_marks_run_failed_with_alert(self, tmp_path: Path):
        handler = self._handler(tmp_path)
        _simulate_run(handler, fail_node=True)
        runs = handler.store.recent_runs(5)
        assert runs[0]["status"] == "failure"
        alert_types = [a["alert_type"] for a in handler.store.recent_alerts(10)]
        assert "failure" in alert_types

    def test_slow_node_alert(self, tmp_path: Path):
        # 预算 0 → 任何耗时都触发 slow_node
        handler = self._handler(tmp_path, node_latency_budget_ms={"default": 0})
        _simulate_run(handler)
        alert_types = [a["alert_type"] for a in handler.store.recent_alerts(10)]
        assert "slow_node" in alert_types

    def test_retry_counted(self, tmp_path: Path):
        handler = self._handler(tmp_path)
        root = uuid.uuid4()
        handler.on_chain_start(None, {}, run_id=root, parent_run_id=None, name="LangGraph")
        for i in range(3):  # recognize 执行 3 次（retry_of=0,1,2）
            node = uuid.uuid4()
            handler.on_chain_start(None, {}, run_id=node, parent_run_id=root, name="recognize")
            handler.on_chain_end({}, run_id=node, parent_run_id=root, name="recognize")
        handler.on_chain_end({}, run_id=root, parent_run_id=None, name="LangGraph")
        with handler.store._conn() as conn:
            retries = [
                r["retry_of"]
                for r in conn.execute("SELECT retry_of FROM node_spans ORDER BY id").fetchall()
            ]
        assert retries == [0, 1, 2]
        # max_retries_per_run=2 → 第 3 次执行（retry_of=2）不超阈值，无 high_retry 告警
        # 超阈值场景：max=0 时任何重复执行都告警
        handler2 = self._handler(tmp_path / "2", max_retries_per_run=0)
        root = uuid.uuid4()
        handler2.on_chain_start(None, {}, run_id=root, parent_run_id=None, name="LangGraph")
        for _ in range(2):
            node = uuid.uuid4()
            handler2.on_chain_start(None, {}, run_id=node, parent_run_id=root, name="recognize")
            handler2.on_chain_end({}, run_id=node, parent_run_id=root, name="recognize")
        handler2.on_chain_end({}, run_id=root, parent_run_id=None, name="LangGraph")
        alert_types = [a["alert_type"] for a in handler2.store.recent_alerts(10)]
        assert "high_retry" in alert_types

    def test_callback_exception_never_raises(self, tmp_path: Path):
        handler = self._handler(tmp_path)
        # 乱序/缺失上下文的回调不应抛异常
        handler.on_chain_end({}, run_id=uuid.uuid4(), parent_run_id=uuid.uuid4(), name="verify")
        handler.on_llm_error(RuntimeError("x"), run_id=uuid.uuid4())
        handler.on_chat_model_end(None, run_id=uuid.uuid4())  # response 为 None


# ---- setup_logging ----


class TestSetupLogging:
    def test_idempotent(self, tmp_path: Path, monkeypatch):
        from invoice_agent.config import AppConfig, CompanyConfig, PathsConfig, RiskConfig

        fake_cfg = AppConfig(
            company=CompanyConfig(name="x", tax_id="y"),
            risk=RiskConfig(),
            monitoring=MonitoringConfig(log_file=str(tmp_path / "logs" / "agent.log")),
            paths=PathsConfig(db=str(tmp_path / "g.db"), knowledge_dir=str(tmp_path / "k")),
        )
        import invoice_agent.monitoring.setup as setup_mod

        monkeypatch.setattr(setup_mod, "get_app_config", lambda: fake_cfg)

        root = logging.getLogger()
        saved = root.handlers[:]
        saved_level = root.level
        try:
            setup_logging()
            setup_logging()
            marked = [h for h in root.handlers if getattr(h, "_invoice_agent_handler", False)]
            assert len(marked) == 2  # console + file，幂等不翻倍
            assert (tmp_path / "logs" / "agent.log").exists()
        finally:
            for h in root.handlers[:]:
                if getattr(h, "_invoice_agent_handler", False) or h not in saved:
                    root.removeHandler(h)
            root.handlers[:] = saved
            root.setLevel(saved_level)
