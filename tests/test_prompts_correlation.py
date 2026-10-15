"""prompt 与监控调用链的关联测试：contextvar → handler → llm_calls 三列落库。"""

import uuid
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from invoice_agent.config import MonitoringConfig
from invoice_agent.monitoring.handler import TraceCallbackHandler
from invoice_agent.monitoring.store import MonitorStore
from invoice_agent.prompts.registry import current_prompt, use


def _chat_result() -> ChatResult:
    msg = AIMessage(
        content="ok",
        response_metadata={
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        },
    )
    return ChatResult(
        generations=[ChatGeneration(message=msg)],
        llm_output={"model_name": "qwen-max"},
    )


def _write_prompt(dir_path: Path, name: str, text: str) -> None:
    payload = {"name": name, "description": "", "active": "v1",
               "versions": [{"version": "v1", "text": text}]}
    (dir_path / f"{name}.yaml").write_text(
        yaml.dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    _write_prompt(d, "qa", "QA_PROMPT_TEXT")
    _write_prompt(d, "critique", "CRITIQUE_PROMPT_TEXT")
    return d


@pytest.fixture
def registry(prompts_dir: Path, monkeypatch):
    """用 tmp prompts 目录替换全局单例。"""
    import invoice_agent.prompts.registry as registry_mod

    monkeypatch.setattr(
        registry_mod,
        "get_registry",
        lambda: registry_mod.PromptRegistry(prompts_dir, snapshot=False),
    )
    return registry_mod.get_registry()


def _handler(tmp_path: Path) -> TraceCallbackHandler:
    return TraceCallbackHandler(MonitorStore(tmp_path / "m.db"), MonitoringConfig(), "t1", "ask")


def _last_llm_row(store: MonitorStore) -> dict:
    with store._conn() as conn:
        return dict(conn.execute("SELECT * FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone())


def _simulate_llm(handler: TraceCallbackHandler):
    root, node, llm = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    handler.on_chain_start(None, {}, run_id=root, parent_run_id=None, name="LangGraph")
    handler.on_chain_start(None, {}, run_id=node, parent_run_id=root, name="finalize")
    handler.on_chat_model_start(None, [], run_id=llm, parent_run_id=node)
    handler.on_chat_model_end(_chat_result(), run_id=llm, parent_run_id=node)
    handler.on_chain_end({}, run_id=node, parent_run_id=root, name="finalize")
    handler.on_chain_end({}, run_id=root, parent_run_id=None, name="LangGraph")


class TestCorrelation:
    def test_use_records_prompt_columns(self, tmp_path: Path, registry):
        handler = _handler(tmp_path)
        with use("qa") as meta:
            _simulate_llm(handler)
        row = _last_llm_row(handler.store)
        assert row["prompt_name"] == meta.name == "qa"
        assert row["prompt_version"] == meta.version
        assert row["prompt_hash"] == meta.hash

    def test_no_context_writes_null(self, tmp_path: Path, registry):
        assert current_prompt() is None
        handler = _handler(tmp_path)
        _simulate_llm(handler)
        row = _last_llm_row(handler.store)
        assert row["prompt_name"] is None
        assert row["prompt_version"] is None
        assert row["prompt_hash"] is None

    def test_on_llm_error_records_prompt(self, tmp_path: Path, registry):
        handler = _handler(tmp_path)
        root, node, llm = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        handler.on_chain_start(None, {}, run_id=root, parent_run_id=None, name="LangGraph")
        handler.on_chain_start(None, {}, run_id=node, parent_run_id=root, name="finalize")
        handler.on_chat_model_start(None, [], run_id=llm, parent_run_id=node)
        with use("qa") as meta:
            handler.on_llm_error(RuntimeError("x"), run_id=llm, parent_run_id=node)
        row = _last_llm_row(handler.store)
        assert row["success"] == 0
        assert row["prompt_name"] == meta.name
        assert row["prompt_hash"] == meta.hash

    def test_write_llm_call_old_signature_still_works(self, tmp_path: Path):
        """不传新参数的旧式调用兼容（三列为 NULL）。"""
        store = MonitorStore(tmp_path / "m.db")
        store.write_llm_call("r1", "recognize", "m", 1, 2, 3, 4.0, 0.0, True)
        row = _last_llm_row(store)
        assert row["prompt_name"] is None

    def test_mock_llm_fn_path_covered(self, tmp_path: Path, registry):
        """llm_fn 注入路径：use() 块内调用 mock，contextvar 对 mock 可见。"""
        seen = {}

        def fake_llm_fn(messages):
            meta = current_prompt()
            seen["meta"] = meta
            return messages[0]["content"]

        from invoice_agent.feedback.critics import critique_answer

        result = critique_answer("回答", {"user_question": "问题"}, fake_llm_fn, 0.7)
        assert seen["meta"] is not None
        assert seen["meta"].name == "critique"
        assert result.passed  # mock 原样返回 system 文本，无法解析 JSON → 放行


class TestSnapshotQueries:
    def test_prompt_usage_and_versions(self, tmp_path: Path, registry):
        store = MonitorStore(tmp_path / "m.db")
        store.write_llm_call(
            "r1", "finalize", "m", 1, 1, 2, 10.0, 0.0, True,
            prompt_name="qa", prompt_version="v1", prompt_hash="abc",
        )
        store.write_llm_call(
            "r2", "finalize", "m", 1, 1, 2, 20.0, 0.0, False,
            prompt_name="qa", prompt_version="v1", prompt_hash="abc",
        )
        usage = store.prompt_usage()
        assert len(usage) == 1
        assert usage[0]["calls"] == 2
        assert usage[0]["success_rate"] == 0.5
        assert usage[0]["avg_latency_ms"] == 15.0
