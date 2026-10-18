"""checkpointer 工厂分发单测：memory 分支与参数校验，不连 PG、不触网。"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from invoice_agent.config import (
    AppConfig,
    CheckpointerConfig,
    CompanyConfig,
    PathsConfig,
    RiskConfig,
)
from invoice_agent.graph.checkpointer import get_checkpointer


def _cfg(backend: str) -> AppConfig:
    return AppConfig(
        company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
        risk=RiskConfig(),
        paths=PathsConfig(db="/tmp/x.db", knowledge_dir="/tmp/k"),
        checkpointer=CheckpointerConfig(backend=backend),
    )


def test_memory_backend_returns_memory_saver():
    """默认 memory 分支：进程内 saver，零 PG。"""
    assert isinstance(get_checkpointer(_cfg("memory")), MemorySaver)


def test_unknown_backend_fails_fast():
    """运行时守卫兜底（pydantic Literal 已在配置层拦截，此处绕过校验直测分发逻辑）。"""
    cfg = _cfg("memory")
    cfg.checkpointer = CheckpointerConfig.model_construct(backend="bogus")
    with pytest.raises(ValueError, match="未知 checkpointer.backend"):
        get_checkpointer(cfg)
