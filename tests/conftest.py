"""全局 fixture：隔离 prompt registry 单例与 contextvar，防跨测试污染。

另：强制单元测试 checkpointer 走 memory 后端（config.yaml 的 postgres 只对真实部署生效）。
"""

import pytest

from invoice_agent.prompts import registry as registry_mod


@pytest.fixture(autouse=True)
def _reset_prompt_state():
    registry_mod.get_registry.cache_clear()
    token = registry_mod._CURRENT_PROMPT.set(None)
    try:
        yield
    finally:
        registry_mod._CURRENT_PROMPT.reset(token)
        registry_mod.get_registry.cache_clear()


@pytest.fixture(autouse=True)
def _memory_checkpointer(request):
    """单元测试不连 PG：强制 checkpointer backend=memory（integration 标记的测试放行）。

    patch load_app_config（而非替换 get_app_config 属性）——test_api.py 的 client
    fixture 会对 get_app_config.cache_clear() 并以其 fake_cfg 覆盖，两套 fixture 自然叠加。
    手动保存/恢复而非 monkeypatch fixture：本 fixture 若请求 monkeypatch 会改变其
    撤销时机，殃及 _reset_prompt_state 的 teardown 顺序。
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    from invoice_agent import config as config_mod
    from invoice_agent.config import CheckpointerConfig

    real_load = config_mod.load_app_config

    def load_memory(path=None):
        return real_load(path).model_copy(
            update={"checkpointer": CheckpointerConfig(backend="memory")}
        )

    config_mod.load_app_config = load_memory
    config_mod.get_app_config.cache_clear()
    try:
        yield
    finally:
        config_mod.load_app_config = real_load
        config_mod.get_app_config.cache_clear()
