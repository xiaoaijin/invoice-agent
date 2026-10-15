"""全局 fixture：隔离 prompt registry 单例与 contextvar，防跨测试污染。"""

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
