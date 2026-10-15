"""配置一致性测试：防止模型名在 .env.example / config.py / config.yaml 间漂移。

背景：config.py 默认值曾用旧模型名（qwen-vl-max 等），而 config.yaml 的
monitoring.pricing 只配了新模型名——不显式设 .env 时 LLM 成本全部记 0。
此测试断言：Settings 三个默认模型名必须都在价目表里。
"""

from __future__ import annotations

from invoice_agent.config import PROJECT_ROOT, Settings, load_app_config


def test_default_models_covered_by_pricing():
    pricing = load_app_config().monitoring.pricing
    defaults = {
        name: Settings.model_fields[name].default
        for name in ("model_vision", "model_chat", "embedding_model")
    }
    for field, model in defaults.items():
        assert model in pricing, (
            f"{field} 默认模型 {model} 不在 config.yaml monitoring.pricing 中，"
            "该模型的 token 成本会被记为 0"
        )


def test_default_models_match_env_example():
    """config.py 的默认模型名应与 .env.example 保持一致。"""
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for field in ("model_vision", "model_chat", "embedding_model"):
        default = Settings.model_fields[field].default
        env_key = field.upper()
        assert f"{env_key}={default}" in env_example, (
            f"{env_key} 在 config.py 的默认值 {default} 与 .env.example 不一致"
        )
