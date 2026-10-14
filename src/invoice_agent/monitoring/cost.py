"""LLM 成本计算：按 config.yaml 中的千 token 价目表计价，未配置的模型成本记 0。"""

from __future__ import annotations


def calc_cost(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing: dict[str, dict[str, float]],
) -> float:
    """返回本次调用的成本（元）。tokens 缺失或模型不在价目表 → 0.0。"""
    if prompt_tokens is None or completion_tokens is None:
        return 0.0
    price = pricing.get(model)
    if not price:
        return 0.0
    return prompt_tokens / 1000 * price.get("input", 0.0) + completion_tokens / 1000 * price.get(
        "output", 0.0
    )
