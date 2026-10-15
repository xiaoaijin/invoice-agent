"""意图识别节点：关键词 → 正则 → LLM 兜底的漏斗分类。

输出两层意图：检索类型（新问题/追问/闲聊）+ 业务领域标签。
前几层为纯字符串匹配（零成本同步），仅都未定性时才调 LLM；
LLM 失败/关闭均静默降级为默认意图，不阻塞主流程。
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from invoice_agent.config import IntentConfig, get_app_config
from invoice_agent.feedback.critics import LlmFn, _extract_json, default_llm_fn
from invoice_agent.graph.state import AgentState
from invoice_agent.prompts.registry import use

logger = logging.getLogger(__name__)

# 追问信号正则：指代上文 / 依赖对话历史的问法（answer._is_follow_up 前缀表的超集）
_FOLLOW_UP_PATTERNS = (
    re.compile(r"^(那|它|这个|这批|还有|为什么|怎么|如上|上述|刚才|继续|上面|按你)"),
    re.compile(r"呢[？?！!]?$"),  # "那出租车票呢"
    re.compile(r"(刚才|上面|之前)(说|提|讲)"),
    re.compile(r"^(再|重新|补充)(说|讲|答|问)"),
)

_VALID_RETRIEVAL_TYPES = {"new_question", "follow_up", "chat"}


class IntentResult(BaseModel):
    """意图识别结果（漏斗产出）。"""

    retrieval_type: Literal["new_question", "follow_up", "chat"] = "new_question"
    domain: str = "policy"  # policy/header/deduction/reimburse/verify/chat
    source: str = "default"  # 溯源：keyword | regex | rule | llm | default


def classify_intent(state: AgentState) -> dict:
    """意图识别节点：intake 后、路由前执行。无提问时零成本跳过。"""
    question = state.get("user_question")
    if not question:
        return {}
    cfg = get_app_config().intent
    if not cfg.enabled:
        return {"intent": IntentResult(source="default").model_dump()}
    result = _classify(question, state.get("chat_history") or [], cfg)
    return {"intent": result.model_dump()}


def _classify(question: str, history: list[dict], cfg: IntentConfig) -> IntentResult:
    """漏斗分类：关键词 → 正则 → 规则短路 → LLM 兜底（前几层命中即返回，零 LLM）。"""
    q = question.strip()
    # 第 1 层：关键词（子串匹配）——闲聊定性 + 领域打标（无论检索类型如何都打）
    domain = _match_domain(q, cfg.domain_keywords)
    if any(kw in q for kw in cfg.chat_keywords):
        return IntentResult(retrieval_type="chat", domain=domain or "chat", source="keyword")
    # 第 2 层：追问正则（需有上文——首轮无指代对象，追问判定无消解价值）
    if history and any(p.search(q) for p in _FOLLOW_UP_PATTERNS):
        return IntentResult(retrieval_type="follow_up", domain=domain, source="regex")
    # 第 3 层：规则短路（避免无谓 LLM）——无历史不可能是追问；长问题 + 领域已命中视为独立新问题
    if not history or (len(q) >= 20 and domain):
        return IntentResult(retrieval_type="new_question", domain=domain, source="rule")
    # 第 4 层：LLM 兜底（仅前面都未定性时才构造 LLM；关闭/不可用 → 规则默认）
    llm_fn: LlmFn = default_llm_fn() if cfg.llm_fallback else None
    if llm_fn is None:
        return IntentResult(retrieval_type="new_question", domain=domain, source="rule")
    try:
        with use("intent") as prompt:
            recent = "\n".join(
                f"{'用户' if e.get('role') == 'user' else '助手'}：{e.get('content', '')}"
                for e in history[-4:]  # 最近 2 轮
            )
            user_content = f"## 最近对话\n{recent or '（无）'}\n\n## 当前问题\n{q}"
            raw = llm_fn(
                [
                    {"role": "system", "content": prompt.text},
                    {"role": "user", "content": user_content},
                ]
            )
    except Exception as exc:  # noqa: BLE001 — 兜底是增强项，失败降级默认
        logger.warning("意图识别 LLM 兜底失败（降级默认）：%s", exc)
        return IntentResult(retrieval_type="new_question", domain=domain, source="default")
    data = _extract_json(raw)
    rtype = str((data or {}).get("retrieval_type", ""))
    if data is None or rtype not in _VALID_RETRIEVAL_TYPES:
        logger.warning("意图识别输出无法解析/非法（降级默认）：%s", str(raw)[:200])
        return IntentResult(retrieval_type="new_question", domain=domain, source="default")
    return IntentResult(retrieval_type=rtype, domain=str(data.get("domain", "")) or domain, source="llm")


def _match_domain(q: str, table: dict[str, list[str]]) -> str:
    """领域打标：命中关键词最多的领域胜出（并列取先配置的），未命中返回空串。"""
    best, best_hits = "", 0
    for dom, kws in table.items():
        hits = sum(1 for kw in kws if kw in q)
        if hits > best_hits:
            best, best_hits = dom, hits
    return best
