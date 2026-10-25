"""路由节点：纯函数条件边。"""

# 免检索的意图类型：闲聊 + 免检索（答案完全由上文/发票结构化数据直接可得）
_SKIP_INTENT_TYPES = {"chat", "no_retrieval"}


def _skip_retrieval(state) -> bool:
    """是否跳过 RAG 检索（route_entry 与 route_after_risk 共用判定，防止语义漂移）。

    - 无用户提问：无问题可答，检索无意义（finalize 有各自兜底）；
    - intent 缺失（旧 checkpointer 线程/直测）：fail-safe 视为需检索；
    - 闲聊/免检索意图：跳过检索直接作答。
    """
    if not state.get("user_question"):
        return True
    intent = state.get("intent")
    if intent is None:
        return False
    return (intent or {}).get("retrieval_type") in _SKIP_INTENT_TYPES


def route_entry(state) -> str:
    """入口路由：有文件走识票；仅提问走纯 RAG（闲聊/免检索短路直接 finalize）。"""
    if state.get("page_images") or state.get("file_path"):
        return "recognize"  # 有发票仍走识票全流程；意图只影响检索 query 组装与是否检索
    if state.get("user_question"):
        # 闲聊/免检索：跳过检索直接作答（state 无 intent 字段视为新问题，兼容旧 checkpointer 线程）
        if _skip_retrieval(state):
            return "finalize"
        return "retrieve_docs"
    return "finalize"


def route_after_risk(state) -> str:
    """风控复核后路由：需要政策依据才检索；识票无提问/闲聊/免检索直达 finalize（合规校验由 verify/dedup/risk 承担，不依赖 RAG）。"""
    return "finalize" if _skip_retrieval(state) else "retrieve_docs"


def route_after_extract(state) -> str | list[str]:
    """提取校验路由：失败重试 / 失败降级 / 通过。

    通过时返回 list 触发 verify 与 dedup 同 superstep 并行（LangGraph 条件边支持 fan-out）。
    """
    invoice = state.get("invoice")
    error = state.get("extract_error")
    if invoice is not None and not error:
        return ["verify", "dedup"]
    retry = state.get("recognize_retry", 0)
    if retry < 2:
        return "recognize"
    return "finalize"


def route_after_critique(state) -> str:
    """回答自评路由：通过（或重试耗尽已放行）→ END；不通过 → finalize 重生成。"""
    critique = state.get("critique") or {}
    if critique.get("passed", True):
        return "END"
    return "finalize"
