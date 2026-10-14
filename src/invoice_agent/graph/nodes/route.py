"""路由节点：纯函数条件边。"""


def route_entry(state) -> str:
    """入口路由：有文件走识票；仅提问走纯 RAG。"""
    if state.get("page_images") or state.get("file_path"):
        return "recognize"
    if state.get("user_question"):
        return "retrieve_docs"
    return "finalize"


def route_after_extract(state) -> str:
    """提取校验路由：失败重试 / 失败降级 / 通过。"""
    invoice = state.get("invoice")
    error = state.get("extract_error")
    if invoice is not None and not error:
        return "verify"
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
