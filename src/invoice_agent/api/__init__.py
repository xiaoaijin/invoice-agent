"""HTTP API 层：供外部系统（Java/ERP）调用，复用 LangGraph 图与各 store。

运行：uv run invoice-agent serve（或 uvicorn invoice_agent.api.app:create_app --factory）

部署约束：LangGraph checkpointer 为进程内 MemorySaver，多轮会话记忆不跨进程共享，
必须单 worker 部署（uvicorn 默认 workers=1）。跨重启的历史恢复走 ConversationStore。
"""

from invoice_agent.api.app import create_app

__all__ = ["create_app"]
