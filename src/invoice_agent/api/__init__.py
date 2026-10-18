"""HTTP API 层：供外部系统（Java/ERP）调用，复用 LangGraph 图与各 store。

运行：uv run invoice-agent serve（或 uvicorn invoice_agent.api.app:create_app --factory）

部署约束：须单 worker 部署（uvicorn 默认 workers=1）——checkpointer 后端由 config.yaml
checkpointer.backend 决定（postgres 时多轮会话状态落 PG，跨进程/跨重启可恢复；memory
为进程内），但监控/台账/会话历史三个 SQLite 文件库与 BM25 进程内索引仍不兼容多 worker。
"""

from invoice_agent.api.app import create_app

__all__ = ["create_app"]
