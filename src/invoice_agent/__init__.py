"""发票助手 Agent：识票、验真、风控、政策问答。"""

from invoice_agent.monitoring.setup import setup_logging

try:
    setup_logging()
except Exception:  # noqa: BLE001, S110 — 日志初始化绝不阻塞导入
    pass
