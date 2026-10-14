"""监控装配：日志初始化 + 一次 run 的采集配置组装。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from invoice_agent.config import get_app_config
from invoice_agent.monitoring.handler import TraceCallbackHandler
from invoice_agent.monitoring.store import MonitorStore

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_MARKER = "_invoice_agent_handler"  # 幂等标记，防重复添加 handler


def setup_logging() -> None:
    """root logger 挂 console + rotating file handler，幂等可重复调用。"""
    try:
        cfg = get_app_config().monitoring
    except Exception:  # noqa: BLE001 — 配置缺失时不阻塞（如单测环境无 config.yaml）
        return
    if not cfg.enabled:
        return

    root = logging.getLogger()
    root.setLevel(cfg.log_level.upper())

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT))
    setattr(console, _MARKER, True)

    log_path = get_app_config().resolve(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    setattr(file_handler, _MARKER, True)

    # 清掉旧的同源 handler 再挂（幂等：重复调用不翻倍）
    for h in [h for h in root.handlers if getattr(h, _MARKER, False)]:
        root.removeHandler(h)
    root.addHandler(console)
    root.addHandler(file_handler)


def build_run_config(
    thread_id: str, run_type: str = "unknown"
) -> tuple[dict, TraceCallbackHandler | None]:
    """组装带监控回调的 invoke/stream config。返回 (config, handler)；监控关闭时 handler 为 None。"""
    cfg = get_app_config()
    if not cfg.monitoring.enabled:
        return {"configurable": {"thread_id": thread_id}}, None
    store = MonitorStore(cfg.monitoring_db_path)
    handler = TraceCallbackHandler(store, cfg.monitoring, thread_id, run_type)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [handler],
        "metadata": {"thread_id": thread_id, "run_type": run_type},
    }
    return config, handler
