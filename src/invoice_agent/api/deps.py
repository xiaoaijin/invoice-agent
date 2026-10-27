"""API 依赖：lru_cache 单例（进程内复用，测试中 cache_clear 隔离）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from invoice_agent.config import get_app_config
from invoice_agent.monitoring.store import MonitorStore
from invoice_agent.services.conversation_store import ConversationStore
from invoice_agent.services.dedup_store import DedupStore


@lru_cache
def get_monitor_store() -> MonitorStore:
    return MonitorStore(get_app_config().monitoring_db_path)


@lru_cache
def get_dedup_store() -> DedupStore:
    """去重/台账/待确认批次共用的 SQLite 库（进程内单例，测试中 cache_clear 隔离）。"""
    return DedupStore(get_app_config().db_path)


@lru_cache
def get_conversation_store() -> ConversationStore:
    """与 UI 同一路径（data/chat_history.db），历史互通。"""
    Path(get_app_config().db_path).parent.mkdir(parents=True, exist_ok=True)
    return ConversationStore(Path(get_app_config().db_path).parent / "chat_history.db")


def uploads_dir() -> Path:
    d = Path(get_app_config().db_path).parent / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d
