"""会话持久化：对话历史 SQLite 存储（近期对话列表 + 消息恢复）。

与去重库/监控库同为独立 SQLite（data/chat_history.db），可随时删除。
消息条目中的 pydantic 模型（InvoiceData / VerificationResult / Finding /
RetrievedChunk）以 `__type__` 标签序列化，恢复时按标签还原为原对象，
UI 渲染层无需区分来自实时运行还是历史加载。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel

from invoice_agent.models.invoice import InvoiceData
from invoice_agent.models.risk import DuplicateHit, Finding, RetrievedChunk
from invoice_agent.services.verify_provider import VerificationResult

_MODEL_BY_NAME: dict[str, type[BaseModel]] = {
    "InvoiceData": InvoiceData,
    "VerificationResult": VerificationResult,
    "Finding": Finding,
    "RetrievedChunk": RetrievedChunk,
    "DuplicateHit": DuplicateHit,
}


def _encode(obj):
    if isinstance(obj, BaseModel):
        return {"__type__": type(obj).__name__, **obj.model_dump(mode="json")}
    raise TypeError(f"不可序列化的条目字段类型：{type(obj)!r}")


def _decode(d: dict):
    if isinstance(d, dict) and (name := d.get("__type__")) in _MODEL_BY_NAME:
        return _MODEL_BY_NAME[name].model_validate(d)
    return d


class ConversationStore:
    """conversations 表：thread_id 主键 + 标题 + 更新时间 + 完整消息 JSON。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at REAL NOT NULL,
                entries TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def save(self, thread_id: str, title: str, entries: list[dict]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?)",
            (thread_id, title, time.time(), json.dumps(entries, default=_encode, ensure_ascii=False)),
        )
        self._conn.commit()

    def list_conversations(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT thread_id, title, updated_at FROM conversations"
            " ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

    def get_entries(self, thread_id: str) -> list[dict]:
        row = self._conn.execute(
            "SELECT entries FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return json.loads(row[0], object_hook=_decode) if row else []

    def delete(self, thread_id: str) -> None:
        self._conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        self._conn.commit()
