"""长期对话记忆（pgvector）。

每轮问答以 "Q+A" 合并文本嵌入后存入 memory_chunks，支持跨会话语义召回；
会话滚动摘要持久化到 memory_summaries，重启不丢。
PostgreSQL 不可用时全部方法静默降级（记录 warning），不阻塞主流程。
"""

from __future__ import annotations

import logging

import psycopg

from invoice_agent.config import get_app_config, get_settings
from invoice_agent.services.pg import connect, init_schema, to_vector_literal

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, embed_fn, enabled: bool = True):
        self._embed_fn = embed_fn
        self._enabled = enabled
        self._ready = False
        if enabled:
            try:
                init_schema()
                self._ready = True
            except psycopg.OperationalError as exc:
                logger.warning("长期记忆降级为进程内模式（PostgreSQL 不可用：%s）", exc)

    # ---- 写入 ----

    def record_turn(self, session_id: str, question: str, answer: str) -> None:
        """记录一轮问答（Q+A 合并文本 → 单向量）。"""
        if not (self._enabled and self._ready and question):
            return
        text = f"问：{question}\n答：{answer[:1000]}"
        try:
            vec = self._embed_fn([text])[0]
            with connect() as conn:
                conn.execute(
                    "INSERT INTO memory_chunks (session_id, kind, content, embedding) "
                    "VALUES (%s, 'turn', %s, %s::vector)",
                    (session_id, text, to_vector_literal(vec)),
                )
        except psycopg.Error as exc:
            logger.warning("记忆写入失败（忽略）：%s", exc)

    def save_summary(self, session_id: str, content: str) -> None:
        """保存会话滚动摘要（覆盖写）。"""
        if not (self._enabled and self._ready and content):
            return
        try:
            with connect() as conn:
                conn.execute(
                    "INSERT INTO memory_summaries (session_id, content, updated_at) "
                    "VALUES (%s, %s, now()) "
                    "ON CONFLICT (session_id) DO UPDATE SET content = EXCLUDED.content, updated_at = now()",
                    (session_id, content),
                )
        except psycopg.Error as exc:
            logger.warning("摘要保存失败（忽略）：%s", exc)

    def load_summary(self, session_id: str) -> str | None:
        if not (self._enabled and self._ready):
            return None
        try:
            with connect() as conn:
                row = conn.execute(
                    "SELECT content FROM memory_summaries WHERE session_id = %s", (session_id,)
                ).fetchone()
            return row[0] if row else None
        except psycopg.Error as exc:
            logger.warning("摘要加载失败（忽略）：%s", exc)
            return None

    # ---- 召回 ----

    def recall(self, query: str, top_k: int) -> list[dict]:
        """跨会话语义召回相关历史问答。返回 [{"session_id", "content", "score"}]。"""
        if not (self._enabled and self._ready and query.strip()):
            return []
        try:
            vec = self._embed_fn([query])[0]
            with connect() as conn:
                rows = conn.execute(
                    "SELECT session_id, content, 1 - (embedding <=> %s::vector) AS score "
                    "FROM memory_chunks WHERE kind = 'turn' "
                    "ORDER BY embedding <=> %s::vector LIMIT %s",
                    (to_vector_literal(vec), to_vector_literal(vec), top_k),
                ).fetchall()
            return [{"session_id": r[0], "content": r[1], "score": float(r[2])} for r in rows]
        except psycopg.Error as exc:
            logger.warning("记忆召回失败（忽略）：%s", exc)
            return []


def _default_embed(texts: list[str]) -> list[list[float]]:
    from invoice_agent.services.llm import get_embedding_model

    return get_embedding_model().embed_documents(texts)


def get_memory_store() -> MemoryStore:
    cfg = get_app_config()
    return MemoryStore(
        _default_embed, enabled=cfg.memory.enabled and bool(get_settings().dashscope_api_key)
    )
