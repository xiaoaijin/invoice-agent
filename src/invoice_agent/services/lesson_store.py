"""自评教训库（pgvector）。

critic 自评发现的问题（检索不足 / 回答缺陷）跨运行持久化到 feedback_lessons，
后续运行遇到相似问题时语义召回，注入检索查询构造或 finalize prompt。
PostgreSQL 不可用时全部方法静默降级（记录 warning），不阻塞主流程。
"""

from __future__ import annotations

import logging

import psycopg

from invoice_agent.config import get_app_config, get_settings
from invoice_agent.services.pg import connect, init_schema, to_vector_literal

logger = logging.getLogger(__name__)


class LessonStore:
    def __init__(self, embed_fn, enabled: bool = True):
        self._embed_fn = embed_fn
        self._enabled = enabled
        self._ready = False
        if enabled:
            try:
                init_schema()
                self._ready = True
            except psycopg.OperationalError as exc:
                logger.warning("教训库降级为不可用模式（PostgreSQL 不可用：%s）", exc)

    # ---- 写入 ----

    def record(self, kind: str, question: str, content: str) -> None:
        """记录一条教训（同 kind+question+content 去重，防反复落库膨胀）。"""
        if not (self._enabled and self._ready and content.strip()):
            return
        try:
            with connect() as conn:
                dup = conn.execute(
                    "SELECT 1 FROM feedback_lessons "
                    "WHERE kind = %s AND question = %s AND content = %s LIMIT 1",
                    (kind, question, content),
                ).fetchone()
                if dup:
                    return
                vec = self._embed_fn([f"{question}\n{content}"])[0]
                conn.execute(
                    "INSERT INTO feedback_lessons (kind, question, content, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (kind, question, content, to_vector_literal(vec)),
                )
        except psycopg.Error as exc:
            logger.warning("教训写入失败（忽略）：%s", exc)

    # ---- 召回 ----

    def recall(
        self, query: str, kind: str | None = None, top_k: int = 3, min_score: float = 0.6
    ) -> list[dict]:
        """语义召回相似问题的教训。返回 [{"kind", "question", "content", "score"}]。"""
        if not (self._enabled and self._ready and query.strip()):
            return []
        try:
            vec = self._embed_fn([query])[0]
            sql = (
                "SELECT kind, question, content, 1 - (embedding <=> %s::vector) AS score "
                "FROM feedback_lessons"
            )
            params: list = [to_vector_literal(vec)]
            if kind is not None:
                sql += " WHERE kind = %s"
                params.append(kind)
            sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params += [to_vector_literal(vec), top_k]
            with connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [
                {"kind": r[0], "question": r[1], "content": r[2], "score": float(r[3])}
                for r in rows
                if float(r[3]) >= min_score
            ]
        except psycopg.Error as exc:
            logger.warning("教训召回失败（忽略）：%s", exc)
            return []


def _default_embed(texts: list[str]) -> list[list[float]]:
    from invoice_agent.services.llm import get_embedding_model

    return get_embedding_model().embed_documents(texts)


def get_lesson_store() -> LessonStore:
    cfg = get_app_config()
    return LessonStore(
        _default_embed, enabled=cfg.feedback.lessons.enabled and bool(get_settings().dashscope_api_key)
    )
