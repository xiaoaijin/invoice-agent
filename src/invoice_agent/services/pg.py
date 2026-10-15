"""PostgreSQL (pgvector) 连接与建表。

知识库 RAG 与长期记忆统一存储。psycopg3 短连接模式，与 SQLite DAO 风格一致；
embedding 以 vector 字面量字符串写入（'[0.1,0.2,...]'::vector），无需 pgvector Python 包。
"""

from __future__ import annotations

import logging

import psycopg

from invoice_agent.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024  # DashScope text-embedding-v3

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

-- 知识库 RAG（原 Chroma 迁移至此）。layer：三层知识库层级（national/local/company）
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    layer TEXT NOT NULL DEFAULT '',
    doc_title TEXT NOT NULL DEFAULT '',
    article_no TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_hnsw
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
-- 旧表迁移：补 layer 列
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS layer TEXT NOT NULL DEFAULT '';

-- 长期对话记忆：每轮问答存一行（Q+A 合并向量），语义召回用
CREATE TABLE IF NOT EXISTS memory_chunks (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'turn',  -- turn | summary
    content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_memory_hnsw
    ON memory_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_chunks(session_id);

-- 每会话的滚动历史摘要（压缩产生的旧对话摘要，跨重启保留）
CREATE TABLE IF NOT EXISTS memory_summaries (
    session_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 自评教训库：critic 发现的问题跨运行持久化，供后续相似问题召回优化检索与回答
-- kind：retrieval（检索教训，注入查询构造）| answer（回答教训，注入 finalize prompt）
CREATE TABLE IF NOT EXISTS feedback_lessons (
    id BIGSERIAL PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'answer',
    question TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_lessons_hnsw
    ON feedback_lessons USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_lessons_kind ON feedback_lessons(kind);
"""


def to_vector_literal(vec: list[float]) -> str:
    """embedding → '[0.1,0.2,...]' 字符串（psycopg 参数化后 ::vector 转换）。"""
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


def connect() -> psycopg.Connection:
    """新建连接（autocommit，短连接模式）。失败抛 psycopg.OperationalError。"""
    return psycopg.connect(get_settings().pg_dsn, autocommit=True)


def init_schema() -> None:
    with connect() as conn:
        # psycopg3 扩展协议不支持一次执行多条语句，逐条执行
        for stmt in SCHEMA.split(";"):
            if stmt.strip():
                conn.execute(stmt)


def pg_available() -> bool:
    try:
        connect().close()
        return True
    except psycopg.OperationalError as exc:
        logger.warning("PostgreSQL 不可用（%s）", exc)
        return False
