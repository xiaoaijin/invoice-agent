"""索引：知识库切分 → 嵌入 → 写入 PostgreSQL (pgvector)。

原 Chroma 方案已迁移至 pgvector，与长期记忆共用一个 PostgreSQL 实例。
"""

from __future__ import annotations

from pathlib import Path

from invoice_agent.services.pg import connect, init_schema, to_vector_literal
from invoice_agent.services.rag.chunking import split_documents
from invoice_agent.services.rag.loaders import load_documents


def build_index(knowledge_dir: Path, embed_fn) -> int:
    """重建知识库索引（全量替换）。embed_fn: list[str] -> list[list[float]]。"""
    docs = load_documents(knowledge_dir)
    chunks = split_documents(docs)
    if not chunks:
        raise RuntimeError(f"知识库为空：{knowledge_dir}")

    init_schema()
    embeddings = embed_fn([c["text"] for c in chunks])
    with connect() as conn:
        conn.execute("DELETE FROM knowledge_chunks")
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings, strict=True)):
            conn.execute(
                "INSERT INTO knowledge_chunks (id, source, doc_title, article_no, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                (
                    f"chunk_{i}",
                    chunk["source"],
                    chunk["doc_title"],
                    chunk["article_no"] or "",
                    chunk["text"],
                    to_vector_literal(vec),
                ),
            )
    return len(chunks)
