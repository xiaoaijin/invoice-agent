"""混合检索：pgvector 语义检索 + BM25（jieba 分词，进程内存构建）→ RRF 融合。

税法条文的「专票/抵扣」等术语精确匹配必须靠关键词；向量负责语义泛化。
"""

from __future__ import annotations

import logging
from functools import lru_cache

import jieba
import psycopg
from rank_bm25 import BM25Okapi

from invoice_agent.models.risk import RetrievedChunk
from invoice_agent.services.pg import connect, init_schema, to_vector_literal

logger = logging.getLogger(__name__)

RRF_K = 60
VECTOR_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 8


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip()]


def rrf_fuse(vec_hits: list[RetrievedChunk], bm25_hits: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """RRF 融合两路检索结果（纯函数，key 为 source+article_no+text 前缀）。"""
    def key(c: RetrievedChunk) -> tuple:
        return (c.source, c.article_no, c.text)

    scores: dict[tuple, float] = {}
    best: dict[tuple, RetrievedChunk] = {}
    for hits in (vec_hits, bm25_hits):
        for rank, chunk in enumerate(hits):
            k = key(chunk)
            scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
            best.setdefault(k, chunk)
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    results = []
    for k, score in top:
        chunk = best[k].model_copy(update={"score": round(score, 6)})
        results.append(chunk)
    return results


class HybridRetriever:
    """pgvector 语义检索 + 进程内 BM25。启动时从 PG 加载语料构建 BM25。"""

    def __init__(self, embed_fn):
        try:
            init_schema()
            self._conn = connect()
        except psycopg.OperationalError as exc:
            raise RuntimeError(
                f"PostgreSQL 不可用（{exc}）。请先启动：docker compose up -d"
            ) from exc
        rows = self._conn.execute(
            "SELECT id, source, doc_title, article_no, content FROM knowledge_chunks ORDER BY id"
        ).fetchall()
        if not rows:
            self._conn.close()
            raise RuntimeError("知识库为空，请先运行：uv run python scripts/build_knowledge_index.py")
        self._chunks = [
            {"id": r[0], "source": r[1], "doc_title": r[2], "article_no": r[3], "text": r[4]}
            for r in rows
        ]
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self._chunks])
        self._embed_fn = embed_fn

    def _vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vec = self._embed_fn([query])[0]
        rows = self._conn.execute(
            "SELECT source, doc_title, article_no, content, 1 - (embedding <=> %s::vector) AS score "
            "FROM knowledge_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (to_vector_literal(vec), to_vector_literal(vec), top_k),
        ).fetchall()
        return [
            RetrievedChunk(source=r[0], doc_title=r[1], article_no=r[2] or None, text=r[3], score=float(r[4]))
            for r in rows
        ]

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = [i for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True) if scores[i] > 0]
        return [
            RetrievedChunk(
                source=self._chunks[i]["source"],
                doc_title=self._chunks[i]["doc_title"],
                article_no=self._chunks[i]["article_no"] or None,
                text=self._chunks[i]["text"],
                score=float(scores[i]),
            )
            for i in ranked[:top_k]
        ]

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[RetrievedChunk]:
        vec_hits = self._vector_search(query, min(VECTOR_TOP_K, len(self._chunks)))
        bm25_hits = self._bm25_search(query, min(BM25_TOP_K, len(self._chunks)))
        return rrf_fuse(vec_hits, bm25_hits, top_k)

    def close(self) -> None:
        self._conn.close()


@lru_cache
def get_retriever() -> HybridRetriever:
    from invoice_agent.services.llm import get_embedding_model

    return HybridRetriever(lambda texts: get_embedding_model().embed_documents(texts))
