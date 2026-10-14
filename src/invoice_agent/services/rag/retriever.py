"""混合检索：Chroma 向量 + BM25（jieba 分词）→ RRF 融合。

税法条文的「专票/抵扣」等术语精确匹配必须靠关键词；向量负责语义泛化。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import chromadb
import jieba
from rank_bm25 import BM25Okapi

from invoice_agent.config import get_app_config
from invoice_agent.models.risk import RetrievedChunk
from invoice_agent.services.rag.indexer import CORPUS_FILENAME

RRF_K = 60
VECTOR_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 8


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip()]


class HybridRetriever:
    """启动时加载 Chroma + BM25。语料为空时抛出明确错误提示先建索引。"""

    def __init__(self, chroma_dir: Path, embed_fn):
        corpus_file = chroma_dir / CORPUS_FILENAME
        if not corpus_file.exists():
            raise RuntimeError(
                f"知识库索引不存在（{chroma_dir}）。请先运行："
                "uv run python scripts/build_knowledge_index.py"
            )
        self._chunks = json.loads(corpus_file.read_text(encoding="utf-8"))
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self._chunks])

        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._col = client.get_or_create_collection("knowledge")
        self._embed_fn = embed_fn

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[RetrievedChunk]:
        # 向量检索
        vec_result = self._col.query(
            query_embeddings=self._embed_fn([query]), n_results=min(VECTOR_TOP_K, len(self._chunks))
        )
        vec_ids = vec_result["ids"][0]
        vec_meta = vec_result["metadatas"][0]
        vec_docs = vec_result["documents"][0]

        # BM25 检索
        scores = self._bm25.get_scores(_tokenize(query))
        bm25_ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm25_ranked = [i for i in bm25_ranked if scores[i] > 0][:BM25_TOP_K]

        # RRF 融合
        rrf: dict[int, float] = {}
        for rank, cid in enumerate(vec_ids):
            idx = int(cid.removeprefix("chunk_"))
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for idx, score in top:
            meta = vec_meta[vec_ids.index(f"chunk_{idx}")] if f"chunk_{idx}" in vec_ids else self._chunks[idx]
            text = vec_docs[vec_ids.index(f"chunk_{idx}")] if f"chunk_{idx}" in vec_ids else self._chunks[idx]["text"]
            results.append(
                RetrievedChunk(
                    text=text,
                    source=meta.get("source", ""),
                    doc_title=meta.get("doc_title", ""),
                    article_no=meta.get("article_no") or None,
                    score=round(score, 6),
                )
            )
        return results


@lru_cache
def get_retriever() -> HybridRetriever:
    from invoice_agent.services.llm import get_embedding_model

    cfg = get_app_config()
    return HybridRetriever(cfg.chroma_dir, lambda texts: get_embedding_model().embed_documents(texts))
