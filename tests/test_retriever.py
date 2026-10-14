"""混合检索单测：用内存小语料 + 假 embed_fn，不联网。"""

from pathlib import Path

import pytest

from invoice_agent.services.rag.chunking import split_documents
from invoice_agent.services.rag.retriever import HybridRetriever


def fake_embed_dim(texts: list[str]) -> list[list[float]]:
    """确定性假嵌入：按字符哈希投影到 32 维。效果足够让测试通过（BM25 主导）。"""
    vecs = []
    for t in texts:
        v = [0.0] * 32
        for ch in t:
            v[ord(ch) % 32] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / norm for x in v])
    return vecs


CORPUS_DOCS = [
    {
        "source": "vat.md",
        "text": "# 增值税暂行条例\n\n## 第二十七条\n\n购进的餐饮服务、居民日常服务和娱乐服务不得抵扣进项税额。\n\n## 第十条\n\n用于集体福利或者个人消费的购进货物的进项税额不得从销项税额中抵扣。\n",
    },
    {
        "source": "company.md",
        "text": "# 公司报销制度\n\n## 第三条\n\n购物卡、礼品卡、储值卡、预付卡等购物凭证不得作为报销凭证。\n\n## 第四条\n\n市内交通费每人每天补贴上限为 80 元。\n",
    },
]


@pytest.fixture
def retriever(tmp_path: Path) -> HybridRetriever:
    """直接用假 embed_fn 构建索引（不联网）。"""
    import chromadb

    chroma_dir = tmp_path / "chroma"
    chunks = split_documents(CORPUS_DOCS)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_or_create_collection("knowledge", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=fake_embed_dim([c["text"] for c in chunks]),
        metadatas=[
            {"source": c["source"], "doc_title": c["doc_title"], "article_no": c["article_no"] or ""}
            for c in chunks
        ],
    )
    import json

    (chroma_dir / "bm25_corpus.json").write_text(json.dumps(chunks, ensure_ascii=False))
    return HybridRetriever(chroma_dir, fake_embed_dim)


class TestRetriever:
    def test_vector_and_bm25_hit(self, retriever):
        """「餐饮 抵扣」应召回餐饮不得抵扣条款。"""
        results = retriever.retrieve("餐饮服务的进项税能不能抵扣")
        assert results
        top_texts = " ".join(r.text for r in results[:4])
        assert "餐饮" in top_texts

    def test_bm25_exact_match(self, retriever):
        """关键词精确匹配（购物卡）应命中公司制度条款。"""
        results = retriever.retrieve("购物卡可以报销吗")
        assert results
        assert any("购物卡" in r.text for r in results[:4])

    def test_metadata_present(self, retriever):
        results = retriever.retrieve("餐饮服务")
        for r in results:
            assert r.source
            assert r.doc_title

    def test_top_k_limit(self, retriever):
        results = retriever.retrieve("发票 报销 抵扣 税率 限额 专票")
        assert len(results) <= 8

    def test_missing_index_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="索引不存在"):
            HybridRetriever(tmp_path / "nowhere", fake_embed_dim)
