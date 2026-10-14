"""检索测试：RRF 融合与 BM25 逻辑离线可测；依赖 PostgreSQL 的部分标记为 integration。"""

import pytest

from invoice_agent.models.risk import RetrievedChunk
from invoice_agent.services.rag.retriever import FINAL_TOP_K, rrf_fuse


def chunk(source: str, text: str, article: str | None = None, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        source=source, doc_title=source, article_no=article, text=text, score=score
    )


class TestRRFFuse:
    def test_both_paths_hit(self):
        """两路都命中同一 chunk 时得分叠加、排最前。"""
        c = chunk("vat.md", "餐饮服务不得抵扣", "第二十七条")
        vec = [chunk("other.md", "无关内容A", score=0.9), c]
        bm25 = [c, chunk("other.md", "无关内容B", score=3.2)]
        merged = rrf_fuse(vec, bm25, top_k=8)
        assert merged[0].text == "餐饮服务不得抵扣"
        # 叠加得分 = 两路 RRF 之和，应高于单路命中
        rrf_single = 1.0 / 60
        assert merged[0].score > rrf_single

    def test_rank_order_matters(self):
        """排名越靠前 RRF 贡献越大。"""
        a = chunk("a.md", "第一条")
        b = chunk("b.md", "第二条")
        merged = rrf_fuse([a, b], [], top_k=8)
        assert merged[0].text == "第一条"
        assert merged[0].score > merged[1].score

    def test_top_k_limit(self):
        vec = [chunk(f"s{i}.md", f"内容{i}") for i in range(20)]
        merged = rrf_fuse(vec, [], top_k=FINAL_TOP_K)
        assert len(merged) == FINAL_TOP_K

    def test_empty_paths(self):
        assert rrf_fuse([], [], top_k=8) == []

    def test_metadata_preserved(self):
        c = chunk("vat.md", "餐饮服务不得抵扣进项税额", "第二十七条")
        merged = rrf_fuse([c], [], top_k=8)
        assert merged[0].source == "vat.md"
        assert merged[0].article_no == "第二十七条"


class TestHybridRetrieverOffline:
    def test_no_pg_raises_runtime_error(self, monkeypatch):
        """PG 不可用时抛 RuntimeError（retrieve_docs 会捕获降级），而非裸异常。"""
        import psycopg

        from invoice_agent.services.rag import retriever as mod

        def fail_connect():
            raise psycopg.OperationalError("connection refused")

        monkeypatch.setattr(mod, "init_schema", fail_connect)
        with pytest.raises(RuntimeError, match="docker compose"):
            mod.HybridRetriever(lambda texts: [[0.0] * 1024 for _ in texts])


@pytest.mark.integration
class TestWithPostgres:
    """需要本地 PostgreSQL（docker compose up -d）+ 真实 embedding 的集成测试。"""

    def test_end_to_end(self):
        from invoice_agent.services.rag.retriever import get_retriever

        retriever = get_retriever()
        results = retriever.retrieve("餐饮服务的进项税能不能抵扣")
        assert results
        assert any("餐饮" in r.text for r in results[:4])
