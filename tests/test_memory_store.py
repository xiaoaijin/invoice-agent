"""长期记忆存储测试：PG 不可用时全部方法静默降级，不抛异常。"""

from invoice_agent.services.memory_store import MemoryStore


class TestDegradeMode:
    def test_pg_down_all_methods_noop(self, monkeypatch):
        """PostgreSQL 不可用 → 构造成功、读写全部 no-op。"""
        import psycopg

        from invoice_agent.services import pg as pg_mod

        def fail_connect():
            raise psycopg.OperationalError("connection refused")

        monkeypatch.setattr(pg_mod, "connect", fail_connect)
        monkeypatch.setattr(pg_mod, "init_schema", fail_connect)

        store = MemoryStore(lambda texts: [[0.0] * 1024 for _ in texts], enabled=True)
        assert not store._ready

        # 全部方法不抛异常
        store.record_turn("s1", "q", "a")
        store.save_summary("s1", "摘要")
        assert store.load_summary("s1") is None
        assert store.recall("任何问题", top_k=5) == []

    def test_disabled_flag(self, monkeypatch):
        """enabled=False 时不触碰 PG。"""
        called = {"n": 0}
        from invoice_agent.services import pg as pg_mod

        def spy_init():
            called["n"] += 1

        monkeypatch.setattr(pg_mod, "init_schema", spy_init)
        store = MemoryStore(lambda texts: [[] for _ in texts], enabled=False)
        store.record_turn("s1", "q", "a")
        assert called["n"] == 0
        assert store.recall("q", top_k=5) == []

    def test_empty_question_skipped(self, monkeypatch):
        from invoice_agent.services import pg as pg_mod

        monkeypatch.setattr(pg_mod, "init_schema", lambda: None)
        embed_calls = []

        store = MemoryStore(lambda texts: embed_calls.append(texts) or [[0.0]], enabled=True)
        store._ready = True
        store.record_turn("s1", "", "a")
        assert embed_calls == []
