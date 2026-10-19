"""教训库测试：PG 不可用时全部方法静默降级；去重与召回过滤用假 conn 验证。

注意 patch lesson_store 模块命名空间（其 from pg import 直接引名称，patch pg_mod 无效，
否则会连到本机真实 PG）。
"""

import pytest

import invoice_agent.services.lesson_store as lesson_store_mod
from invoice_agent.services.lesson_store import LessonStore


def _fake_conn(select_rows=None):
    """最小假连接：支持 with、execute（SELECT 走预置行，INSERT 计数）。"""

    class _FakeConn:
        def __init__(self):
            self.select_rows = select_rows or []
            self.inserts: list[tuple] = []
            self.last_sql = ""
            self.last_params: tuple = ()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            if sql.lstrip().startswith(("SELECT", "--")):
                self.last_sql, self.last_params = sql, params or ()
            else:
                self.inserts.append((sql, params))
            return self

        def fetchone(self):
            return self.select_rows[0] if self.select_rows else None

        def fetchall(self):
            return self.select_rows

    return _FakeConn()


class TestDegradeMode:
    def test_pg_down_all_methods_noop(self, monkeypatch):
        """PostgreSQL 不可用 → 构造成功、读写全部 no-op。"""
        import psycopg

        def fail_connect():
            raise psycopg.OperationalError("connection refused")

        monkeypatch.setattr(lesson_store_mod, "connect", fail_connect)
        monkeypatch.setattr(lesson_store_mod, "init_schema", fail_connect)

        store = LessonStore(lambda texts: [[0.0] * 1024 for _ in texts], enabled=True)
        assert not store._ready

        # 全部方法不抛异常
        store.record("retrieval", "问题", "缺少抵扣条款")
        assert store.recall("问题", kind="retrieval", top_k=3) == []

    def test_disabled_flag(self, monkeypatch):
        """enabled=False 时不触碰 PG。"""
        called = {"n": 0}

        def spy_init():
            called["n"] += 1

        monkeypatch.setattr(lesson_store_mod, "init_schema", spy_init)
        store = LessonStore(lambda texts: [[] for _ in texts], enabled=False)
        store.record("retrieval", "q", "c")
        assert called["n"] == 0
        assert store.recall("q", top_k=3) == []

    def test_empty_content_skipped(self, monkeypatch):
        monkeypatch.setattr(lesson_store_mod, "init_schema", lambda: None)
        embed_calls = []

        store = LessonStore(lambda texts: embed_calls.append(texts) or [[0.0] * 1024], enabled=True)
        store._ready = True
        store.record("retrieval", "q", "  ")
        assert embed_calls == []


class TestRecordDedup:
    def test_same_question_content_no_second_insert(self, monkeypatch):
        conn = _fake_conn(select_rows=[(1,)])  # 去重查询命中
        monkeypatch.setattr(lesson_store_mod, "init_schema", lambda: None)
        monkeypatch.setattr(lesson_store_mod, "connect", lambda: conn)

        store = LessonStore(lambda texts: [[0.0] * 1024], enabled=True)
        store._ready = True
        store.record("retrieval", "问题", "缺少抵扣条款")
        assert conn.inserts == []  # 已存在 → 跳过写入

    def test_new_lesson_inserted(self, monkeypatch):
        conn = _fake_conn(select_rows=[])  # 去重查询未命中
        monkeypatch.setattr(lesson_store_mod, "init_schema", lambda: None)
        monkeypatch.setattr(lesson_store_mod, "connect", lambda: conn)

        store = LessonStore(lambda texts: [[0.0] * 1024], enabled=True)
        store._ready = True
        store.record("retrieval", "问题", "缺少抵扣条款")
        assert len(conn.inserts) == 1


class TestRecall:
    def test_min_score_and_kind_filter(self, monkeypatch):
        rows = [
            ("retrieval", "q1", "命中", 0.9),
            ("retrieval", "q2", "低于阈值被过滤", 0.4),
        ]
        conn = _fake_conn(select_rows=rows)
        monkeypatch.setattr(lesson_store_mod, "init_schema", lambda: None)
        monkeypatch.setattr(lesson_store_mod, "connect", lambda: conn)

        store = LessonStore(lambda texts: [[0.0] * 1024], enabled=True)
        store._ready = True
        lessons = store.recall("问题", kind="retrieval", top_k=3, min_score=0.6)
        assert [l["content"] for l in lessons] == ["命中"]
        assert lessons[0]["kind"] == "retrieval"
        assert "kind = %s" in conn.last_sql  # kind 条件下推 SQL

    def test_exception_returns_empty(self, monkeypatch):
        import psycopg

        def fail_connect():
            raise psycopg.OperationalError("down")

        monkeypatch.setattr(lesson_store_mod, "init_schema", lambda: None)
        monkeypatch.setattr(lesson_store_mod, "connect", fail_connect)

        store = LessonStore(lambda texts: [[0.0] * 1024], enabled=True)
        store._ready = True
        assert store.recall("问题", top_k=3) == []


@pytest.mark.integration
class TestWithPostgres:
    """需要本地 PostgreSQL + 真实 embedding 的集成测试（docker compose up -d）。"""

    def test_record_and_recall(self):
        from invoice_agent.services.lesson_store import get_lesson_store

        store = get_lesson_store()
        assert store._ready, "本地 PG 未启动或 feedback.lessons 未启用"
        store.record(
            "retrieval", "出租车发票能报销吗", "注意补充出租车费报销与市内交通费条款"
        )
        lessons = store.recall("打车票可以走报销流程吗", kind="retrieval", top_k=3, min_score=0.0)
        assert lessons and any("出租车" in l["content"] for l in lessons)
        # 去重：同 kind+question+content 二次写入不报错也不重复
        store.record("retrieval", "出租车发票能报销吗", "注意补充出租车费报销与市内交通费条款")
        again = store.recall("出租车发票能报销吗", kind="retrieval", top_k=10, min_score=0.0)
        assert sum(1 for l in again if l["content"] == "注意补充出租车费报销与市内交通费条款") == 1
