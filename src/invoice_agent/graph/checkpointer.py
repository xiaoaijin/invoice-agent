"""LangGraph checkpointer 工厂：memory（进程内）/ postgres（跨进程）双后端。

build_graph() 为 lru_cache 进程级单例，get_checkpointer() 每进程至多执行一次；
PostgresSaver 持有进程级专用长驻连接（不复用 pg.py 短连接——saver 在连接上自主
管理事务状态，与其他子系统共用会有事务交错风险）。setup() 幂等（按
checkpoint_migrations 表增量迁移），需 autocommit 连接（迁移含
CREATE INDEX CONCURRENTLY，不能进事务块）。
"""

from __future__ import annotations

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from invoice_agent.config import AppConfig


def build_postgres_checkpointer(dsn: str) -> BaseCheckpointSaver:
    """建长驻连接 + PostgresSaver 并 setup()。PG 不可用时启动 fail-fast。"""
    try:
        # autocommit 必须 True（setup 的 CONCURRENTLY 索引不能进事务）；
        # prepare_threshold=0 与官方 from_conn_string 一致（二进制协议性能）
        conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=0)
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            f"checkpointer.backend=postgres 但 PostgreSQL 不可用（{exc}）。"
            "请先启动 PG：docker compose up -d，或 uv run --no-sync python scripts/local_pg.py；"
            "单进程演示可在 config.yaml 将 checkpointer.backend 改回 memory"
        ) from exc
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


def get_checkpointer(cfg: AppConfig | None = None) -> BaseCheckpointSaver:
    """按 config.checkpointer.backend 分发。cfg 供测试注入，memory 分支零 PG 可测。"""
    # 延迟导入是故意的：函数体内解析模块属性，使测试 monkeypatch
    # load_app_config（tests/conftest.py）对本工厂生效
    from invoice_agent.config import get_app_config, get_settings

    cfg = cfg or get_app_config()
    backend = cfg.checkpointer.backend
    if backend == "memory":
        return MemorySaver()
    if backend == "postgres":
        return build_postgres_checkpointer(cfg.checkpointer.dsn or get_settings().pg_dsn)
    raise ValueError(f"未知 checkpointer.backend：{backend}（可选 memory / postgres）")
