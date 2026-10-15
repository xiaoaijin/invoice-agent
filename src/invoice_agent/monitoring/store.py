"""监控持久层：SQLite 短连接模式（照抄 DedupStore）。

监控是旁路数据：所有写方法吞异常只打 warning，绝不反噬主流程。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    thread_id TEXT,
    run_type TEXT,
    started_at TEXT,
    ended_at TEXT,
    status TEXT,                    -- running / success / failure
    total_latency_ms REAL,
    node_count INTEGER,
    retry_count INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_yuan REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS node_spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    node TEXT,
    seq INTEGER,
    started_at TEXT,
    latency_ms REAL,
    status TEXT,                    -- success / failure
    retry_of INTEGER DEFAULT 0,     -- 同节点第几次执行（0=首次）
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON node_spans(run_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    node TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms REAL,
    cost_yuan REAL,
    success INTEGER,
    error TEXT,
    prompt_name TEXT,               -- 本次调用使用的 prompt 名（registry 登记）
    prompt_version TEXT,            -- 所用版本（A/B 对照追溯）
    prompt_hash TEXT                -- 文本 sha256 前 12 位（跨库追溯同一文本）
);
CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_calls(run_id);

CREATE TABLE IF NOT EXISTS prompt_versions (
    name TEXT,
    version TEXT,
    hash TEXT,                      -- 文本 sha256 前 12 位
    description TEXT,
    text TEXT,                      -- 全文快照（YAML 为源，此处只做追溯）
    created_at TEXT,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    created_at TEXT,
    alert_type TEXT,                -- slow_node/slow_run/failure/high_retry/low_success_rate
    severity TEXT,                  -- WARN / ERROR
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
"""

# 旧库轻量迁移：llm_calls 的 prompt 追溯三列（新库由 _SCHEMA 直接建出，执行报 duplicate column 跳过）
_MIGRATIONS = (
    "ALTER TABLE llm_calls ADD COLUMN prompt_name TEXT",
    "ALTER TABLE llm_calls ADD COLUMN prompt_version TEXT",
    "ALTER TABLE llm_calls ADD COLUMN prompt_hash TEXT",
)


class MonitorStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """旧库补列。SQLite 不支持 ADD COLUMN IF NOT EXISTS，靠捕获 duplicate column 跳过。"""
        for ddl in _MIGRATIONS:
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError as exc:
                logger.debug("监控迁移跳过：%s（%s）", ddl, exc)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- 写入（全部吞异常）----

    def start_run(self, run_id: str, thread_id: str, run_type: str) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO runs (run_id, thread_id, run_type, started_at, status) "
                    "VALUES (?,?,?,?, 'running')",
                    (run_id, thread_id, run_type, _now()),
                )
        except Exception as exc:  # noqa: BLE001 — 监控不反噬主流程
            logger.warning("监控写库失败（start_run）：%s", exc)

    def end_run(
        self,
        run_id: str,
        status: str,
        total_latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_yuan: float,
        error: str | None = None,
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE runs SET ended_at=?, status=?, total_latency_ms=?, "
                    "node_count=(SELECT COUNT(*) FROM node_spans WHERE run_id=?), "
                    "retry_count=(SELECT COUNT(*) FROM node_spans WHERE run_id=? AND retry_of>0), "
                    "prompt_tokens=?, completion_tokens=?, cost_yuan=?, error=? "
                    "WHERE run_id=?",
                    (
                        _now(),
                        status,
                        total_latency_ms,
                        run_id,
                        run_id,
                        prompt_tokens,
                        completion_tokens,
                        cost_yuan,
                        error,
                        run_id,
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("监控写库失败（end_run）：%s", exc)

    def write_span(
        self,
        run_id: str,
        node: str,
        seq: int,
        started_at: str,
        latency_ms: float,
        status: str,
        retry_of: int,
        error: str | None = None,
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO node_spans "
                    "(run_id, node, seq, started_at, latency_ms, status, retry_of, error) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, node, seq, started_at, latency_ms, status, retry_of, error),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("监控写库失败（write_span）：%s", exc)

    def write_llm_call(
        self,
        run_id: str,
        node: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        latency_ms: float,
        cost_yuan: float,
        success: bool,
        error: str | None = None,
        *,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO llm_calls "
                    "(run_id, node, model, prompt_tokens, completion_tokens, total_tokens, "
                    " latency_ms, cost_yuan, success, error, "
                    " prompt_name, prompt_version, prompt_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        node,
                        model,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        latency_ms,
                        cost_yuan,
                        int(success),
                        error,
                        prompt_name,
                        prompt_version,
                        prompt_hash,
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("监控写库失败（write_llm_call）：%s", exc)

    def upsert_prompt_version(
        self, name: str, version: str, hash: str, description: str, text: str
    ) -> None:
        """prompt 版本全文快照（registry 加载时调用；name+version 为主键，重复加载不重复）。"""
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO prompt_versions "
                    "(name, version, hash, description, text, created_at) VALUES (?,?,?,?,?,?)",
                    (name, version, hash, description, text, _now()),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("监控写库失败（upsert_prompt_version）：%s", exc)

    def write_alert(self, run_id: str, alert_type: str, severity: str, detail: str) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO alerts (run_id, created_at, alert_type, severity, detail) "
                    "VALUES (?,?,?,?,?)",
                    (run_id, _now(), alert_type, severity, detail),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("监控写库失败（write_alert）：%s", exc)

    # ---- 查询 ----

    def recent_runs(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def run_detail(self, run_id: str) -> dict:
        """单个 run 的节点时间线。"""
        with self._conn() as conn:
            run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            spans = conn.execute(
                "SELECT * FROM node_spans WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()
        return {"run": dict(run) if run else None, "spans": [dict(s) for s in spans]}

    def stats(self, window: int = 20) -> dict:
        """近 window 次 run 的聚合指标。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS ok, "
                "AVG(total_latency_ms) AS avg_latency, "
                "SUM(COALESCE(prompt_tokens,0)) AS prompt_tokens, "
                "SUM(COALESCE(completion_tokens,0)) AS completion_tokens, "
                "SUM(COALESCE(cost_yuan,0)) AS cost "
                "FROM (SELECT * FROM runs ORDER BY started_at DESC LIMIT ?)",
                (window,),
            ).fetchone()
            alerts = conn.execute("SELECT COUNT(*) AS n FROM alerts").fetchone()
        n = row["n"] or 0
        return {
            "run_count": n,
            "success_rate": (row["ok"] or 0) / n if n else None,
            "avg_latency_ms": row["avg_latency"],
            "prompt_tokens": row["prompt_tokens"] or 0,
            "completion_tokens": row["completion_tokens"] or 0,
            "cost_yuan": row["cost"] or 0.0,
            "alert_count": alerts["n"] or 0,
        }

    def recent_alerts(self, limit: int = 30) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def prompt_usage(self, limit: int = 200) -> list[dict]:
        """按 prompt 名+版本聚合调用量/成功率/均耗时/成本（A/B 对照数据源）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT prompt_name, prompt_version, COUNT(*) AS calls, "
                "AVG(latency_ms) AS avg_latency_ms, "
                "SUM(COALESCE(cost_yuan, 0)) AS cost_yuan, "
                "AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) AS success_rate "
                "FROM llm_calls WHERE prompt_name IS NOT NULL "
                "GROUP BY prompt_name, prompt_version "
                "ORDER BY prompt_name, calls DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_prompt_versions(self) -> list[dict]:
        """prompt 版本快照清单（UI 展示用）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, version, hash, description, created_at FROM prompt_versions "
                "ORDER BY name, version"
            ).fetchall()
        return [dict(r) for r in rows]


def _now() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="milliseconds")
