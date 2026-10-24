"""SQLite 存储：报销去重 + Mock 验真注册表 + 报销批次台账。

短连接模式（每次操作即开即关），避免 Streamlit 多线程复用连接的问题。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from invoice_agent.models.reimburse import ReimburseBatch, SubmissionResult
from invoice_agent.models.risk import DuplicateHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reimbursements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number TEXT NOT NULL,
    invoice_code TEXT,
    seller_tax_id TEXT,
    buyer_tax_id TEXT,
    total_amount REAL NOT NULL,
    invoice_date TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted',
    submitted_at TEXT NOT NULL,
    UNIQUE(invoice_number, seller_tax_id)
);
CREATE INDEX IF NOT EXISTS idx_reimb_seller ON reimbursements(seller_tax_id, invoice_date);

CREATE TABLE IF NOT EXISTS verified_registry (
    invoice_number TEXT PRIMARY KEY,
    is_authentic INTEGER NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS reimbursement_batches (
    batch_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    total_amount REAL NOT NULL,
    item_count INTEGER NOT NULL,
    category_summary TEXT NOT NULL,   -- CategoryTotal 列表 JSON
    provider TEXT NOT NULL,
    instance_id TEXT,
    status TEXT NOT NULL DEFAULT 'submitted',
    submitted_at TEXT NOT NULL
);
"""

# 老库幂等迁移：reimbursements 追加的可空列（SQLite 无 ADD COLUMN IF NOT EXISTS）
_EXTRA_COLUMNS = ("batch_id TEXT", "expense_category TEXT")


class DedupStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._ensure_columns(conn)

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        """老库自动迁移：按 PRAGMA 检查后逐列 ALTER（幂等）。"""
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(reimbursements)")}
        for col in _EXTRA_COLUMNS:
            name = col.split()[0]
            if name not in existing:
                conn.execute(f"ALTER TABLE reimbursements ADD COLUMN {col}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- 去重 ----

    def find_duplicates(
        self, invoice_number: str, seller_tax_id: str | None, file_sha256: str
    ) -> list[DuplicateHit]:
        """查重：同号+同销售方的已提交记录；file_sha256 命中时精确标记同一文件。"""
        hits: list[DuplicateHit] = []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reimbursements WHERE invoice_number = ? "
                "AND (? IS NULL OR seller_tax_id = ? OR seller_tax_id IS NULL)",
                (invoice_number, seller_tax_id, seller_tax_id),
            ).fetchall()
        for r in rows:
            hits.append(
                DuplicateHit(
                    invoice_number=r["invoice_number"],
                    invoice_date=date.fromisoformat(r["invoice_date"]),
                    total_amount=r["total_amount"],
                    first_submitted_at=datetime.fromisoformat(r["submitted_at"]),
                    status="同一文件重复上传" if r["file_sha256"] == file_sha256 else r["status"],
                )
            )
        return hits

    def find_recent_by_seller(
        self, seller_tax_id: str, days: int, exclude_number: str
    ) -> list[dict]:
        """连号/拆分判定用：查同销售方近期记录。"""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT invoice_number, invoice_date, total_amount, invoice_code FROM reimbursements "
                "WHERE seller_tax_id = ? AND invoice_date >= ? AND invoice_number != ? "
                "ORDER BY invoice_date DESC",
                (seller_tax_id, since, exclude_number),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_reimbursement(
        self,
        invoice_number: str,
        invoice_code: str | None,
        seller_tax_id: str | None,
        buyer_tax_id: str | None,
        total_amount: float,
        invoice_date: date,
        file_sha256: str,
        status: str = "submitted",
        batch_id: str | None = None,
        expense_category: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reimbursements "
                "(invoice_number, invoice_code, seller_tax_id, buyer_tax_id, total_amount, "
                " invoice_date, file_sha256, status, submitted_at, batch_id, expense_category) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    invoice_number,
                    invoice_code,
                    seller_tax_id,
                    buyer_tax_id,
                    total_amount,
                    invoice_date.isoformat(),
                    file_sha256,
                    status,
                    datetime.now().isoformat(),
                    batch_id,
                    expense_category,
                ),
            )

    # ---- 报销批次 ----

    def save_batch(self, batch: ReimburseBatch, result: SubmissionResult) -> None:
        """单事务写入批次汇总 + 逐张 upsert 台账（识别失败的条目不进台账）。

        与单票「确认后才写台账」约定一致：写入即参与后续重复报销检测。
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reimbursement_batches "
                "(batch_id, title, total_amount, item_count, category_summary, provider, "
                " instance_id, status, submitted_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    batch.batch_id,
                    batch.title,
                    batch.total_amount,
                    len(batch.items),
                    json.dumps(
                        [t.model_dump(mode="json") for t in batch.category_totals],
                        ensure_ascii=False,
                    ),
                    result.provider,
                    result.instance_id,
                    result.status,
                    result.submitted_at.isoformat(),
                ),
            )
            for item in batch.items:
                inv = item.invoice
                if inv is None:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO reimbursements "
                    "(invoice_number, invoice_code, seller_tax_id, buyer_tax_id, total_amount, "
                    " invoice_date, file_sha256, status, submitted_at, batch_id, expense_category) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        inv.invoice_number,
                        inv.invoice_code,
                        inv.seller_tax_id,
                        inv.buyer_tax_id,
                        inv.total_amount,
                        str(inv.invoice_date),
                        item.file_sha256,
                        result.status,
                        result.submitted_at.isoformat(),
                        batch.batch_id,
                        item.category.value,
                    ),
                )

    def get_batch(self, batch_id: str) -> dict | None:
        """批次汇总（category_summary 反序列化为列表）；不存在返回 None。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reimbursement_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["category_summary"] = json.loads(d["category_summary"])
        return d

    def list_batches(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reimbursement_batches ORDER BY submitted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["category_summary"] = json.loads(d["category_summary"])
            result.append(d)
        return result

    def batch_items(self, batch_id: str) -> list[dict]:
        """批次内的逐张台账记录（识别失败未入台账的不在其中）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reimbursements WHERE batch_id = ? ORDER BY id",
                (batch_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Mock 验真注册表 ----

    def set_verification(self, invoice_number: str, is_authentic: bool, note: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO verified_registry (invoice_number, is_authentic, note) "
                "VALUES (?,?,?)",
                (invoice_number, int(is_authentic), note),
            )

    def get_verification(self, invoice_number: str) -> bool | None:
        """命中返回真伪；未命中返回 None。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT is_authentic FROM verified_registry WHERE invoice_number = ?",
                (invoice_number,),
            ).fetchone()
        return None if row is None else bool(row["is_authentic"])
