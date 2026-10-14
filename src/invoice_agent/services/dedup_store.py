"""SQLite 存储：报销去重 + Mock 验真注册表。

短连接模式（每次操作即开即关），避免 Streamlit 多线程复用连接的问题。
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

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
"""


class DedupStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

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
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reimbursements "
                "(invoice_number, invoice_code, seller_tax_id, buyer_tax_id, total_amount, "
                " invoice_date, file_sha256, status, submitted_at) VALUES (?,?,?,?,?,?,?,?,?)",
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
                ),
            )

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
