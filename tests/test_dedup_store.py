import sqlite3
from datetime import date, datetime
from pathlib import Path

from invoice_agent.models.invoice import InvoiceData, InvoiceTypeEnum
from invoice_agent.models.reimburse import (
    BatchItem,
    CategoryTotal,
    ExpenseCategory,
    ReimburseBatch,
    SubmissionResult,
)
from invoice_agent.services.dedup_store import DedupStore


class TestDedupStore:
    def test_init_creates_tables(self, tmp_path: Path):
        DedupStore(tmp_path / "a.db")
        assert (tmp_path / "a.db").exists()
        # 二次初始化不报错
        DedupStore(tmp_path / "a.db")

    def test_duplicate_detection(self, tmp_path: Path):
        store = DedupStore(tmp_path / "b.db")
        store.save_reimbursement(
            invoice_number="10001",
            invoice_code="code",
            seller_tax_id="S1",
            buyer_tax_id="B1",
            total_amount=100.0,
            invoice_date=date(2026, 9, 1),
            file_sha256="sha_a",
        )
        hits = store.find_duplicates("10001", "S1", "sha_other")
        assert len(hits) == 1
        assert hits[0].status == "submitted"  # 不同文件

        hits_same_file = store.find_duplicates("10001", "S1", "sha_a")
        assert hits_same_file[0].status == "同一文件重复上传"

    def test_no_false_positive(self, tmp_path: Path):
        store = DedupStore(tmp_path / "c.db")
        assert store.find_duplicates("10002", "S1", "sha") == []

    def test_recent_by_seller(self, tmp_path: Path):
        store = DedupStore(tmp_path / "d.db")
        store.save_reimbursement(
            invoice_number="20001",
            invoice_code=None,
            seller_tax_id="S2",
            buyer_tax_id=None,
            total_amount=500.0,
            invoice_date=date.today(),
            file_sha256="x",
        )
        recent = store.find_recent_by_seller("S2", days=7, exclude_number="99999")
        assert len(recent) == 1
        assert recent[0]["invoice_number"] == "20001"

    def test_verification_registry(self, tmp_path: Path):
        store = DedupStore(tmp_path / "e.db")
        assert store.get_verification("30001") is None
        store.set_verification("30001", is_authentic=True)
        assert store.get_verification("30001") is True


class TestLegacyMigration:
    def _create_legacy_db(self, path: Path) -> None:
        """模拟未迁移的老库：reimbursements 无 batch_id/expense_category 两列。"""
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE reimbursements (
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
            """
        )
        conn.execute(
            "INSERT INTO reimbursements (invoice_number, total_amount, invoice_date, "
            "file_sha256, submitted_at) VALUES ('90001', 50.0, '2026-01-01', 'sha', '2026-01-01')"
        )
        conn.commit()
        conn.close()

    def test_open_legacy_db_migrates(self, tmp_path: Path):
        """老库直接打开不炸，且自动补齐新列（历史数据保留）。"""
        path = tmp_path / "legacy.db"
        self._create_legacy_db(path)
        store = DedupStore(path)
        rows = store.find_recent_by_seller("S", days=7, exclude_number="x")
        assert rows == []  # 老数据 seller_tax_id 为 NULL，不受影响
        # 新列可用
        store.save_reimbursement(
            invoice_number="90002",
            invoice_code=None,
            seller_tax_id="S1",
            buyer_tax_id=None,
            total_amount=10.0,
            invoice_date=date(2026, 10, 1),
            file_sha256="sha2",
            batch_id="b1",
            expense_category="餐饮",
        )
        assert store.batch_items("b1")[0]["invoice_number"] == "90002"


class TestBatchStore:
    def _make_batch(self, batch_id: str) -> ReimburseBatch:
        invoice = InvoiceData(
            invoice_type=InvoiceTypeEnum.SPECIAL,
            invoice_number="80001",
            invoice_date=date(2026, 10, 1),
            buyer_name="买方",
            seller_name="卖方",
            total_amount=113.0,
        )
        return ReimburseBatch(
            batch_id=batch_id,
            title="十月报销",
            items=[
                BatchItem(
                    file_name="a.png",
                    file_sha256="sha_a",
                    invoice=invoice,
                    category=ExpenseCategory.DINING,
                )
            ],
            category_totals=[CategoryTotal(category=ExpenseCategory.DINING, count=1, total_amount=113.0)],
            total_amount=113.0,
        )

    def test_save_and_get_batch(self, tmp_path: Path):
        store = DedupStore(tmp_path / "f.db")
        result = SubmissionResult(
            status="submitted", instance_id="mock-x", provider="mock", message="ok"
        )
        store.save_batch(self._make_batch("b1"), result)
        got = store.get_batch("b1")
        assert got["title"] == "十月报销"
        assert got["instance_id"] == "mock-x"
        assert got["category_summary"][0]["category"] == "餐饮"
        assert got["submitted_at"] == result.submitted_at.isoformat()

    def test_get_batch_missing_returns_none(self, tmp_path: Path):
        store = DedupStore(tmp_path / "g.db")
        assert store.get_batch("nope") is None

    def test_list_batches_ordering(self, tmp_path: Path):
        store = DedupStore(tmp_path / "h.db")
        for i in ("b1", "b2"):
            store.save_batch(
                self._make_batch(i),
                SubmissionResult(
                    status="submitted", instance_id="x", provider="mock", message="ok",
                    submitted_at=datetime(2026, 10, 1, 10, 0, int(i[-1])),
                ),
            )
        ids = [b["batch_id"] for b in store.list_batches()]
        assert ids == ["b2", "b1"]  # 按提交时间倒序

    def test_batch_items_roundtrip(self, tmp_path: Path):
        store = DedupStore(tmp_path / "i.db")
        store.save_batch(
            self._make_batch("b3"),
            SubmissionResult(status="submitted", instance_id="x", provider="mock", message="ok"),
        )
        items = store.batch_items("b3")
        assert len(items) == 1
        assert items[0]["batch_id"] == "b3"
        assert items[0]["expense_category"] == "餐饮"
