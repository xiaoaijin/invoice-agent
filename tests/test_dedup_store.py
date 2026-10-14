from datetime import date
from pathlib import Path

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
            invoice_number="20001", invoice_code=None, seller_tax_id="S2",
            buyer_tax_id=None, total_amount=500.0,
            invoice_date=date.today(), file_sha256="x",
        )
        recent = store.find_recent_by_seller("S2", days=7, exclude_number="99999")
        assert len(recent) == 1
        assert recent[0]["invoice_number"] == "20001"

    def test_verification_registry(self, tmp_path: Path):
        store = DedupStore(tmp_path / "e.db")
        assert store.get_verification("30001") is None
        store.set_verification("30001", is_authentic=True)
        assert store.get_verification("30001") is True
