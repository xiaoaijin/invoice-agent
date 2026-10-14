"""风控规则单测：全部用构造数据，不调 LLM。"""

from datetime import date, timedelta
from pathlib import Path

from invoice_agent.config import AppConfig, CompanyConfig, PathsConfig, RiskConfig
from invoice_agent.models.invoice import InvoiceData, InvoiceItem, InvoiceTypeEnum
from invoice_agent.models.risk import RiskLevel
from invoice_agent.rules.risk_rules import run_risk_rules
from invoice_agent.services.dedup_store import DedupStore

CONFIG = AppConfig(
    company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
    risk=RiskConfig(
        serial_gap_tolerance=5,
        split_amount_threshold=5000,
        split_window_days=7,
        split_min_count=3,
        reimburse_deadline_days=90,
        sensitive_keywords=["礼品卡", "购物卡", "烟", "酒"],
    ),
    paths=PathsConfig(),
)


def make_invoice(**kw) -> InvoiceData:
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number="30000001",
        invoice_date=date.today(),
        buyer_name="示例科技有限公司",
        buyer_tax_id="91330100MA27XXXXXX",
        seller_name="卖方公司",
        seller_tax_id="91330100MA28YYYYYY",
        amount_without_tax=100.0,
        tax_amount=13.0,
        total_amount=113.0,
        items=[InvoiceItem(name="办公用品", amount=100.0)],
    )
    base.update(kw)
    return InvoiceData(**base)


def make_ctx(tmp_path: Path) -> dict:
    return {"dedup_store": DedupStore(tmp_path / "rules.db"), "duplicate_hits": []}


class TestRules:
    def test_clean_invoice(self, tmp_path):
        findings = run_risk_rules(make_invoice(), make_ctx(tmp_path), CONFIG)
        assert findings == []

    def test_r001_buyer_mismatch(self, tmp_path):
        inv = make_invoice(buyer_tax_id="91330100MA00WRONGX")
        findings = run_risk_rules(inv, make_ctx(tmp_path), CONFIG)
        r001 = [f for f in findings if f.rule_id == "R001"]
        assert len(r001) == 1
        assert r001[0].level == RiskLevel.HIGH

    def test_r002_duplicate(self, tmp_path):
        from invoice_agent.models.risk import DuplicateHit

        ctx = make_ctx(tmp_path)
        ctx["duplicate_hits"] = [
            DuplicateHit(
                invoice_number="30000001", invoice_date=date(2026, 9, 1),
                total_amount=113.0, first_submitted_at=date(2026, 9, 2), status="reimbursed",
            )
        ]
        findings = run_risk_rules(make_invoice(), ctx, CONFIG)
        assert any(f.rule_id == "R002" and f.level == RiskLevel.HIGH for f in findings)

    def test_r003_serial(self, tmp_path):
        ctx = make_ctx(tmp_path)
        store = ctx["dedup_store"]
        # 库内 30000003 与当前 30000001 差 2 ≤ 5 → 连号
        store.save_reimbursement(
            "30000003", None, "91330100MA28YYYYYY", None, 100.0,
            date.today() - timedelta(days=1), "sha",
        )
        findings = run_risk_rules(make_invoice(), ctx, CONFIG)
        r003 = [f for f in findings if f.rule_id == "R003"]
        assert len(r003) == 1 and "30000003" in r003[0].message

    def test_r004_split(self, tmp_path):
        ctx = make_ctx(tmp_path)
        store = ctx["dedup_store"]
        # 库内 2 张共 4900，本张 200 → 合计 5100 ≥ 5000 且 3 张 → 拆分
        for num, amount in [("40001", 2800.0), ("40002", 2100.0)]:
            store.save_reimbursement(
                num, None, "91330100MA28YYYYYY", None, amount,
                date.today() - timedelta(days=1), "sha",
            )
        inv = make_invoice(total_amount=200.0, amount_without_tax=188.68, tax_amount=11.32)
        findings = run_risk_rules(inv, ctx, CONFIG)
        assert any(f.rule_id == "R004" for f in findings)

    def test_r005_sensitive(self, tmp_path):
        inv = make_invoice(items=[InvoiceItem(name="购物卡 面值1000", amount=1000.0)])
        findings = run_risk_rules(inv, make_ctx(tmp_path), CONFIG)
        r005 = [f for f in findings if f.rule_id == "R005"]
        assert len(r005) == 1 and r005[0].level == RiskLevel.HIGH

    def test_r006_deadline(self, tmp_path):
        inv = make_invoice(invoice_date=date.today() - timedelta(days=120))
        findings = run_risk_rules(inv, make_ctx(tmp_path), CONFIG)
        assert any(f.rule_id == "R006" for f in findings)

    def test_r007_future_date(self, tmp_path):
        inv = make_invoice(invoice_date=date.today() + timedelta(days=3))
        findings = run_risk_rules(inv, make_ctx(tmp_path), CONFIG)
        assert any(f.rule_id == "R007" for f in findings)

    def test_r008_amount_mismatch(self, tmp_path):
        inv = make_invoice(amount_without_tax=100.0, tax_amount=13.0, total_amount=500.0)
        findings = run_risk_rules(inv, make_ctx(tmp_path), CONFIG)
        assert any(f.rule_id == "R008" for f in findings)

    def test_r008_items_mismatch(self, tmp_path):
        inv = make_invoice(
            amount_without_tax=100.0,
            items=[InvoiceItem(name="A", amount=50.0), InvoiceItem(name="B", amount=30.0)],
        )
        findings = run_risk_rules(inv, make_ctx(tmp_path), CONFIG)
        assert any(f.rule_id == "R008" for f in findings)
