"""预置 Mock 数据：验真注册表（含一张假票）+ 历史报销记录（连号/拆分演示用）。

用法：uv run python scripts/seed_mock_data.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from invoice_agent.config import get_app_config
from invoice_agent.services.dedup_store import DedupStore


def main() -> int:
    cfg = get_app_config()
    store = DedupStore(cfg.db_path)

    # 验真注册表：99999999 为假票，28374651 为真票
    store.set_verification("99999999", is_authentic=False, note="模拟假票（演示 FAIL 分支）")
    store.set_verification("28374651", is_authentic=True, note="模拟真票")

    # 历史报销记录：给样本 06（连号）与 05（敏感品后多张）提供背景数据
    # 06 的号码 28374655 与库内 28374651 差 4 ≤ 5 → R003 连号
    # 04 的销售方在 7 天窗口内已有 2 张大额 → 上传第 3 张触发 R004
    fake_sha = "seed"
    store.save_reimbursement(
        invoice_number="28374651",
        invoice_code="044032000111",
        seller_tax_id="91330100MA28YYYYYY",
        buyer_tax_id=cfg.company.tax_id,
        total_amount=1130.0,
        invoice_date=date(2026, 9, 30),
        file_sha256=fake_sha,
        status="reimbursed",
    )
    for i, (num, amount, day) in enumerate(
        [("41526001", 2800.0, 5), ("41526002", 1500.0, 6)], start=1
    ):
        store.save_reimbursement(
            invoice_number=num,
            invoice_code="044032000113",
            seller_tax_id="91330100MA28YYYYYY",
            buyer_tax_id=cfg.company.tax_id,
            total_amount=amount,
            invoice_date=date(2026, 10, day),
            file_sha256=fake_sha,
            status="submitted",
        )

    print(f"Mock 数据已写入 {cfg.db_path}")
    print("  - 验真注册表：99999999=假票，28374651=真票")
    print("  - 历史报销：28374651（连号演示）、41526001/41526002（拆分演示）")
    print(f"  - 写入时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
