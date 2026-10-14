"""发票验真接口抽象 + Mock 实现。

未来接入百望云/诺诺/税务平台时，新增 Provider 实现并改 get_verify_provider 工厂即可，
图节点代码零改动。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime

from invoice_agent.config import get_app_config, get_settings
from invoice_agent.models.invoice import InvoiceData
from invoice_agent.models.risk import VerificationResult
from invoice_agent.services.dedup_store import DedupStore


class InvoiceVerifyProvider(ABC):
    name: str = "base"

    @abstractmethod
    def verify(self, invoice: InvoiceData) -> VerificationResult: ...


class MockInvoiceVerifyProvider(InvoiceVerifyProvider):
    """确定性 Mock 验真。

    规则（保证同号结果可复现，便于测试）：
    1. 格式校验（发票号 8/20 位、税号 15~20 位）不通过 → FAIL；
    2. 命中 SQLite verified_registry → 返回登记的真伪结果；
    3. 未命中 → 按 invoice_number 哈希伪随机：85% PASS / 15% FAIL。
    """

    name = "mock"

    def __init__(self, store: DedupStore):
        self._store = store

    def verify(self, invoice: InvoiceData) -> VerificationResult:
        # 1. 格式校验
        n = invoice.invoice_number.strip()
        if len(n) not in (8, 20) or not n.isdigit():
            return self._result("FAIL", f"发票号码格式异常（{n}）：应为 8 位或 20 位数字")
        for tax_id, label in ((invoice.buyer_tax_id, "购买方"), (invoice.seller_tax_id, "销售方")):
            if tax_id and not (15 <= len(tax_id.strip()) <= 20):
                return self._result("FAIL", f"{label}税号格式异常（{tax_id}）：应为 15~20 位")

        # 2. 注册表命中
        registered = self._store.get_verification(invoice.invoice_number)
        if registered is not None:
            if registered:
                return self._result("PASS", "查验通过（Mock 注册表登记为真票）")
            return self._result("FAIL", "查验不符：该票在登记表中为假票")

        # 3. 哈希伪随机
        digest = int(hashlib.sha256(n.encode()).hexdigest(), 16)
        if digest % 100 < 85:
            return self._result("PASS", "查验通过（Mock 伪随机，未命中注册表）")
        return self._result("FAIL", "查验不符（Mock 伪随机，未命中注册表）")

    @staticmethod
    def _result(status: str, message: str) -> VerificationResult:
        return VerificationResult(
            status=status, provider="mock", message=message, checked_at=datetime.now()
        )


def get_verify_provider(store: DedupStore | None = None) -> InvoiceVerifyProvider:
    provider = get_settings().verify_provider
    if provider == "mock":
        return MockInvoiceVerifyProvider(store or DedupStore(get_app_config().db_path))
    raise ValueError(f"未知验真提供方：{provider}（当前支持 mock）")
