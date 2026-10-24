"""报销审批流程提供方抽象 + Mock 实现。

Mock 阶段本地打桩模拟提交成功；未来接真实飞书时新增 FeishuExpenseFlowProvider
并改 get_expense_flow_provider 工厂即可，编排/UI/API 层零改动。

真实飞书实现要点（三类接口）：
1. POST /open-apis/auth/v3/tenant_access_token/internal（app_id + app_secret）
   取 tenant_access_token；
2. 组装 approval_code（.env 的 FEISHU_APPROVAL_CODE）+ user_id + form
   （按 category_totals 填明细表格节点）；
3. POST /open-apis/approval/v4/instances 创建审批实例，取返回的 data.instance_id。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime

from invoice_agent.config import get_settings
from invoice_agent.models.reimburse import ReimburseBatch, SubmissionResult


class ExpenseFlowProvider(ABC):
    name: str = "base"

    @abstractmethod
    def submit(self, batch: ReimburseBatch) -> SubmissionResult: ...


class MockExpenseFlowProvider(ExpenseFlowProvider):
    """确定性 Mock 提交：恒成功，instance_id 由 batch_id 哈希派生（同批次同结果）。"""

    name = "mock"

    def submit(self, batch: ReimburseBatch) -> SubmissionResult:
        digest = hashlib.sha256(batch.batch_id.encode()).hexdigest()[:12]
        return SubmissionResult(
            status="submitted",
            instance_id=f"mock-{digest}",
            provider=self.name,
            message=(
                f"Mock 提交成功：{batch.title}（{len(batch.items)} 张，"
                f"合计 ¥{batch.total_amount:,.2f}）"
            ),
            submitted_at=datetime.now(),
        )


def get_expense_flow_provider() -> ExpenseFlowProvider:
    provider = get_settings().expense_flow_provider
    if provider == "mock":
        return MockExpenseFlowProvider()
    if provider == "feishu":
        from invoice_agent.services import feishu_provider  # 未来新增，接入零改动

        return feishu_provider.FeishuExpenseFlowProvider()
    raise ValueError(f"未知报销流程提供方：{provider}（当前支持 mock）")
