"""风险与验真相关模型。"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    HIGH = "高"  # 疑似违规/不能报销
    MEDIUM = "中"  # 需人工确认
    LOW = "低"  # 提示性质

    @property
    def rank(self) -> int:
        return {"低": 1, "中": 2, "高": 3}[self.value]


class Finding(BaseModel):
    rule_id: str  # "R001" 抬头不符 / "R002" 重复 ...
    rule_name: str
    level: RiskLevel
    message: str  # 面向用户的中文描述
    evidence: dict = Field(default_factory=dict)  # 规则命中的原始数据，便于追溯


class VerificationResult(BaseModel):
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    provider: str  # "mock" / "baiwang" / ...
    message: str
    checked_at: datetime = Field(default_factory=datetime.now)


class DuplicateHit(BaseModel):
    invoice_number: str
    invoice_date: date
    total_amount: float
    first_submitted_at: datetime
    status: str  # "submitted" / "reimbursed" / "rejected"


class RetrievedChunk(BaseModel):
    text: str
    source: str  # 文件名
    layer: str = ""  # 三层知识库层级：national（国家法规）/ local（地方细则）/ company（公司制度）
    doc_title: str  # 章节/文档标题
    article_no: str | None = None  # "第二十七条"
    score: float = 0.0


class RiskReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)

    @property
    def max_level(self) -> RiskLevel | None:
        return max((f.level for f in self.findings), key=lambda lv: lv.rank, default=None)

    @property
    def is_clean(self) -> bool:
        return not self.findings
