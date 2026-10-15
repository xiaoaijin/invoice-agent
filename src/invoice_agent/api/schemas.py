"""API 请求/响应模型：只服务 HTTP 边界，与 models/ 的领域模型分离。

序列化约定：响应中的 pydantic 领域对象统一 model_dump(mode="json")（date 等自动转 str）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None  # 缺省时服务端生成并在响应回传


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None  # 同一 session_id 多轮共享记忆


class AskResponse(BaseModel):
    session_id: str
    answer: str
    run_id: str | None = None


class ChatResponse(AskResponse):
    pass


class InvoiceResponse(BaseModel):
    session_id: str
    run_id: str | None = None
    answer: str | None = None
    extract_error: str | None = None
    pages_truncated: int = 0
    invoice: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    duplicate_hits: list[dict[str, Any]] = Field(default_factory=list)
    risk_findings: list[dict[str, Any]] = Field(default_factory=list)


class RunDetailResponse(BaseModel):
    run: dict[str, Any] | None = None
    spans: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
