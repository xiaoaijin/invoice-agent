"""AgentState：LangGraph 全局状态定义。"""

from __future__ import annotations

from typing import TypedDict

from invoice_agent.models.invoice import InvoiceData
from invoice_agent.models.risk import DuplicateHit, Finding, RetrievedChunk, VerificationResult


class AgentState(TypedDict, total=False):
    # 输入
    session_id: str
    file_path: str | None  # 上传的发票文件（图/PDF），可为空
    user_question: str | None  # 用户提问，可为空

    # 识票
    file_type: str  # png/jpg/jpeg/pdf
    page_images: list[str]  # base64 data URL（PDF 拆页后）
    invoice: InvoiceData | None
    extract_error: str | None
    recognize_retry: int  # 提取重试计数（上限 2）

    # 验真 / 去重 / 风控
    verification: VerificationResult | None
    duplicate_hits: list[DuplicateHit]
    risk_findings: list[Finding]

    # RAG
    rag_chunks: list[RetrievedChunk]

    # 自反馈（回答自评回环 / 风控复核）
    critique: dict | None  # CritiqueResult.model_dump()（回答自评结果）
    answer_retry: int  # 答案重生成计数（上限 feedback.answer.max_retries）
    critic_feedback: str | None  # 自评不通过时传给 finalize 的改进提示
    risk_review: dict | None  # RiskReview.model_dump()（风控复核：removed/added/summary）

    # 输出
    answer: str | None  # 最终回答（Markdown）

    # 多轮对话记忆（checkpointer 按 thread_id 持久化整个 state）
    chat_history: list[dict]  # [{"role": "user"|"tool"|"assistant", "content": str}]
    history_summary: str  # 压缩产生的早期对话滚动摘要
