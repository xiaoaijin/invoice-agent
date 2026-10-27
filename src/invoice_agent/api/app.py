"""FastAPI 应用工厂与路由。

错误处理约定（业务节点内部已静默降级，故）：
- LLM/PG/检索失败 → 200（answer 内含降级提示）
- 文件超大小上限 → 413；扩展名不在白名单 → 415；请求体校验失败 → 422
- run 不存在 → 404；待确认批次不存在 → 404、已过期 → 410
- 服务未配置（缺 API key 等 RuntimeError）→ 503；未捕获 → 500

报销提交为两步：preview 落服务端待确认批次，人工审核后 confirm 才真正
提交审批流并写台账（批次内容以服务端存储为准，防篡改）。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from invoice_agent.api import deps
from invoice_agent.api.schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    InvoiceResponse,
    ReimburseConfirmRequest,
    ReimburseConfirmResponse,
    ReimbursePreviewResponse,
    RunDetailResponse,
)
from invoice_agent.config import get_app_config
from invoice_agent.monitoring.setup import setup_logging
from invoice_agent.runner import iter_run, run_stream
from invoice_agent.services.reimburse_flow import (
    PendingExpiredError,
    PendingNotFoundError,
    confirm_pending_batch,
    run_batch,
    save_pending,
)

DEFAULT_INVOICE_QUESTION = "这张发票能否报销？有什么风险？"
MAX_BATCH_FILES = 50  # 单次批量报销的文件数上限（防误传与资源耗尽）


def _dump(obj):
    """pydantic 领域对象 → JSON 可序列化 dict，其余原样。"""
    return obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="invoice-agent API",
        description="发票助手 Agent：识票→验真→查重→风控→政策问答",
        version="0.1.0",
    )

    @app.exception_handler(RuntimeError)
    async def _runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        # require_api_key 等服务未配置场景 → 503（区别于代码缺陷的 500）
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": f"内部错误：{exc}"}, status_code=500)

    @app.get("/healthz")
    def healthz() -> dict:
        cfg = get_app_config()  # 配置缺失会抛 FileNotFoundError → 500
        data_dir = Path(cfg.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".healthz"
        probe.write_text("ok")
        probe.unlink()
        return {"status": "ok"}

    def _save_upload(file: UploadFile) -> Path:
        """校验并落盘上传文件（/invoices 与 /reimbursements 共用）。

        扩展名不在白名单 → 415；超大小上限 → 413。
        """
        cfg = get_app_config().upload
        fname = Path(file.filename or "upload").name  # 防路径穿越
        suffix = Path(fname).suffix.lower()
        if suffix not in cfg.allowed_extensions:
            raise HTTPException(415, f"不支持的文件类型：{suffix}（支持 {cfg.allowed_extensions}）")
        max_bytes = cfg.max_size_mb * 1024 * 1024
        chunks: list[bytes] = []
        size = 0
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"文件超过大小上限：{cfg.max_size_mb}MB")
            chunks.append(chunk)
        dest = deps.uploads_dir() / f"{uuid.uuid4().hex[:8]}_{fname}"
        dest.write_bytes(b"".join(chunks))
        return dest

    @app.post("/invoices", response_model=InvoiceResponse)
    def submit_invoice(
        file: UploadFile = File(...),  # noqa: B008 — FastAPI 依赖注入惯用法
        question: str = Form(default=DEFAULT_INVOICE_QUESTION),
        session_id: str = Form(default=None),
    ) -> InvoiceResponse:
        dest = _save_upload(file)
        sid = session_id or uuid.uuid4().hex
        state, handler = run_stream(str(dest), question, sid)
        return InvoiceResponse(
            session_id=sid,
            run_id=handler.run_id if handler else None,
            answer=state.get("answer"),
            extract_error=state.get("extract_error"),
            pages_truncated=state.get("pages_truncated", 0),
            invoice=_dump(state.get("invoice")),
            verification=_dump(state.get("verification")),
            duplicate_hits=[_dump(d) for d in state.get("duplicate_hits") or []],
            risk_findings=[_dump(f) for f in state.get("risk_findings") or []],
        )

    @app.post("/reimbursements/preview", response_model=ReimbursePreviewResponse)
    def preview_reimbursements(
        files: list[UploadFile] = File(...),  # noqa: B008 — FastAPI 依赖注入惯用法
        title: str = Form(default=None),
    ) -> ReimbursePreviewResponse:
        """多文件批量识别并分类统计，落服务端待确认批次（不提交、不写台账）。

        单张失败不中断，见 items[].error；人工审核后凭 batch_id 调 confirm 提交。
        """
        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(413, f"单次最多 {MAX_BATCH_FILES} 个文件")
        paths = [_save_upload(f) for f in files]
        batch = run_batch(paths, title=title or None)
        save_pending(batch, store=deps.get_dedup_store())
        return ReimbursePreviewResponse(
            batch_id=batch.batch_id, batch=batch.model_dump(mode="json")
        )

    @app.get("/reimbursements/pending")
    def list_pending_reimbursements() -> dict:
        """待确认批次列表（跨重启可查，人工审核入口）。"""
        return {"pending": deps.get_dedup_store().list_pending_batches()}

    @app.post("/reimbursements/{batch_id}/confirm", response_model=ReimburseConfirmResponse)
    def confirm_reimbursement(batch_id: str, req: ReimburseConfirmRequest) -> ReimburseConfirmResponse:
        """人工确认后提交审批流程并写台账（去重即刻生效）。

        批次内容只认服务端待确认存储（防篡改）；exclude_numbers 可剔除个别
        发票（服务端重算合计）。provider 提交失败批次保留可重试。
        """
        try:
            batch, result = confirm_pending_batch(
                batch_id,
                exclude_numbers=set(req.exclude_numbers),
                store=deps.get_dedup_store(),
            )
        except PendingNotFoundError as exc:
            raise HTTPException(404, f"{exc}（请先 POST /reimbursements/preview）") from exc
        except PendingExpiredError as exc:
            raise HTTPException(410, f"{exc}") from exc
        return ReimburseConfirmResponse(
            batch_id=batch.batch_id, submission=result.model_dump(mode="json")
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        sid = req.session_id or uuid.uuid4().hex
        state, handler = run_stream(None, req.question, sid)
        return AskResponse(
            session_id=sid,
            answer=state.get("answer") or "",
            run_id=handler.run_id if handler else None,
        )

    def _chat_turn(sid: str, message: str, *, on_token=None) -> tuple[dict, object | None]:
        """跑一轮对话并落库 ConversationStore（与 UI 的 data/chat_history.db 互通）。"""
        state, handler = run_stream(
            None, message, sid, on_token=on_token
        )
        store = deps.get_conversation_store()
        entries = store.get_entries(sid)
        entries.append({"role": "user", "text": message})
        entries.append(
            {
                "role": "assistant",
                "kind": "qa",
                "answer": state.get("answer"),
                "rag_chunks": [_dump(c) for c in state.get("rag_chunks") or []],
            }
        )
        # 会话标题取首条用户输入（与 UI _conv_title 约定一致）；entries[0] 恒为用户条目
        store.save(sid, (entries[0].get("text") or "对话")[:18], entries)
        return state, handler

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        sid = req.session_id or uuid.uuid4().hex
        state, handler = _chat_turn(sid, req.message)
        return ChatResponse(
            session_id=sid,
            answer=state.get("answer") or "",
            run_id=handler.run_id if handler else None,
        )

    @app.get("/chat/{session_id}/history")
    def chat_history(session_id: str) -> dict:
        store = deps.get_conversation_store()
        entries = store.get_entries(session_id)
        return {"session_id": session_id, "entries": [_dump(e) for e in entries]}

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        """SSE 流式对话：node/token/done 三类事件，data: {json}\\n\\n。"""
        sid = req.session_id or uuid.uuid4().hex

        def gen():
            final: dict = {}
            handler = None
            for event, payload, h in iter_run(None, req.message, sid):
                handler = h
                if event == "node":
                    yield f'data: {json.dumps({"type": "node", "node": payload}, ensure_ascii=False)}\n\n'
                elif event == "token":
                    yield f'data: {json.dumps({"type": "token", "text": payload}, ensure_ascii=False)}\n\n'
                elif event == "done":
                    final = payload
            # 与非流式 /chat 一致：落库会话历史
            store = deps.get_conversation_store()
            entries = store.get_entries(sid)
            entries.append({"role": "user", "text": req.message})
            entries.append(
                {
                    "role": "assistant",
                    "kind": "qa",
                    "answer": final.get("answer"),
                    "rag_chunks": [_dump(c) for c in final.get("rag_chunks") or []],
                }
            )
            store.save(sid, req.message[:18], entries)
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "session_id": sid,
                        "answer": final.get("answer"),
                        "run_id": handler.run_id if handler else None,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/runs/{run_id}", response_model=RunDetailResponse)
    def run_detail(run_id: str) -> RunDetailResponse:
        detail = deps.get_monitor_store().run_detail(run_id)
        if detail["run"] is None:
            raise HTTPException(404, f"run 不存在：{run_id}")
        return RunDetailResponse(**detail)

    return app
