"""FastAPI 服务层测试：全部离线（节点打桩 + 临时目录配置）。

复用 e2e 测试模式；区别在于 patch 的是 config.load_app_config（get_app_config
是 lru_cache 包装，调用时才解析 load_app_config，改它可全局生效，免去逐模块
patch get_app_config）。
"""

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invoice_agent.api import deps as api_deps
from invoice_agent.api.app import create_app
from invoice_agent.config import AppConfig, CompanyConfig, PathsConfig, RiskConfig
from invoice_agent.graph import builder as builder_mod
from invoice_agent.models.invoice import InvoiceData, InvoiceTypeEnum


def make_invoice(**kw) -> InvoiceData:
    base = dict(
        invoice_type=InvoiceTypeEnum.SPECIAL,
        invoice_code="044032000111",
        invoice_number="50000001",
        invoice_date=date(2026, 10, 1),
        buyer_name="示例科技有限公司",
        buyer_tax_id="91330100MA27XXXXXX",
        seller_name="卖方公司",
        seller_tax_id="91330100MA28YYYYYY",
        amount_without_tax=100.0,
        tax_amount=13.0,
        total_amount=113.0,
        items=[],
    )
    base.update(kw)
    return InvoiceData(**base)


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """打桩全链路的 TestClient：config 指向 tmp_path，LLM/检索全短路。"""
    fake_cfg = AppConfig(
        company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
        risk=RiskConfig(),
        paths=PathsConfig(
            db=str(tmp_path / "data" / "invoice_agent.db"),
            knowledge_dir=str(tmp_path / "knowledge"),
        ),
    )
    from invoice_agent import config as config_mod
    from invoice_agent.graph import nodes
    from invoice_agent.graph.nodes import critic

    monkeypatch.setattr(config_mod, "load_app_config", lambda path=None: fake_cfg)
    config_mod.get_app_config.cache_clear()

    # 自反馈短路（critic 会调真实 LLM）
    monkeypatch.setattr(
        critic, "critic_answer", lambda state: {"critique": {"passed": True}, "critic_feedback": None}
    )
    monkeypatch.setattr(critic, "risk_review", lambda state: {})

    # recognize 返回固定发票；检索返回空；finalize 输出确定性文本
    monkeypatch.setattr(
        nodes.recognize,
        "recognize",
        lambda state: {"invoice": make_invoice(), "extract_error": None, "recognize_retry": 0},
    )
    import invoice_agent.graph.nodes.answer as answer_mod

    monkeypatch.setattr(answer_mod, "retrieve_docs", lambda state: {"rag_chunks": []})
    captured: dict = {}

    def fake_finalize(state):
        captured["history_len"] = len(state.get("chat_history") or [])
        # 与真实 finalize 一致：把本轮问答写回 chat_history（checkpointer 据此跨轮记忆）
        return {
            "answer": f"OK q={state.get('user_question')}",
            "chat_history": list(state.get("chat_history") or [])
            + [
                {"role": "user", "content": state.get("user_question")},
                {"role": "assistant", "content": f"OK q={state.get('user_question')}"},
            ],
        }

    monkeypatch.setattr(nodes.answer, "finalize", fake_finalize)

    # 图与 store 单例全部重建，避免污染其他测试
    builder_mod.build_graph.cache_clear()
    api_deps.get_monitor_store.cache_clear()
    api_deps.get_conversation_store.cache_clear()
    yield TestClient(create_app()), captured
    config_mod.get_app_config.cache_clear()
    builder_mod.build_graph.cache_clear()
    api_deps.get_monitor_store.cache_clear()
    api_deps.get_conversation_store.cache_clear()


def _png_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return buf.getvalue()


class TestHealthz:
    def test_ok(self, client):
        c, _ = client
        resp = c.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestInvoices:
    def test_happy_path(self, client, tmp_path):
        c, _ = client
        resp = c.post(
            "/invoices",
            files={"file": ("inv.png", _png_bytes(), "image/png")},
            data={"question": "能报销吗"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["invoice"]["invoice_number"] == "50000001"
        assert body["invoice"]["invoice_date"] == "2026-10-01"  # date 自动转 str
        assert body["answer"] == "OK q=能报销吗"
        assert body["run_id"]
        assert body["extract_error"] is None
        # 文件按 UI 同款约定落盘 data/uploads/
        uploads = tmp_path / "data" / "uploads"
        assert list(uploads.glob("*_inv.png"))

    def test_oversize_413(self, client, monkeypatch):
        from invoice_agent import config as config_mod
        from invoice_agent.config import UploadConfig

        c, _ = client
        # 把上限压到极小，免得真造 10MB 文件
        orig = config_mod.load_app_config
        monkeypatch.setattr(
            config_mod,
            "load_app_config",
            lambda path=None: orig().model_copy(
                update={"upload": UploadConfig(max_size_mb=0)}, deep=True
            ),
        )
        config_mod.get_app_config.cache_clear()
        resp = c.post(
            "/invoices", files={"file": ("inv.png", _png_bytes(), "image/png")}
        )
        assert resp.status_code == 413

    def test_bad_extension_415(self, client):
        c, _ = client
        resp = c.post("/invoices", files={"file": ("evil.exe", b"MZ", "application/x-exe")})
        assert resp.status_code == 415

    def test_recognize_error_still_200(self, client, monkeypatch):
        """识票失败是业务降级（节点内兜底），HTTP 层仍 200。"""
        from invoice_agent.graph import nodes

        c, _ = client
        monkeypatch.setattr(
            nodes.recognize,
            "recognize",
            lambda state: {"invoice": None, "extract_error": "无法识别", "recognize_retry": 99},
        )
        resp = c.post(
            "/invoices", files={"file": ("inv.png", _png_bytes(), "image/png")}
        )
        assert resp.status_code == 200
        assert resp.json()["extract_error"] == "无法识别"


class TestAskChat:
    def test_ask(self, client):
        c, _ = client
        resp = c.post("/ask", json={"question": "餐饮发票能抵扣吗"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "OK q=餐饮发票能抵扣吗"
        assert body["session_id"]
        assert body["run_id"]

    def test_chat_multi_turn_memory(self, client):
        """同 session_id 两轮：第二轮 finalize 能看到第一轮的 chat_history。"""
        c, captured = client
        sid = c.post("/chat", json={"message": "第一问"}).json()["session_id"]
        assert captured["history_len"] == 0
        resp = c.post("/chat", json={"message": "第二问", "session_id": sid})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid
        assert captured["history_len"] == 2  # 第一轮的 user+assistant

    def test_chat_history_endpoint(self, client):
        c, _ = client
        sid = c.post("/chat", json={"message": "记一下"}).json()["session_id"]
        resp = c.get(f"/chat/{sid}/history")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert [e["role"] for e in entries] == ["user", "assistant"]
        assert entries[0]["text"] == "记一下"

    def test_chat_stream_sse(self, client):
        c, _ = client
        resp = c.post("/chat/stream", json={"message": "流式问题"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        types = [e["type"] for e in events]
        assert types[0] == "node"
        assert types[-1] == "done"
        assert events[-1]["answer"] == "OK q=流式问题"
        # 流式对话同样落库
        sid = events[-1]["session_id"]
        hist = c.get(f"/chat/{sid}/history").json()["entries"]
        assert hist[0]["text"] == "流式问题"


class TestRuns:
    def test_404(self, client):
        c, _ = client
        assert c.get("/runs/nonexistent").status_code == 404

    def test_detail_includes_llm_calls(self, client):
        c, _ = client
        run_id = c.post("/ask", json={"question": "q"}).json()["run_id"]
        detail = c.get(f"/runs/{run_id}").json()
        assert detail["run"]["run_id"] == run_id
        assert detail["run"]["status"] == "success"
        assert isinstance(detail["spans"], list)
        assert isinstance(detail["llm_calls"], list)


class TestErrorMapping:
    def test_runtime_error_503(self, client, monkeypatch):
        """服务未配置（如缺 API key）→ 503。"""
        import invoice_agent.api.app as app_mod

        c, _ = client

        def boom(*a, **kw):
            raise RuntimeError("DASHSCOPE_API_KEY 未配置。")

        monkeypatch.setattr(app_mod, "run_stream", boom)
        resp = c.post("/ask", json={"question": "q"})
        assert resp.status_code == 503
        assert "DASHSCOPE_API_KEY" in resp.json()["detail"]
