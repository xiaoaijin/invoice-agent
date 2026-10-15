"""intake 节点测试：大小上限、页数截断（防大 PDF 全页转 base64 爆内存）。"""

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from invoice_agent.config import AppConfig, CompanyConfig, PathsConfig, RiskConfig, UploadConfig
from invoice_agent.graph.nodes import intake as intake_mod
from invoice_agent.graph.nodes.intake import intake


@pytest.fixture
def _cfg(monkeypatch):
    fake_cfg = AppConfig(
        company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
        risk=RiskConfig(),
        paths=PathsConfig(db="data/invoice_agent.db", knowledge_dir="knowledge"),
        upload=UploadConfig(max_size_mb=1, max_pdf_pages=5),
    )
    monkeypatch.setattr(intake_mod, "get_app_config", lambda: fake_cfg)


def _make_pdf(path: Path, pages: int) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    for i in range(pages):
        c.drawString(100, 700, f"page {i + 1}")
        c.showPage()
    c.save()


def test_pdf_pages_truncated(tmp_path: Path, _cfg):
    """7 页 PDF 只取前 5 页，pages_truncated=2。"""
    p = tmp_path / "multi.pdf"
    _make_pdf(p, 7)
    result = intake({"file_path": str(p)})
    assert result["file_type"] == "pdf"
    assert len(result["page_images"]) == 5
    assert result["pages_truncated"] == 2
    assert all(u.startswith("data:image/png;base64,") for u in result["page_images"])


def test_pdf_within_limit_no_truncation(tmp_path: Path, _cfg):
    p = tmp_path / "small.pdf"
    _make_pdf(p, 3)
    result = intake({"file_path": str(p)})
    assert len(result["page_images"]) == 3
    assert "pages_truncated" not in result


def test_oversize_file_rejected(tmp_path: Path, _cfg):
    """超过 max_size_mb 直接报 extract_error（不读入内存）。"""
    p = tmp_path / "big.png"
    p.write_bytes(b"\x89PNG" + b"0" * (1024 * 1024 + 1))
    result = intake({"file_path": str(p)})
    assert "page_images" not in result
    assert "大小上限" in result["extract_error"]


def test_disallowed_extension_rejected(tmp_path: Path, _cfg):
    p = tmp_path / "evil.exe"
    p.write_bytes(b"MZ")
    result = intake({"file_path": str(p)})
    assert "不支持的文件类型" in result["extract_error"]


def test_missing_file(tmp_path: Path, _cfg):
    result = intake({"file_path": str(tmp_path / "nope.png")})
    assert "文件不存在" in result["extract_error"]
