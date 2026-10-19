"""intake 节点测试：大小上限、页数截断（防大 PDF 全页转 base64 爆内存）、识票前图像预处理。"""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image
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


def _decode_page(data_url: str) -> Image.Image:
    """从 data URL 解码出 PIL Image（预处理断言用）。"""
    payload = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _make_png(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
    # 纯色图压缩后极小，不会触发大小上限
    Image.new(mode, size, (200, 30, 40)).save(path, format="PNG")


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
    assert all(u.startswith("data:image/jpeg;base64,") for u in result["page_images"])


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


class TestImagePreprocess:
    """识票前预处理：长边缩放 + 统一 JPEG 重编码（压缩 vision 输入 token）。"""

    def test_oversize_image_resized_to_long_edge(self, tmp_path: Path, _cfg):
        p = tmp_path / "wide.png"
        _make_png(p, (3000, 1000))
        result = intake({"file_path": str(p)})
        assert result["page_images"][0].startswith("data:image/jpeg;base64,")
        img = _decode_page(result["page_images"][0])
        assert max(img.size) == 2200  # 长边收敛到 image_max_long_edge_px
        assert img.size == (2200, 733)  # 等比缩放
        assert img.mode == "RGB"

    def test_small_image_keeps_size(self, tmp_path: Path, _cfg):
        """小于阈值只做 JPEG 重编码，不缩放。"""
        p = tmp_path / "small.png"
        _make_png(p, (800, 600))
        result = intake({"file_path": str(p)})
        img = _decode_page(result["page_images"][0])
        assert img.size == (800, 600)
        assert result["page_images"][0].startswith("data:image/jpeg;base64,")

    def test_rgba_converted_to_rgb(self, tmp_path: Path, _cfg):
        """RGBA（透明通道）JPEG 编码前转 RGB，不报错。"""
        p = tmp_path / "alpha.png"
        _make_png(p, (100, 80), mode="RGBA")
        result = intake({"file_path": str(p)})
        assert _decode_page(result["page_images"][0]).mode == "RGB"

    def test_pdf_render_dpi_from_config(self, tmp_path: Path, monkeypatch):
        """PDF 渲染 DPI 取自 config（72dpi 下 A4 宽 595pt → 595px）。"""
        fake_cfg = AppConfig(
            company=CompanyConfig(name="示例科技有限公司", tax_id="91330100MA27XXXXXX"),
            risk=RiskConfig(),
            paths=PathsConfig(db="data/invoice_agent.db", knowledge_dir="knowledge"),
            upload=UploadConfig(max_size_mb=10, max_pdf_pages=5, pdf_render_dpi=72),
        )
        monkeypatch.setattr(intake_mod, "get_app_config", lambda: fake_cfg)
        p = tmp_path / "one.pdf"
        _make_pdf(p, 1)
        result = intake({"file_path": str(p)})
        img = _decode_page(result["page_images"][0])
        assert abs(img.size[0] - 595) <= 2
