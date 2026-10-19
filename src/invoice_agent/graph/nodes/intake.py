"""intake 节点：读取文件、判断类型、PDF 逐页转 base64 图片。

识票前统一做图像预处理（缩放 + JPEG 重编码）压缩 vision 输入 token；
只改内存中的 page_images，不动磁盘原文件——dedup 的 file_sha256 语义不受影响。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pymupdf
from PIL import Image

from invoice_agent.config import get_app_config
from invoice_agent.graph.state import AgentState

SUPPORTED = {".png", ".jpg", ".jpeg", ".pdf"}  # config 未配置时的兜底白名单


def _to_data_url(img_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"


def _resize_encode(img: Image.Image) -> tuple[bytes, str]:
    """识票前预处理：长边超限等比缩放，统一转 RGB 后 JPEG 编码（RGBA/L/P 兼容）。"""
    cfg = get_app_config().upload
    max_edge = cfg.image_max_long_edge_px
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=cfg.image_jpeg_quality)
    return buf.getvalue(), "image/jpeg"


def intake(state: AgentState) -> dict:
    file_path = state.get("file_path")
    if not file_path:
        return {}
    p = Path(file_path)
    if not p.exists():
        return {"extract_error": f"文件不存在：{file_path}"}
    cfg = get_app_config().upload
    allowed = set(cfg.allowed_extensions) or SUPPORTED
    suffix = p.suffix.lower()
    if suffix not in allowed:
        return {"extract_error": f"不支持的文件类型：{suffix}（支持 {sorted(allowed)}）"}
    max_bytes = cfg.max_size_mb * 1024 * 1024
    if p.stat().st_size > max_bytes:
        return {"extract_error": f"文件超过大小上限：{p.stat().st_size} > {max_bytes} 字节"}

    if suffix == ".pdf":
        page_images: list[str] = []
        truncated = 0
        with pymupdf.open(str(p)) as doc:
            for i, page in enumerate(doc):
                if i >= cfg.max_pdf_pages:
                    truncated = len(doc) - cfg.max_pdf_pages
                    break
                pix = page.get_pixmap(dpi=cfg.pdf_render_dpi)
                with Image.open(io.BytesIO(pix.tobytes("png"))) as page_img:
                    data, mime = _resize_encode(page_img)
                page_images.append(_to_data_url(data, mime))
        if not page_images:
            return {"extract_error": "PDF 无内容页"}
        result: dict = {"file_type": "pdf", "page_images": page_images}
        if truncated:
            result["pages_truncated"] = truncated
        return result

    with Image.open(p) as img:
        data, mime = _resize_encode(img)
    return {
        "file_type": suffix.lstrip("."),
        "page_images": [_to_data_url(data, mime)],
    }
