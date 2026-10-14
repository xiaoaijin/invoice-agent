"""intake 节点：读取文件、判断类型、PDF 逐页转 base64 图片。"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pymupdf
from PIL import Image

from invoice_agent.graph.state import AgentState

SUPPORTED = {".png", ".jpg", ".jpeg", ".pdf"}


def _to_data_url(img_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"


def intake(state: AgentState) -> dict:
    file_path = state.get("file_path")
    if not file_path:
        return {}
    p = Path(file_path)
    if not p.exists():
        return {"extract_error": f"文件不存在：{file_path}"}
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        return {"extract_error": f"不支持的文件类型：{suffix}（支持 {sorted(SUPPORTED)}）"}

    if suffix == ".pdf":
        page_images: list[str] = []
        with pymupdf.open(str(p)) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                page_images.append(_to_data_url(pix.tobytes("png"), "image/png"))
        if not page_images:
            return {"extract_error": "PDF 无内容页"}
        return {"file_type": "pdf", "page_images": page_images}

    img = Image.open(p)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "file_type": suffix.lstrip("."),
        "page_images": [_to_data_url(buf.getvalue(), "image/png")],
    }
