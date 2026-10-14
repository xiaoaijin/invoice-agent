"""政策语料加载：md / pdf / docx → 统一 markdown 文本。"""

from __future__ import annotations

from pathlib import Path


def load_documents(knowledge_dir: Path) -> list[dict]:
    """返回 [{"source": 文件名, "text": 全文}]。"""
    docs: list[dict] = []
    for p in sorted(knowledge_dir.rglob("*")):
        if p.suffix.lower() == ".md":
            docs.append({"source": p.name, "text": p.read_text(encoding="utf-8")})
        elif p.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(str(p)) as pdf:
                text = "\n".join(page.get_text() for page in pdf)
            docs.append({"source": p.name, "text": text})
        elif p.suffix.lower() == ".docx":
            import docx

            d = docx.Document(str(p))
            text = "\n".join(para.text for para in d.paragraphs)
            docs.append({"source": p.name, "text": text})
    return docs
