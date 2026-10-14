"""政策语料加载：md / pdf / docx → 统一 markdown 文本。

三层知识库：national（国家法规）/ local（地方细则）/ company（公司制度），
层级由 knowledge_dir 下的一级子目录名决定。
"""

from __future__ import annotations

from pathlib import Path


def _layer_of(path: Path, knowledge_dir: Path) -> str:
    """文件相对 knowledge_dir 的一级子目录名作为层级；根目录散落文件无层级。"""
    rel = path.relative_to(knowledge_dir)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def load_documents(knowledge_dir: Path) -> list[dict]:
    """返回 [{"source": 文件名, "layer": 层级, "text": 全文}]。"""
    docs: list[dict] = []
    for p in sorted(knowledge_dir.rglob("*")):
        if p.suffix.lower() in {".md", ".pdf", ".docx"}:
            if p.suffix.lower() == ".md":
                text = p.read_text(encoding="utf-8")
            elif p.suffix.lower() == ".pdf":
                import pymupdf

                with pymupdf.open(str(p)) as pdf:
                    text = "\n".join(page.get_text() for page in pdf)
            else:
                import docx

                d = docx.Document(str(p))
                text = "\n".join(para.text for para in d.paragraphs)
            docs.append(
                {"source": p.name, "layer": _layer_of(p, knowledge_dir), "text": text}
            )
    return docs
