"""政策语料切分：按 Markdown 标题 / 「第X条」切，条款为最小引用单元。

政策问答的引用粒度天然是「条」，按字符硬切会把条款切碎导致引用残缺；
单块超过 ~600 字再用带 overlap 的长度切分兜底。
"""

from __future__ import annotations

import re

MAX_CHUNK = 600
OVERLAP = 50

# 匹配「第一章」「第二十七条」「第 12 条」等
ARTICLE_RE = re.compile(r"^(#{1,4}\s*)?第[一二三四五六七八九十百零\d]+[章条款]")
HEADING_RE = re.compile(r"^#{1,6}\s+")


def split_documents(docs: list[dict]) -> list[dict]:
    """返回 [{"text", "source", "layer", "doc_title", "article_no"}]。"""
    chunks: list[dict] = []
    for doc in docs:
        lines = doc["text"].splitlines()
        # 按标题/条款边界分组
        blocks: list[tuple[str, list[str]]] = []  # (boundary_line, body_lines)
        current_title = doc["source"]
        buf: list[str] = []
        for line in lines:
            stripped = line.strip()
            m_article = ARTICLE_RE.match(stripped)
            is_article = bool(m_article) and m_article.group(0)[-1] in "条款"
            if HEADING_RE.match(stripped) or m_article:
                if any(s.strip() for s in buf):
                    blocks.append((current_title, buf))
                if HEADING_RE.match(stripped) and not is_article:
                    # 章/节标题行 → 更新 doc_title
                    current_title = stripped.lstrip("#").strip()
                # 条款行（第X条/款）不更新 doc_title：条款编号已提取进 article_no
                # 元数据，doc_title 保持所属章节，避免引用出现「第五条第五条」重复
                buf = [line]
            else:
                buf.append(line)
        if any(s.strip() for s in buf):
            blocks.append((current_title, buf))

        for title, body_lines in blocks:
            text = "\n".join(body_lines).strip()
            if not text:
                continue
            m = re.search(r"第[一二三四五六七八九十百零\d]+[条款]", text)
            article_no = m.group(0) if m else None
            for piece in _split_long(text):
                chunks.append(
                    {
                        "text": piece,
                        "source": doc["source"],
                        "layer": doc.get("layer", ""),
                        "doc_title": title,
                        "article_no": article_no,
                    }
                )
    return chunks


def _split_long(text: str) -> list[str]:
    """长度兜底切分（带 overlap），仅对超长块生效。"""
    if len(text) <= MAX_CHUNK:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + MAX_CHUNK])
        start += MAX_CHUNK - OVERLAP
    return pieces
