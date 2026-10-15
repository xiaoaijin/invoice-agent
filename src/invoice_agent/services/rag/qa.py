"""带引用的政策问答 prompt 约束。"""

LAYER_LABELS = {
    "national": "国家法规",
    "local": "地方细则",
    "company": "公司制度",
}


def format_chunks_for_prompt(chunks) -> str:
    """把检索结果格式化为参考条款文本（含层级与元数据编号）。"""
    if not chunks:
        return "（未检索到相关条款）"
    parts = []
    for i, c in enumerate(chunks, 1):
        article = (
            f"第{c.article_no}"
            if c.article_no and not c.article_no.startswith("第")
            else (c.article_no or "")
        )
        if article and article in c.doc_title:  # 标题已含条款号，避免「第五条第五条」
            article = ""
        layer = LAYER_LABELS.get(getattr(c, "layer", ""), "")
        tag = f"【{layer}】" if layer else ""
        loc = f"{c.doc_title} {article}".strip()
        head = f"[来源{i}]{tag}《{c.source}》{loc}"
        parts.append(f"{head}\n{c.text}")
    return "\n\n---\n\n".join(parts)
