"""带引用的政策问答 prompt 约束。"""

LAYER_LABELS = {
    "national": "国家法规",
    "local": "地方细则",
    "company": "公司制度",
}

QA_SYSTEM_PROMPT = """你是公司报销政策与税法助手。请基于提供的参考条款回答用户问题。

参考条款来自三层知识库：【国家法规】为底线；【地方细则】在不抵触国家法规的前提下补充细化；【公司制度】在不低于国家法规和地方细则要求的前提下可以更严格。

硬性要求：
1. 回答中的每个政策结论必须标注 [来源N]，文末列出 [N]【层级】《文档名》第X条：原文摘录。
2. 引用的条款编号必须来自参考条款自带的元数据，禁止自己编造条款编号。
3. 参考条款不足以支撑结论时，明确回答"现有知识库中未找到相关条款，请咨询财务部"，禁止编造。
4. 三层条款对同一事项的限额或口径不一致时，按更严格者执行，并在回答中指出冲突及各自层级。
5. 回答使用简体中文，简洁分点。"""


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
