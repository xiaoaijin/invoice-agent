"""带引用的政策问答 prompt 约束。"""

QA_SYSTEM_PROMPT = """你是公司报销政策与税法助手。请基于提供的参考条款回答用户问题。

硬性要求：
1. 回答中的每个政策结论必须标注 [来源N]，文末列出 [N] 《文档名》第X条：原文摘录。
2. 引用的条款编号必须来自参考条款自带的元数据，禁止自己编造条款编号。
3. 参考条款不足以支撑结论时，明确回答"现有知识库中未找到相关条款，请咨询财务部"，禁止编造。
4. 回答使用简体中文，简洁分点。"""


def format_chunks_for_prompt(chunks) -> str:
    """把检索结果格式化为参考条款文本（含元数据编号）。"""
    if not chunks:
        return "（未检索到相关条款）"
    parts = []
    for i, c in enumerate(chunks, 1):
        article = f"第{c.article_no}" if c.article_no and not c.article_no.startswith("第") else (c.article_no or "")
        head = f"[来源{i}] 《{c.source}》{c.doc_title}{article}"
        parts.append(f"{head}\n{c.text}")
    return "\n\n---\n\n".join(parts)
