"""索引：嵌入写入 Chroma + BM25 语料落盘 JSON。

BM25 语料（chunk 全文+元数据）存 data/chroma 同级 JSON，进程启动时内存构建。
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from invoice_agent.services.rag.chunking import split_documents
from invoice_agent.services.rag.loaders import load_documents

CORPUS_FILENAME = "bm25_corpus.json"


def build_index(knowledge_dir: Path, chroma_dir: Path, embed_fn) -> int:
    """构建向量索引 + BM25 语料。embed_fn: list[str] -> list[list[float]]。"""
    docs = load_documents(knowledge_dir)
    chunks = split_documents(docs)
    if not chunks:
        raise RuntimeError(f"知识库为空：{knowledge_dir}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    # 重建：避免旧向量污染
    for name in client.list_collections():
        if name.name == "knowledge":
            client.delete_collection("knowledge")
    col = client.get_or_create_collection("knowledge", metadata={"hnsw:space": "cosine"})

    texts = [c["text"] for c in chunks]
    embeddings = embed_fn(texts)
    col.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": c["source"],
                "doc_title": c["doc_title"],
                "article_no": c["article_no"] or "",
            }
            for c in chunks
        ],
    )

    (chroma_dir / CORPUS_FILENAME).write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    return len(chunks)
