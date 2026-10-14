"""构建/重建 RAG 知识库索引（需 DASHSCOPE_API_KEY）。

用法：uv run python scripts/build_knowledge_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from invoice_agent.config import get_app_config, require_api_key
from invoice_agent.services.rag.indexer import build_index


def main() -> int:
    require_api_key()
    cfg = get_app_config()
    from invoice_agent.services.llm import get_embedding_model

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return get_embedding_model().embed_documents(texts)

    n = build_index(cfg.knowledge_dir, embed_fn)
    print(f"索引构建完成：{n} 个知识块 → PostgreSQL (pgvector)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
