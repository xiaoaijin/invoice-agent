"""dedup 节点：查重（只查不写；用户确认报销后才写入）。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from invoice_agent.config import get_app_config
from invoice_agent.graph.state import AgentState
from invoice_agent.services.dedup_store import DedupStore


def file_sha256(path: str | None) -> str:
    if not path or not Path(path).exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dedup(state: AgentState) -> dict:
    invoice = state.get("invoice")
    if invoice is None:
        return {}
    store = DedupStore(get_app_config().db_path)
    sha = file_sha256(state.get("file_path"))
    hits = store.find_duplicates(invoice.invoice_number, invoice.seller_tax_id, sha)
    return {"duplicate_hits": hits}
