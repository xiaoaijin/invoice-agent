"""verify 节点：调用验真 Provider。"""

from __future__ import annotations

from invoice_agent.config import get_app_config
from invoice_agent.graph.state import AgentState
from invoice_agent.services.dedup_store import DedupStore
from invoice_agent.services.verify_provider import get_verify_provider


def verify(state: AgentState) -> dict:
    invoice = state.get("invoice")
    if invoice is None:
        return {}
    store = DedupStore(get_app_config().db_path)
    result = get_verify_provider(store).verify(invoice)
    return {"verification": result}
