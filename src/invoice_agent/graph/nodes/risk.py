"""risk 节点：规则引擎。"""

from __future__ import annotations

from invoice_agent.config import get_app_config
from invoice_agent.graph.state import AgentState
from invoice_agent.rules.risk_rules import run_risk_rules
from invoice_agent.services.dedup_store import DedupStore


def risk(state: AgentState) -> dict:
    invoice = state.get("invoice")
    if invoice is None:
        return {}
    ctx = {
        "dedup_store": DedupStore(get_app_config().db_path),
        "duplicate_hits": state.get("duplicate_hits") or [],
    }
    findings = run_risk_rules(invoice, ctx, get_app_config())
    return {"risk_findings": findings}
