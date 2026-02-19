from __future__ import annotations

from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.nodes.aggregate_report import aggregate_report_node
from app.nodes.discover_urls import discover_urls_node
from app.nodes.fetch_sources import fetch_sources_node
from app.nodes.index_sources import index_sources_node
from app.nodes.score_kpis import score_kpis_node


class OrchestratorState(TypedDict, total=False):
    run_id: str
    company_name: str
    company_domain: str
    target_urls: List[str]
    url_count: int
    sources: List[Dict]
    collection_id: str
    kpi_results: List[Dict]
    missing_evidence: List[str]
    report_path: str
    overall_score: float


def build_orchestrator_graph():
    builder = StateGraph(OrchestratorState)
    builder.add_node("discover_urls", discover_urls_node)
    builder.add_node("fetch_sources", fetch_sources_node)
    builder.add_node("index_sources", index_sources_node)
    builder.add_node("score_kpis", score_kpis_node)
    builder.add_node("aggregate_report", aggregate_report_node)

    builder.set_entry_point("discover_urls")
    builder.add_edge("discover_urls", "fetch_sources")
    builder.add_edge("fetch_sources", "index_sources")
    builder.add_edge("index_sources", "score_kpis")
    builder.add_edge("score_kpis", "aggregate_report")
    builder.add_edge("aggregate_report", END)

    return builder.compile()
