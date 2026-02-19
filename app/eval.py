from __future__ import annotations

from statistics import pstdev
from typing import Dict, List

from app.kpi_catalog import load_kpi_catalog
from app.nodes.analyze import analyze_node


EXPECTED_PAGE_TYPES = ["home", "about", "product", "pricing", "careers"]


def coverage_ratio(sources: List[Dict]) -> float:
    found = {source["page_type"] for source in sources}
    covered = sum(1 for page_type in EXPECTED_PAGE_TYPES if page_type in found)
    return covered / len(EXPECTED_PAGE_TYPES)


def empty_retrieval_rate(retrieval_stats: Dict[str, int]) -> float:
    total = retrieval_stats.get("total_kpis", 0)
    if total == 0:
        return 0.0
    return retrieval_stats.get("empty_retrievals", 0) / total


def groundedness_ratio(kpi_results: List[Dict]) -> float:
    if not kpi_results:
        return 0.0
    grounded = 0
    for result in kpi_results:
        citations = result.get("citations", [])
        if citations and all(cite.get("quote") for cite in citations):
            grounded += 1
    return grounded / len(kpi_results)


def score_stability_std(state: Dict, sample_size: int = 3, reruns: int = 3) -> float:
    kpis = load_kpi_catalog()[:sample_size]
    scores_by_kpi = {kpi.kpi_id: [] for kpi in kpis}

    for idx in range(reruns):
        rerun_state = dict(state)
        rerun_state["run_seed"] = f"{state['run_id']}_rerun_{idx}"
        rerun_results = analyze_node(rerun_state)["kpi_results"]
        result_map = {item["kpi_id"]: item for item in rerun_results}
        for kpi in kpis:
            scores_by_kpi[kpi.kpi_id].append(result_map[kpi.kpi_id]["score"])

    stds = [pstdev(scores) for scores in scores_by_kpi.values() if scores]
    if not stds:
        return 0.0
    return sum(stds) / len(stds)


def run_evaluation(state: Dict) -> Dict:
    return {
        "coverage_ratio": coverage_ratio(state.get("sources", [])),
        "empty_retrieval_rate": empty_retrieval_rate(state.get("retrieval_stats", {})),
        "groundedness_ratio": groundedness_ratio(state.get("kpi_results", [])),
        "score_stability_std": score_stability_std(state),
    }
