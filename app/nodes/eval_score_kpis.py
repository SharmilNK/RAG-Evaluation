"""
eval_score_kpis.py
Identical to score_kpis_node but loads app/kpis_47.yaml instead of the default
app/kpis.yaml — no modification to the original score_kpis.py required.
Also stores the human-readable KPI name on each result so compare_ground_truth
can do name-based fuzzy matching.
"""
from __future__ import annotations

from typing import Dict, List

from app.kpi_catalog import load_kpi_catalog
from app.kpi_scoring import score_rubric_kpi
from app.vectorstore import build_collection

KPIS_47_PATH = "app/kpis_47.yaml"


def eval_score_kpis_node(state: Dict) -> Dict:
    """
    Score all 47 KPIs from kpis_47.yaml using the same rubric-scoring logic
    as the existing score_kpis_node, but against the pre-indexed eval sources.

    State keys read:  run_id, company_domain, sources
    State keys set:   kpi_results, missing_evidence, kpi_definitions
    """
    run_id = state["run_id"]
    company_domain = state.get("company_domain", "")
    sources: List[Dict] = state.get("sources", [])

    collection = build_collection(run_id)
    kpis = load_kpi_catalog(KPIS_47_PATH)

    results = []
    missing: List[str] = []

    for kpi in kpis:
        # All 47 AlixPartners KPIs are rubric type
        if kpi.type == "rubric":
            result, missing_flag = score_rubric_kpi(
                kpi, collection, company_domain=company_domain, full_sources=sources
            )
        else:
            # Quantitative fallback (unlikely with kpis_47.yaml but safe)
            from app.kpi_scoring import score_quant_kpi
            result, missing_flag = score_quant_kpi(kpi, sources)

        result_dict = result.model_dump()
        # Store the human-readable name so compare_ground_truth can match by name
        result_dict["_name"] = kpi.name

        results.append(result_dict)
        if missing_flag:
            missing.append(kpi.kpi_id)

    return {
        "kpi_results": results,
        "missing_evidence": missing,
        "kpi_definitions": [kpi.model_dump() for kpi in kpis],
    }
