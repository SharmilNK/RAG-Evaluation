from __future__ import annotations

from typing import Dict, List

from app.kpi_catalog import load_kpi_catalog
from app.kpi_scoring import score_quant_kpi, score_rubric_kpi
from app.vectorstore import build_collection


def score_kpis_node(state: Dict) -> Dict:
    run_id = state["run_id"]
    company_domain = state.get("company_domain", "")
    sources: List[Dict] = state.get("sources", [])

    collection = build_collection(run_id)
    kpis = load_kpi_catalog()

    results = []
    missing: List[str] = []

    for kpi in kpis:
        if kpi.type == "rubric":
            result, missing_flag = score_rubric_kpi(kpi, collection, company_domain=company_domain, full_sources=sources)
        else:
            result, missing_flag = score_quant_kpi(kpi, sources)

        results.append(result.model_dump())
        if missing_flag:
            missing.append(kpi.kpi_id)

    return {
        "kpi_results": results,
        "missing_evidence": missing,
        "kpi_definitions": [kpi.model_dump() for kpi in kpis],  # passed to evaluate_rag_node for ground truth lookup
    }
