from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List

import yaml

from app.debug_log import get_debug
from app.models import AggregatedKPIResult, KPIDriverResult, ReportArtifact


def _aggregate_by_pillar(results: List[KPIDriverResult]) -> List[AggregatedKPIResult]:
    pillar_map: Dict[str, List[KPIDriverResult]] = {}
    for result in results:
        pillar_map.setdefault(result.pillar, []).append(result)

    aggregated: List[AggregatedKPIResult] = []
    for pillar, items in pillar_map.items():
        total_weight = sum(item.confidence for item in items)
        if total_weight == 0:
            score = 0.0
        else:
            score = sum(item.score * item.confidence for item in items) / total_weight
        confidence = sum(item.confidence for item in items) / len(items)
        aggregated.append(
            AggregatedKPIResult(
                pillar=pillar,
                score=round(score, 2),
                confidence=round(confidence, 2),
                kpis=[item.kpi_id for item in items],
            )
        )

    return aggregated


def _overall_score(pillars: List[AggregatedKPIResult]) -> float:
    total_weight = sum(pillar.confidence for pillar in pillars)
    if total_weight == 0:
        return 0.0
    return round(
        sum(pillar.score * pillar.confidence for pillar in pillars) / total_weight,
        2,
    )


def aggregate_report_node(state: Dict) -> Dict:
    run_id = state["run_id"]
    company_name = state["company_name"]
    company_domain = state["company_domain"]
    url_count = state.get("url_count", 0)

    kpi_results = [KPIDriverResult(**item) for item in state.get("kpi_results", [])]
    pillar_scores = _aggregate_by_pillar(kpi_results)
    overall_score = _overall_score(pillar_scores)

    report = ReportArtifact(
        run_id=run_id,
        company_name=company_name,
        company_domain=company_domain,
        timestamp=datetime.now(timezone.utc).isoformat(),
        url_count=url_count,
        kpi_results=kpi_results,
        pillar_scores=pillar_scores,
        overall_score=overall_score,
        missing_evidence=state.get("missing_evidence", []),
        debug_log=get_debug() if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"} else None,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"report_{run_id}.yaml")

    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(report.model_dump(), handle, sort_keys=False)

    return {"report_path": output_path, "overall_score": overall_score}
