from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    source_id: str
    url: str
    title: str
    text: str
    domain: str
    retrieved_at: str
    tier: int
    page_type: Optional[str] = None
    # v2: Content-based tier classification metadata
    tier_reason: Optional[str] = None
    content_score: Optional[float] = None
    content_signals: Optional[Dict[str, object]] = None


class Citation(BaseModel):
    source_id: str
    url: str
    quote: str


class KPIDefinition(BaseModel):
    kpi_id: str
    name: str
    pillar: str
    type: str
    question: str
    rubric: Optional[List[str]] = None
    quant_rule: Optional[Dict[str, object]] = None
    evidence_requirements: Optional[str] = None


class KPIDriver(BaseModel):
    kpi_id: str
    name: str
    question: str
    parent_id: Optional[str] = None


class KPIDriverResult(BaseModel):
    kpi_id: str
    pillar: str
    type: str
    score: float
    confidence: float
    rationale: str
    citations: List[Citation] = Field(default_factory=list)
    details: Optional[Dict[str, object]] = None


class AggregatedKPIResult(BaseModel):
    pillar: str
    score: float
    confidence: float
    kpis: List[str] = Field(default_factory=list)


class ReportArtifact(BaseModel):
    run_id: str
    company_name: str
    company_domain: str
    timestamp: str
    url_count: int
    kpi_results: List[KPIDriverResult]
    pillar_scores: List[AggregatedKPIResult]
    overall_score: float
    missing_evidence: List[str]
    debug_log: Optional[List[str]] = None
