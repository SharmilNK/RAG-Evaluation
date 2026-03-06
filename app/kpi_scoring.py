from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from app.debug_log import add_debug
from app.models import Citation, KPIDefinition, KPIDriverResult
from app.vectorstore import retrieve_evidence
from app.tier_weighting import (
    retrieve_evidence_weighted,
    calculate_tier_quality,
    get_tier_distribution,
)
from app.corroboration import detect_corroboration
from app.dynamic_retrieval import determine_optimal_k
from app.source_eval import (
    detect_semantic_corroboration,
    calculate_freshness_boost,
    calculate_authority_boost,
    detect_contradictions,
    detect_source_independence,
)


POSITIVE_HINTS = ["strong", "improve", "growth", "quality", "efficient", "optimized"]
NEGATIVE_HINTS = ["risk", "decline", "issue", "problem", "weak"]
_LAST_LLM_CALL_AT: Optional[float] = None


def _extract_json(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _llm_score(prompt: str) -> Optional[dict]:
    global _LAST_LLM_CALL_AT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
            add_debug("[llm] missing OPENAI_API_KEY; using fallback")
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_retries = int(os.getenv("VITELIS_OPENAI_MAX_RETRIES", "3"))
    base_backoff = float(os.getenv("VITELIS_OPENAI_BACKOFF", "1.5"))
    min_delay = float(os.getenv("VITELIS_OPENAI_MIN_DELAY", "5.0"))

    for attempt in range(max_retries + 1):
        if _LAST_LLM_CALL_AT is not None:
            elapsed = time.time() - _LAST_LLM_CALL_AT
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a strict JSON generator."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=30,
            )
            _LAST_LLM_CALL_AT = time.time()
            if response.status_code == 429 and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after and retry_after.isdigit() else base_backoff * (2**attempt)
                if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
                    add_debug(f"[llm] 429 received; retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
            if response.status_code != 200:
                print(f"[KPI scoring] OpenAI API returned {response.status_code}; using heuristic fallback.")
                return None
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = _extract_json(content)
            if not data:
                print("[KPI scoring] LLM response had no valid JSON; using heuristic fallback.")
            if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
                add_debug("[llm] response parsed" if data else "[llm] response missing JSON")
            return data
        except requests.RequestException as exc:
            if attempt < max_retries:
                wait_time = base_backoff * (2**attempt)
                if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
                    add_debug(f"[llm] request failed; retrying in {wait_time:.1f}s ({exc})")
                time.sleep(wait_time)
                continue
            print(f"[KPI scoring] LLM call failed after retries ({exc!s}); using heuristic fallback.")
            if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
                add_debug(f"[llm] request failed; using fallback ({exc})")
            return None


def _score5_from_rubric(rubric: Optional[List[str]]) -> Optional[str]:
    """Extract the Score-5 (Quality Criteria = High) description from the rubric."""
    if not rubric:
        return None
    for line in rubric:
        stripped = line.strip()
        if stripped.startswith("5:"):
            return stripped[2:].strip()
    return None


def _fallback_rubric_score(
    text_blob: str,
    rubric: Optional[List[str]] = None,
) -> Tuple[int, float, str]:
    """
    Score 1-5 when LLM is unavailable.
    If rubric has Score-5 text, use evidence overlap with that ideal; else use keyword hints.
    """
    ideal_5 = _score5_from_rubric(rubric) if rubric else None
    if ideal_5 and text_blob:
        # Rubric-aware: how much does evidence overlap with "ideal (5)" description?
        ideal_tokens = set(re.findall(r"\b\w{3,}\b", ideal_5.lower()))
        evidence_tokens = set(re.findall(r"\b\w{3,}\b", text_blob.lower()))
        if ideal_tokens:
            overlap = len(ideal_tokens & evidence_tokens) / len(ideal_tokens)
            if overlap >= 0.35:
                score = 5
            elif overlap >= 0.22:
                score = 4
            elif overlap >= 0.12:
                score = 3
            elif overlap >= 0.05:
                score = 2
            else:
                score = 1
            confidence = 0.35 + min(0.25, overlap)
            rationale = "Heuristic fallback: evidence overlap with Quality Criteria (5=High)."
            return score, round(confidence, 2), rationale

    # Legacy: positive/negative keyword counts when no rubric or no ideal_5
    score = 3
    for word in POSITIVE_HINTS:
        if word in text_blob:
            score += 1
    for word in NEGATIVE_HINTS:
        if word in text_blob:
            score -= 1
    score = max(1, min(5, score))
    confidence = 0.4 if text_blob else 0.2
    rationale = "Heuristic fallback scoring based on evidence text."
    return score, confidence, rationale


def _build_citations(evidences: List[Tuple[dict, str]], limit: int = 3) -> List[Citation]:
    citations = []
    for metadata, doc in evidences[:limit]:
        citations.append(
            Citation(
                source_id=metadata.get("source_id", ""),
                url=metadata.get("url", ""),
                quote=doc[:240],
            )
        )
    return citations


def _build_full_text_evidences(
    chunk_evidences: List[Tuple[dict, str, float]],
    full_sources: List[Dict],
) -> List[Tuple[dict, str, float]]:
    """
    Build evidence tuples using FULL source text instead of small chunks.

    The v2 evaluation functions (corroboration, freshness, contradictions)
    need full document text to work properly — small 500-char chunks don't
    contain enough dates, claims, or context.

    This maps chunk source_ids back to their full source documents and
    returns one evidence tuple per unique source with the complete text.
    """
    if not full_sources:
        return chunk_evidences

    # Index full sources by source_id
    source_by_id: Dict[str, Dict] = {}
    for source in full_sources:
        sid = source.get("source_id", "")
        if sid:
            source_by_id[sid] = source

    # Get unique source_ids referenced in chunks
    seen: set = set()
    full_evidences: List[Tuple[dict, str, float]] = []

    for meta, _doc, score in chunk_evidences:
        sid = meta.get("source_id", "")
        if sid in seen or sid not in source_by_id:
            continue
        seen.add(sid)

        source = source_by_id[sid]
        full_meta = {
            "source_id": sid,
            "url": source.get("url", meta.get("url", "")),
            "title": source.get("title", ""),
            "domain": source.get("domain", ""),
            "tier": source.get("tier", meta.get("tier", 3)),
            "retrieved_at": source.get("retrieved_at", ""),
            "page_type": source.get("page_type", ""),
        }
        full_evidences.append((full_meta, source.get("text", ""), score))

    return full_evidences if full_evidences else chunk_evidences


def calculate_enhanced_confidence(
    base_confidence: float,
    evidences: List[Tuple[dict, str, float]],
    corroboration: float,
    llm_citations_present: bool,
    company_domain: str = "",
    full_sources: Optional[List[Dict]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate enhanced confidence score with all v2 quality factors.

    Components (v1 — retained):
    - Base confidence from LLM or fallback
    - Tier quality boost (0 to +0.15)
    - Source diversity boost (+0.05 if 3+ unique sources)
    - Citation penalty (-0.3 if LLM missing citations)
    - Low evidence penalty (-0.2 if <3 chunks)

    Components (v2 — new):
    - Semantic corroboration boost (0 to +0.15) — claim-level agreement
    - Source independence check — reduces corroboration for syndicated duplicates
    - Freshness boost/penalty (-0.15 to +0.10) — recency of evidence
    - Authority boost (0 to +0.15) — 3rd-party validation
    - Contradiction penalty (0 to -0.20) — conflicting evidence

    Args:
        evidences: Chunk-level evidence from ChromaDB (used for tier quality)
        full_sources: Full source documents from pipeline state (used for
            v2 evaluation: corroboration, freshness, authority, contradictions).
            If None, falls back to using chunks (less accurate).

    Returns:
        Tuple of (final_confidence, evaluation_details dict)
    """
    enabled = os.getenv("VITELIS_ENHANCED_CONFIDENCE", "true").lower() in {"1", "true"}

    eval_details: Dict[str, Any] = {}

    if not enabled:
        return base_confidence, eval_details

    confidence = base_confidence

    # --- v1: Tier quality boost (uses chunks — fine for tier metadata) ---
    tier_boost = calculate_tier_quality(evidences)
    confidence += tier_boost

    # Build full-text evidences for v2 functions
    # Chunks are ~500 chars — too small for date parsing, claim extraction, etc.
    # Full sources have the complete page text (often 1000+ words)
    full_evidences = _build_full_text_evidences(evidences, full_sources or [])

    # --- v2: Semantic corroboration (uses full text for claim extraction) ---
    semantic_corr = detect_semantic_corroboration(full_evidences)
    corroboration_max = float(os.getenv("VITELIS_CORROBORATION_BOOST_MAX", "0.15"))
    corr_boost = semantic_corr["corroboration_score"] * corroboration_max
    eval_details["semantic_corroboration"] = semantic_corr

    # --- v2: Source independence check — reduce corroboration for near-duplicate sources ---
    independence = detect_source_independence(full_evidences)
    corr_boost = max(0.0, corr_boost + independence["corroboration_penalty"])
    eval_details["source_independence"] = independence

    confidence += corr_boost

    # --- v2: Freshness boost/penalty (uses full text for date parsing) ---
    freshness_boost, freshness_per_source = calculate_freshness_boost(full_evidences)
    confidence += freshness_boost
    eval_details["freshness"] = {
        "boost": freshness_boost,
        "per_source": freshness_per_source,
    }

    # --- v2: Authority boost (uses full text + URL for classification) ---
    authority_boost, authority_per_source = calculate_authority_boost(full_evidences, company_domain)
    confidence += authority_boost
    eval_details["authority"] = {
        "boost": authority_boost,
        "per_source": authority_per_source,
    }

    # --- v2: Contradiction penalty (uses full text for opposing claims) ---
    contradiction_info = detect_contradictions(full_evidences)
    confidence += contradiction_info["confidence_penalty"]
    eval_details["contradictions"] = contradiction_info

    # --- v1: Source diversity boost ---
    unique_sources = len({meta.get("source_id", "") for meta, _, _ in evidences if meta.get("source_id")})
    if unique_sources >= 3:
        diversity_boost = float(os.getenv("VITELIS_DIVERSITY_BOOST", "0.05"))
        confidence += diversity_boost

    # --- v1: Citation penalty ---
    if not llm_citations_present and base_confidence > 0.3:
        confidence -= 0.3

    # --- v1: Low evidence penalty ---
    if len(evidences) < 3:
        confidence -= 0.2

    return round(max(0.0, min(1.0, confidence)), 2), eval_details


def score_rubric_kpi(
    kpi: KPIDefinition,
    collection,
    k: int = 6,
    company_domain: str = "",
    full_sources: Optional[List[Dict]] = None,
) -> Tuple[KPIDriverResult, bool]:
    # Determine optimal k (dynamic tuning if enabled)
    k_used = determine_optimal_k(kpi, default_k=k)

    # Retrieve evidence with tier weighting (if enabled)
    tier_weighting_enabled = os.getenv("VITELIS_ENABLE_TIER_WEIGHTING", "true").lower() in {"1", "true"}

    if tier_weighting_enabled:
        evidences = retrieve_evidence_weighted(collection, kpi.question, k=k_used)
    else:
        # Fall back to standard retrieval
        evidences_basic = retrieve_evidence(collection, kpi.question, k=k_used)
        # Convert to format expected by new functions: add dummy score
        evidences = [(meta, doc, 0.5) for meta, doc in evidences_basic]

    # Extract text for fallback scoring (handle both formats)
    if evidences and len(evidences[0]) == 3:
        evidence_text = " ".join(doc for _, doc, _ in evidences).lower()
        citations = _build_citations([(meta, doc) for meta, doc, _ in evidences])
    else:
        evidence_text = " ".join(doc for _, doc in evidences).lower()
        citations = _build_citations(evidences)

    # Detect cross-source corroboration (if enabled)
    corroboration_enabled = os.getenv("VITELIS_ENABLE_CORROBORATION", "true").lower() in {"1", "true"}
    corroboration = 0.0
    if corroboration_enabled and evidences:
        corroboration = detect_corroboration(evidences, min_sources=2)

    if not evidences:
        return (
            KPIDriverResult(
                kpi_id=kpi.kpi_id,
                pillar=kpi.pillar,
                type=kpi.type,
                score=1,
                confidence=0.2,
                rationale="No evidence retrieved for this KPI.",
                citations=[],
                details={"llm_used": False},
            ),
            True,
        )

    rubric = "\n".join(kpi.rubric or [])
    ideal_5 = _score5_from_rubric(kpi.rubric)

    # Send full chunk (500 chars) so LLM has full context; avoid truncation.
    char_limit = 500
    if evidences and len(evidences[0]) == 3:
        evidence_block = "\n".join(
            f"- [{meta.get('source_id', '')}] {meta.get('url', '')}: {doc[:char_limit]}"
            for meta, doc, _ in evidences
        )
    else:
        evidence_block = "\n".join(
            f"- [{meta.get('source_id', '')}] {meta.get('url', '')}: {doc[:char_limit]}"
            for meta, doc in evidences
        )

    ideal_line = f"\nIdeal (5 = High): \"{ideal_5}\"\nScore how close the evidence supports this level (1 = not at all, 5 = fully).\n" if ideal_5 else ""

    prompt = f"""
    You are a Business Analyst. Your task is to audit the following context using 
    the KPI Drivers.
    Score the KPI from 1-5 using the rubric and evidence.
    Return strict JSON: {{"kpi_id": "...", "score": 1-5, "confidence": 0-1, "rationale": "...", "citations": [{{"source_id": "...", "url": "...", "quote": "..."}}]}}.
    Citations must include source_id, url, quote.

    KPI: {kpi.name}
    Question: {kpi.question}
    Rubric:
    {rubric}
    {ideal_line}
    Evidence:
    {evidence_block}
    """

    llm_data: Optional[dict] = None
    try:
        llm_data = _llm_score(prompt)
    except requests.RequestException:
        llm_data = None

    if llm_data:
        score = int(llm_data.get("score", 3))
        confidence = float(llm_data.get("confidence", 0.5))
        rationale = str(llm_data.get("rationale", ""))
        raw_citations = llm_data.get("citations") or []
        parsed_citations = []
        for item in raw_citations:
            if not isinstance(item, dict):
                continue
            parsed_citations.append(
                Citation(
                    source_id=str(item.get("source_id", "")),
                    url=str(item.get("url", "")),
                    quote=str(item.get("quote", ""))[:240],
                )
            )
        missing_flag = False
        if not parsed_citations:
            missing_flag = True

        # Calculate enhanced confidence (v2: returns tuple with eval details)
        final_confidence, eval_details = calculate_enhanced_confidence(
            base_confidence=confidence,
            evidences=evidences,
            corroboration=corroboration,
            llm_citations_present=bool(parsed_citations),
            company_domain=company_domain,
            full_sources=full_sources,
        )

        # Build extended details
        tier_distribution = get_tier_distribution(evidences)
        unique_sources = len({meta.get("source_id", "") for meta, _, _ in evidences if meta.get("source_id")})

        details = {
            "llm_used": True,
            "tier_distribution": tier_distribution,
            "corroboration_score": eval_details.get("semantic_corroboration", {}).get("corroboration_score", corroboration),
            "unique_sources": unique_sources,
            "k_used": k_used,
            # v2: Full source evaluation breakdown
            "source_evaluation": eval_details,
        }

        return (
            KPIDriverResult(
                kpi_id=kpi.kpi_id,
                pillar=kpi.pillar,
                type=kpi.type,
                score=max(1, min(5, score)),
                confidence=final_confidence,
                rationale=rationale or "LLM scoring with evidence.",
                citations=parsed_citations or citations,
                details=details,
            ),
            missing_flag,
        )

    score, confidence, rationale = _fallback_rubric_score(evidence_text, rubric=kpi.rubric)

    # Calculate enhanced confidence for fallback path too (v2: tuple return)
    final_confidence, eval_details = calculate_enhanced_confidence(
        base_confidence=confidence,
        evidences=evidences,
        corroboration=corroboration,
        llm_citations_present=False,
        company_domain=company_domain,
        full_sources=full_sources,
    )

    # Build extended details
    tier_distribution = get_tier_distribution(evidences)
    unique_sources = len({meta.get("source_id", "") for meta, _, _ in evidences if meta.get("source_id")})

    details = {
        "llm_used": False,
        "tier_distribution": tier_distribution,
        "corroboration_score": eval_details.get("semantic_corroboration", {}).get("corroboration_score", corroboration),
        "unique_sources": unique_sources,
        "k_used": k_used,
        "source_evaluation": eval_details,
    }

    return (
        KPIDriverResult(
            kpi_id=kpi.kpi_id,
            pillar=kpi.pillar,
            type=kpi.type,
            score=score,
            confidence=final_confidence,
            rationale=rationale,
            citations=citations,
            details=details,
        ),
        False,
    )


def _count_mentions(sources: List[Dict], keywords: Iterable[str]) -> Tuple[int, List[Dict]]:
    count = 0
    matched_sources: List[Dict] = []
    for source in sources:
        text = source.get("text", "").lower()
        matched = False
        for keyword in keywords:
            if keyword.lower() in text:
                count += text.count(keyword.lower())
                matched = True
        if matched:
            matched_sources.append(source)
    return count, matched_sources


def _parse_dates(text: str) -> List[datetime]:
    results: List[datetime] = []
    numeric = re.findall(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    for year, month, day in numeric:
        try:
            results.append(datetime(int(year), int(month), int(day), tzinfo=timezone.utc))
        except ValueError:
            continue

    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    month_year = re.findall(r"\b([A-Za-z]{3,9})\s+(20\d{2})\b", text)
    for month_name, year in month_year:
        month = month_map.get(month_name.lower())
        if not month:
            continue
        try:
            results.append(datetime(int(year), int(month), 1, tzinfo=timezone.utc))
        except ValueError:
            continue

    return results


def _normalize_thresholds(thresholds: List[int]) -> List[int]:
    if len(thresholds) >= 4:
        return thresholds[:4]
    return (thresholds + [3, 5, 8, 12])[:4]


def _score_from_thresholds(value: int, thresholds: List[int]) -> int:
    thresholds = _normalize_thresholds(thresholds)
    if value < thresholds[0]:
        return 1
    if value < thresholds[1]:
        return 2
    if value < thresholds[2]:
        return 3
    if value < thresholds[3]:
        return 4
    return 5


def _score_recency(days: int, thresholds: List[int]) -> int:
    thresholds = _normalize_thresholds(thresholds)
    if days <= thresholds[0]:
        return 5
    if days <= thresholds[1]:
        return 4
    if days <= thresholds[2]:
        return 3
    if days <= thresholds[3]:
        return 2
    return 1


def score_quant_kpi(kpi: KPIDefinition, sources: List[Dict]) -> Tuple[KPIDriverResult, bool]:
    rule = kpi.quant_rule or {}
    method = str(rule.get("method", ""))
    keywords = rule.get("keywords", [])
    thresholds = rule.get("thresholds", [1, 3, 5, 8])

    if method == "count_mentions":
        count, matched_sources = _count_mentions(sources, keywords)
        score = _score_from_thresholds(count, thresholds)
        confidence = 0.7 if count > 0 else 0.3
        citations = []
        for source in matched_sources[:2]:
            citations.append(
                Citation(
                    source_id=source.get("source_id", ""),
                    url=source.get("url", ""),
                    quote=source.get("text", "")[:240],
                )
            )
        return (
            KPIDriverResult(
                kpi_id=kpi.kpi_id,
                pillar=kpi.pillar,
                type=kpi.type,
                score=score,
                confidence=round(confidence, 2),
                rationale=f"Matched {count} keyword mentions.",
                citations=citations,
                details={"count": count, "keywords": keywords},
            ),
            count == 0,
        )

    if method == "recency_days":
        now = datetime.now(timezone.utc)
        best_days: Optional[int] = None
        for source in sources:
            text = source.get("text", "").lower()
            if not any(keyword.lower() in text for keyword in keywords):
                continue
            for dt in _parse_dates(text):
                days = (now - dt).days
                if best_days is None or days < best_days:
                    best_days = days

        if best_days is None:
            return (
                KPIDriverResult(
                    kpi_id=kpi.kpi_id,
                    pillar=kpi.pillar,
                    type=kpi.type,
                    score=1,
                    confidence=0.2,
                    rationale="No dated AI announcement found.",
                    citations=[],
                    details={"days_since": None},
                ),
                True,
            )

        score = _score_recency(best_days, thresholds)
        confidence = 0.6
        return (
            KPIDriverResult(
                kpi_id=kpi.kpi_id,
                pillar=kpi.pillar,
                type=kpi.type,
                score=score,
                confidence=round(confidence, 2),
                rationale=f"Most recent AI mention {best_days} days ago.",
                citations=[],
                details={"days_since": best_days},
            ),
            False,
        )

    return (
        KPIDriverResult(
            kpi_id=kpi.kpi_id,
            pillar=kpi.pillar,
            type=kpi.type,
            score=1,
            confidence=0.2,
            rationale="Unknown quant rule; defaulted to 1.",
            citations=[],
            details={"method": method},
        ),
        True,
    )
