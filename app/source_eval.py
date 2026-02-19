"""
Advanced Source Evaluation Engine — v2
======================================
Five major upgrades to source quality assessment:

1. CONTENT-BASED TIER CLASSIFICATION
   Old: URL keyword matching ("investor" in URL → Tier 1)
   New: Analyzes actual page content — structural depth, data density,
        specificity of claims, presence of metrics/numbers, document formality.

2. SEMANTIC CORROBORATION
   Old: Keyword overlap between sources (shared words = corroboration)
   New: Extracts discrete *claims* from each source, then checks whether
        different sources make the same claim. Two pages both saying
        "we use AI for network optimization" is real corroboration.
        Two pages both containing the word "AI" is not.

3. SOURCE FRESHNESS WEIGHTING
   Old: No date awareness — a 2019 blog and a 2025 annual report weighted equally.
   New: Parses dates from each source, computes a freshness score (0–1),
        and applies a configurable decay. Recent sources boost confidence;
        stale sources penalize it.

4. AUTHORITY SIGNAL DETECTION
   Old: No distinction between company self-claims and third-party validation.
   New: Classifies each source as FIRST_PARTY (company's own site),
        THIRD_PARTY_NEWS (press coverage), THIRD_PARTY_ANALYST (analyst/research),
        REGULATORY (government/regulatory body), or UNKNOWN. Third-party
        validation of a claim is worth more than self-reporting.

5. CONTRADICTION DETECTION
   Old: None — conflicting evidence was invisible.
   New: Scans evidence pairs for semantic contradictions — e.g., one source
        says "large AI team" while another says "small data science effort".
        Flags these and penalizes confidence when contradictions are found.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


# ============================================================================
# 1. CONTENT-BASED TIER CLASSIFICATION
# ============================================================================

# Signals that indicate high-quality, substantive content
_TIER1_CONTENT_SIGNALS = {
    "metrics": [
        r"\d+\.?\d*\s*%",               # percentages
        r"\$\s*\d+",                      # dollar amounts
        r"\d+\s*(million|billion|mn|bn)", # financial figures
        r"revenue|EBITDA|margin|ROI|ARR|MRR",
        r"year-over-year|YoY|quarter",
    ],
    "structure": [
        r"(executive\s+summary|table\s+of\s+contents|key\s+(findings|takeaways))",
        r"(methodology|framework|appendix|glossary)",
        r"(figure\s+\d|table\s+\d|chart\s+\d|exhibit\s+\d)",
    ],
    "specificity": [
        r"(case\s+study|deployment|implementation|pilot|POC|proof\s+of\s+concept)",
        r"(benchmark|accuracy|precision|recall|F1|latency\s+\d)",
        r"(patent|peer.reviewed|published\s+in|IEEE|arxiv|ACM)",
    ],
    "governance": [
        r"(board\s+of\s+directors|committee|oversight|audit)",
        r"(policy|regulation|compliance|GDPR|ISO\s+\d)",
        r"(risk\s+assessment|impact\s+assessment|due\s+diligence)",
    ],
}

_TIER3_SIGNALS = [
    r"(cookie\s+(policy|notice|consent)|privacy\s+policy|terms\s+(of\s+use|and\s+conditions))",
    r"(subscribe|newsletter|sign\s+up|follow\s+us|share\s+this)",
    r"(404|page\s+not\s+found|error|coming\s+soon)",
    r"(skip\s+to\s+(main\s+)?content|navigation|breadcrumb)",
]


def classify_tier_content(url: str, text: str, title: str = "") -> Dict[str, Any]:
    """
    Classify source tier based on actual content analysis, not just URL keywords.

    Returns a dict with:
        - tier: int (1, 2, or 3)
        - tier_reason: str explaining why
        - content_signals: dict of detected signal categories
        - content_score: float (0–1) representing content quality

    Scoring rubric:
        - Count signals across 4 categories (metrics, structure, specificity, governance)
        - Weight by text length (very short pages penalized)
        - Apply URL-hint bonus (not decisive, but additive)
        - Tier 1: content_score >= 0.55
        - Tier 2: content_score >= 0.25
        - Tier 3: below 0.25 or dominated by boilerplate
    """
    text_lower = text.lower()
    url_lower = url.lower()
    combined = f"{title} {text}".lower()

    # --- Check for Tier 3 disqualifiers first ---
    boilerplate_hits = 0
    for pattern in _TIER3_SIGNALS:
        if re.search(pattern, combined, re.IGNORECASE):
            boilerplate_hits += 1

    if boilerplate_hits >= 2 and len(text) < 500:
        return {
            "tier": 3,
            "tier_reason": "Boilerplate/navigation page with minimal content",
            "content_signals": {"boilerplate_hits": boilerplate_hits},
            "content_score": 0.05,
        }

    # --- Score content signals ---
    signal_counts: Dict[str, int] = {}
    signal_details: Dict[str, List[str]] = {}

    for category, patterns in _TIER1_CONTENT_SIGNALS.items():
        hits = []
        for pattern in patterns:
            matches = re.findall(pattern, combined, re.IGNORECASE)
            if matches:
                hits.extend(matches[:3])  # Cap per pattern
        signal_counts[category] = len(hits)
        signal_details[category] = [str(h) for h in hits[:5]]

    total_signals = sum(signal_counts.values())

    # --- Text quality factors ---
    word_count = len(text.split())
    sentence_count = len(re.findall(r"[.!?]+", text)) or 1

    # Longer, well-structured content scores higher
    length_factor = min(1.0, word_count / 800)  # Maxes at ~800 words

    # Average sentence length — too short = listy/nav, too long = fine
    avg_sentence_len = word_count / sentence_count
    structure_factor = min(1.0, avg_sentence_len / 15) if avg_sentence_len > 3 else 0.2

    # --- URL hint bonus (additive, not decisive) ---
    url_bonus = 0.0
    url_tier1_hints = ["investor", "annual", "report", "press", "product", "pricing"]
    url_tier2_hints = ["blog", "news", "about", "careers", "sustainability"]
    if any(hint in url_lower for hint in url_tier1_hints):
        url_bonus = 0.12
    elif any(hint in url_lower for hint in url_tier2_hints):
        url_bonus = 0.06

    # --- Compute composite content score ---
    # Signals contribute up to 0.5, length up to 0.25, structure up to 0.15, URL up to 0.12
    signal_score = min(0.5, total_signals * 0.06)
    content_score = signal_score + (length_factor * 0.25) + (structure_factor * 0.15) + url_bonus
    content_score = round(min(1.0, content_score), 3)

    # --- Assign tier ---
    if content_score >= 0.55:
        tier = 1
        tier_reason = f"High-quality content: {total_signals} quality signals, {word_count} words"
    elif content_score >= 0.25:
        tier = 2
        tier_reason = f"Moderate content: {total_signals} signals, {word_count} words"
    else:
        tier = 3
        tier_reason = f"Thin content: {total_signals} signals, {word_count} words"

    return {
        "tier": tier,
        "tier_reason": tier_reason,
        "content_signals": {
            "signal_counts": signal_counts,
            "signal_examples": signal_details,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "boilerplate_hits": boilerplate_hits,
        },
        "content_score": content_score,
    }


# ============================================================================
# 2. SEMANTIC CORROBORATION
# ============================================================================

# Keywords that signal a sentence contains a factual claim worth comparing
_CLAIM_KEYWORDS = [
    # Technology & AI
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "generative ai", "genai", "neural network", "nlp", "computer vision",
    "automation", "chatbot", "predictive", "algorithm",
    # Actions
    "deploy", "implement", "launch", "integrate", "develop", "build",
    "partner", "collaborate", "invest", "acquire", "adopt",
    # Results / metrics
    "revenue", "growth", "increase", "decrease", "improve", "reduce",
    "customer", "employee", "team", "engineer",
    # Strategy
    "strategy", "framework", "governance", "compliance", "policy",
    "roadmap", "vision", "initiative", "program",
]


def extract_claims(text: str, max_claims: int = 30) -> List[str]:
    """
    Extract discrete factual claims from source text.

    Uses a two-pass approach:
    1. Split text into sentences
    2. Keep sentences that contain claim-relevant keywords AND are
       substantive (not navigation/boilerplate)

    This works on full source documents (1000+ words) where pattern-only
    extraction would miss too many claims.
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    claims: List[str] = []
    seen: Set[str] = set()

    for sentence in sentences:
        normalized = " ".join(sentence.split()).strip()

        # Skip too short or too long
        if len(normalized) < 30 or len(normalized) > 500:
            continue

        # Skip boilerplate
        lower = normalized.lower()
        if any(bp in lower for bp in [
            "cookie", "privacy policy", "terms of use", "skip to",
            "subscribe", "newsletter", "copyright", "all rights reserved",
            "click here", "read more", "learn more", "sign up",
        ]):
            continue

        # Must contain at least one claim keyword
        keyword_hits = sum(1 for kw in _CLAIM_KEYWORDS if kw in lower)
        if keyword_hits == 0:
            continue

        # Dedup by first 60 chars
        key = lower[:60]
        if key not in seen:
            seen.add(key)
            claims.append(normalized)

    return claims[:max_claims]


def _claim_similarity(claim_a: str, claim_b: str) -> float:
    """
    Compute similarity between two claims using weighted token overlap.

    We strip stop words and compare meaningful tokens. We also give
    extra weight to numbers and named entities (capitalized words).
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "will", "would", "could", "should", "this",
        "that", "these", "those", "it", "its", "we", "our", "us", "their",
    }

    def tokenize(text: str) -> Counter:
        tokens = re.findall(r"\b\w+\b", text.lower())
        meaningful = [t for t in tokens if t not in stop_words and len(t) > 2]
        return Counter(meaningful)

    tokens_a = tokenize(claim_a)
    tokens_b = tokenize(claim_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Intersection over union (Jaccard on multisets)
    intersection = sum((tokens_a & tokens_b).values())
    union = sum((tokens_a | tokens_b).values())

    if union == 0:
        return 0.0

    base_sim = intersection / union

    # Bonus for shared numbers (strong signal)
    nums_a = set(re.findall(r"\d+\.?\d*", claim_a))
    nums_b = set(re.findall(r"\d+\.?\d*", claim_b))
    shared_nums = nums_a & nums_b
    num_bonus = min(0.15, len(shared_nums) * 0.05)

    return min(1.0, base_sim + num_bonus)


def detect_semantic_corroboration(
    evidences: List[Tuple[dict, str, float]],
    similarity_threshold: float = 0.35,
    min_sources: int = 2,
) -> Dict[str, Any]:
    """
    Detect claim-level corroboration across sources.

    Process:
    1. Extract claims from each source's text
    2. Compare claims across *different* sources
    3. If Source A and Source B make similar claims → corroborated
    4. Score based on number of corroborated claims and source diversity

    Returns:
        - corroboration_score: float (0–1)
        - corroborated_claims: list of {"claim_a", "claim_b", "source_a", "source_b", "similarity"}
        - claim_counts: dict of source_id → number of claims extracted
        - unique_sources: int
    """
    # Group text by source_id
    source_texts: Dict[str, str] = {}
    for meta, doc, _ in evidences:
        sid = meta.get("source_id", "")
        if not sid:
            continue
        source_texts.setdefault(sid, "")
        source_texts[sid] += " " + doc

    unique_sources = len(source_texts)
    if unique_sources < min_sources:
        return {
            "corroboration_score": 0.0,
            "corroborated_claims": [],
            "claim_counts": {},
            "unique_sources": unique_sources,
        }

    # Extract claims per source
    source_claims: Dict[str, List[str]] = {}
    claim_counts: Dict[str, int] = {}
    for sid, text in source_texts.items():
        claims = extract_claims(text)
        source_claims[sid] = claims
        claim_counts[sid] = len(claims)

    # Cross-source claim comparison
    source_ids = list(source_claims.keys())
    corroborated: List[Dict[str, Any]] = []

    for i in range(len(source_ids)):
        for j in range(i + 1, len(source_ids)):
            sid_a, sid_b = source_ids[i], source_ids[j]
            for claim_a in source_claims[sid_a]:
                for claim_b in source_claims[sid_b]:
                    sim = _claim_similarity(claim_a, claim_b)
                    if sim >= similarity_threshold:
                        corroborated.append({
                            "claim_a": claim_a[:200],
                            "claim_b": claim_b[:200],
                            "source_a": sid_a,
                            "source_b": sid_b,
                            "similarity": round(sim, 3),
                        })

    # Score: based on corroborated claim count and source diversity
    if not corroborated:
        # Fall back to source diversity alone (mild score)
        diversity_score = min(0.3, unique_sources * 0.1)
        return {
            "corroboration_score": round(diversity_score, 2),
            "corroborated_claims": [],
            "claim_counts": claim_counts,
            "unique_sources": unique_sources,
        }

    # More corroborated claims = higher score
    claim_score = min(0.5, len(corroborated) * 0.08)

    # Source diversity of corroborated claims
    corr_sources = set()
    for c in corroborated:
        corr_sources.add(c["source_a"])
        corr_sources.add(c["source_b"])
    diversity_score = min(0.5, len(corr_sources) * 0.12)

    total_score = round(min(1.0, claim_score + diversity_score), 2)

    return {
        "corroboration_score": total_score,
        "corroborated_claims": corroborated[:15],  # Cap for report size
        "claim_counts": claim_counts,
        "unique_sources": unique_sources,
    }


# ============================================================================
# 3. SOURCE FRESHNESS WEIGHTING
# ============================================================================

def _extract_dates(text: str) -> List[datetime]:
    """Extract dates from text, supporting multiple formats."""
    results: List[datetime] = []

    # ISO/numeric: 2025-01-15, 2025/01/15
    for match in re.finditer(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        try:
            results.append(datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc))
        except ValueError:
            continue

    # "January 2025", "Jan 2025", "15 January 2025"
    month_map = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12,
    }

    # "15 January 2025"
    for match in re.finditer(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})\b", text):
        month = month_map.get(match.group(2).lower())
        if month:
            try:
                results.append(datetime(int(match.group(3)), month, int(match.group(1)), tzinfo=timezone.utc))
            except ValueError:
                continue

    # "January 2025"
    for match in re.finditer(r"\b([A-Za-z]{3,9})\s+(20\d{2})\b", text):
        month = month_map.get(match.group(1).lower())
        if month:
            try:
                results.append(datetime(int(match.group(2)), month, 15, tzinfo=timezone.utc))
            except ValueError:
                continue

    return results


def calculate_freshness(
    text: str,
    retrieved_at: str = "",
    max_age_days: int = 730,
) -> Dict[str, Any]:
    """
    Calculate freshness score for a source document.

    Freshness is based on the most recent date found in the content.
    If no dates are found, we use retrieved_at as a weak signal.

    Scoring (exponential decay):
        - 0–30 days old:   freshness = 1.0–0.9
        - 30–90 days:      freshness = 0.9–0.7
        - 90–180 days:     freshness = 0.7–0.5
        - 180–365 days:    freshness = 0.5–0.3
        - 365–730 days:    freshness = 0.3–0.1
        - >730 days:       freshness = 0.1

    Returns:
        - freshness_score: float (0–1)
        - newest_date: str (ISO) or None
        - dates_found: int
        - age_days: int or None
    """
    now = datetime.now(timezone.utc)
    dates = _extract_dates(text)

    # Use retrieved_at as fallback date
    if retrieved_at and not dates:
        try:
            ret_dt = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
            dates = [ret_dt]
        except (ValueError, TypeError):
            pass

    if not dates:
        return {
            "freshness_score": 0.3,  # Unknown age → moderate penalty
            "newest_date": None,
            "dates_found": 0,
            "age_days": None,
        }

    newest = max(dates)
    age_days = max(0, (now - newest).days)

    # Exponential decay: freshness = max(0.1, 1.0 - (age/max_age)^0.7)
    ratio = min(1.0, age_days / max_age_days)
    freshness_score = max(0.1, round(1.0 - (ratio ** 0.7), 3))

    return {
        "freshness_score": freshness_score,
        "newest_date": newest.isoformat(),
        "dates_found": len(dates),
        "age_days": age_days,
    }


def calculate_freshness_boost(
    evidences: List[Tuple[dict, str, float]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate confidence boost/penalty based on evidence freshness.

    Returns:
        - boost: float (-0.15 to +0.10)
        - details: freshness breakdown per source
    """
    if not evidences:
        return 0.0, {}

    max_boost = float(os.getenv("VITELIS_FRESHNESS_BOOST_MAX", "0.10"))
    max_penalty = float(os.getenv("VITELIS_FRESHNESS_PENALTY_MAX", "0.15"))

    per_source: Dict[str, Dict] = {}
    scores: List[float] = []

    for meta, doc, _ in evidences:
        sid = meta.get("source_id", "")
        if sid in per_source:
            continue  # One score per source
        freshness = calculate_freshness(doc, meta.get("retrieved_at", ""))
        per_source[sid] = freshness
        scores.append(freshness["freshness_score"])

    if not scores:
        return 0.0, per_source

    avg_freshness = sum(scores) / len(scores)

    # Map avg freshness to boost/penalty
    # avg >= 0.7 → positive boost (up to +0.10)
    # avg 0.4–0.7 → neutral (0)
    # avg < 0.4 → penalty (down to -0.15)
    if avg_freshness >= 0.7:
        boost = max_boost * ((avg_freshness - 0.7) / 0.3)
    elif avg_freshness < 0.4:
        boost = -max_penalty * ((0.4 - avg_freshness) / 0.4)
    else:
        boost = 0.0

    return round(boost, 3), per_source


# ============================================================================
# 4. AUTHORITY SIGNAL DETECTION
# ============================================================================

AUTHORITY_FIRST_PARTY = "first_party"
AUTHORITY_THIRD_PARTY_NEWS = "third_party_news"
AUTHORITY_THIRD_PARTY_ANALYST = "third_party_analyst"
AUTHORITY_REGULATORY = "regulatory"
AUTHORITY_UNKNOWN = "unknown"

# Known third-party news/analyst/regulatory domains
_NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "cnbc.com", "ft.com", "bbc.com",
    "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
    "zdnet.com", "venturebeat.com", "theregister.com", "businessinsider.com",
    "forbes.com", "fortune.com", "wsj.com", "nytimes.com", "theguardian.com",
}

_ANALYST_DOMAINS = {
    "gartner.com", "forrester.com", "mckinsey.com", "bcg.com", "bain.com",
    "deloitte.com", "accenture.com", "kpmg.com", "pwc.com", "ey.com",
    "idc.com", "statista.com", "cbinsights.com", "crunchbase.com",
}

_REGULATORY_DOMAINS = {
    "europa.eu", "gov.uk", "sec.gov", "ftc.gov", "nist.gov",
    "ico.org.uk", "ofcom.org.uk",
}

# Content signals for authority type
_ANALYST_CONTENT_SIGNALS = [
    r"(gartner|forrester|mckinsey|bcg|deloitte|accenture|idc)\s",
    r"(magic\s+quadrant|wave\s+report|market\s+guide|hype\s+cycle)",
    r"(analyst|research\s+firm|industry\s+report|market\s+research)",
]

_NEWS_CONTENT_SIGNALS = [
    r"(according\s+to\s+(sources|reports|analysts))",
    r"(reported\s+by|press\s+release|newsroom)",
    r"(exclusive|breaking|reported\s+that)",
]

_REGULATORY_CONTENT_SIGNALS = [
    r"(regulation|directive|act\s+20\d{2}|compliance\s+requirement)",
    r"(European\s+Commission|Parliament|Congress|Federal\s+Trade)",
]


def classify_authority(
    url: str,
    text: str,
    company_domain: str = "",
) -> Dict[str, Any]:
    """
    Classify the authority type of a source.

    A source's authority affects how much weight its claims should carry:
    - first_party: Company's own website. Good for product details, weak for
      unverified claims like "market leader".
    - third_party_news: Independent journalism. Strong for events, announcements.
    - third_party_analyst: Analyst/consulting firms. Strong for market positioning.
    - regulatory: Government/regulatory. Strong for compliance, policy.
    - unknown: Can't determine.

    Returns:
        - authority_type: str
        - authority_score: float (0–1, higher = more authoritative for claims)
        - authority_reason: str
        - is_third_party: bool
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    text_lower = text.lower()

    # --- Domain-based classification ---
    # Check if it's the company's own domain
    company_base = company_domain.lower().replace("www.", "") if company_domain else ""
    if company_base and (domain == company_base or domain.endswith("." + company_base)):
        return {
            "authority_type": AUTHORITY_FIRST_PARTY,
            "authority_score": 0.5,
            "authority_reason": f"Company's own domain ({domain})",
            "is_third_party": False,
        }

    # Check known domains
    for known_domain in _ANALYST_DOMAINS:
        if domain == known_domain or domain.endswith("." + known_domain):
            return {
                "authority_type": AUTHORITY_THIRD_PARTY_ANALYST,
                "authority_score": 0.9,
                "authority_reason": f"Known analyst/consulting domain ({domain})",
                "is_third_party": True,
            }

    for known_domain in _NEWS_DOMAINS:
        if domain == known_domain or domain.endswith("." + known_domain):
            return {
                "authority_type": AUTHORITY_THIRD_PARTY_NEWS,
                "authority_score": 0.8,
                "authority_reason": f"Known news domain ({domain})",
                "is_third_party": True,
            }

    for known_domain in _REGULATORY_DOMAINS:
        if domain == known_domain or domain.endswith("." + known_domain):
            return {
                "authority_type": AUTHORITY_REGULATORY,
                "authority_score": 0.95,
                "authority_reason": f"Regulatory/government domain ({domain})",
                "is_third_party": True,
            }

    # --- Content-based fallback ---
    for pattern in _ANALYST_CONTENT_SIGNALS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return {
                "authority_type": AUTHORITY_THIRD_PARTY_ANALYST,
                "authority_score": 0.7,
                "authority_reason": "Analyst content signals detected in text",
                "is_third_party": True,
            }

    for pattern in _NEWS_CONTENT_SIGNALS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return {
                "authority_type": AUTHORITY_THIRD_PARTY_NEWS,
                "authority_score": 0.65,
                "authority_reason": "News content signals detected in text",
                "is_third_party": True,
            }

    for pattern in _REGULATORY_CONTENT_SIGNALS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return {
                "authority_type": AUTHORITY_REGULATORY,
                "authority_score": 0.75,
                "authority_reason": "Regulatory content signals detected in text",
                "is_third_party": True,
            }

    return {
        "authority_type": AUTHORITY_UNKNOWN,
        "authority_score": 0.4,
        "authority_reason": f"Unknown domain ({domain}), no authority signals",
        "is_third_party": domain != company_base,
    }


def calculate_authority_boost(
    evidences: List[Tuple[dict, str, float]],
    company_domain: str = "",
) -> Tuple[float, Dict[str, Dict]]:
    """
    Calculate confidence boost based on authority diversity.

    Having third-party validation alongside first-party claims is ideal.
    All first-party = lower authority boost.
    Mix of first + third party = highest boost.

    Returns:
        - boost: float (0 to +0.15)
        - per_source: dict of source_id → authority details
    """
    max_boost = float(os.getenv("VITELIS_AUTHORITY_BOOST_MAX", "0.15"))

    per_source: Dict[str, Dict] = {}
    authority_types: List[str] = []

    for meta, doc, _ in evidences:
        sid = meta.get("source_id", "")
        if sid in per_source:
            continue
        auth = classify_authority(meta.get("url", ""), doc, company_domain)
        per_source[sid] = auth
        authority_types.append(auth["authority_type"])

    if not authority_types:
        return 0.0, per_source

    has_first_party = AUTHORITY_FIRST_PARTY in authority_types
    has_third_party = any(
        t in authority_types
        for t in (AUTHORITY_THIRD_PARTY_NEWS, AUTHORITY_THIRD_PARTY_ANALYST, AUTHORITY_REGULATORY)
    )

    if has_first_party and has_third_party:
        # Best case: claims validated by external sources
        boost = max_boost
    elif has_third_party:
        # Only third-party (no company confirmation) — still good
        boost = max_boost * 0.7
    elif has_first_party:
        # Only self-reported — weakest
        boost = max_boost * 0.2
    else:
        boost = 0.0

    return round(boost, 3), per_source


# ============================================================================
# 5. CONTRADICTION DETECTION
# ============================================================================

# Contradiction signal patterns — pairs of opposing concepts
_CONTRADICTION_PAIRS = [
    # Scale contradictions
    (r"(large|extensive|significant|major|massive)\s+(team|workforce|department|investment|budget)",
     r"(small|limited|minimal|nascent|emerging)\s+(team|workforce|department|investment|budget)"),
    # Growth contradictions
    (r"(growing|increasing|expanding|accelerating)\s+(revenue|adoption|usage|deployment)",
     r"(declining|decreasing|shrinking|slowing)\s+(revenue|adoption|usage|deployment)"),
    # Maturity contradictions
    (r"(mature|established|proven|production.grade|enterprise.ready)",
     r"(early.stage|experimental|pilot|prototype|proof.of.concept|nascent)"),
    # Capability contradictions
    (r"(leading|best.in.class|state.of.the.art|advanced|sophisticated)",
     r"(lagging|behind|catching\s+up|basic|rudimentary)"),
    # Commitment contradictions
    (r"(committed|dedicated|priorit|strategic\s+focus|core\s+to)",
     r"(exploring|considering|evaluating|no\s+plans|deprioritiz)"),
]


def detect_contradictions(
    evidences: List[Tuple[dict, str, float]],
) -> Dict[str, Any]:
    """
    Detect contradictions between evidence from different sources.

    Process:
    1. Group evidence by source
    2. For each pair of sources, check if one matches the "positive" pattern
       while the other matches the "negative" pattern
    3. Flag contradictions with details

    Returns:
        - has_contradictions: bool
        - contradiction_count: int
        - contradictions: list of dicts with details
        - confidence_penalty: float (0 to -0.20)
    """
    # Group by source
    source_texts: Dict[str, str] = {}
    for meta, doc, _ in evidences:
        sid = meta.get("source_id", "")
        if not sid:
            continue
        source_texts.setdefault(sid, "")
        source_texts[sid] += " " + doc

    source_ids = list(source_texts.keys())
    contradictions: List[Dict[str, Any]] = []

    for i in range(len(source_ids)):
        for j in range(i + 1, len(source_ids)):
            sid_a, sid_b = source_ids[i], source_ids[j]
            text_a = source_texts[sid_a].lower()
            text_b = source_texts[sid_b].lower()

            for pos_pattern, neg_pattern in _CONTRADICTION_PAIRS:
                # Check A=positive, B=negative
                pos_match_a = re.search(pos_pattern, text_a, re.IGNORECASE)
                neg_match_b = re.search(neg_pattern, text_b, re.IGNORECASE)

                if pos_match_a and neg_match_b:
                    contradictions.append({
                        "source_a": sid_a,
                        "source_b": sid_b,
                        "claim_a": pos_match_a.group(0)[:150],
                        "claim_b": neg_match_b.group(0)[:150],
                        "type": "opposing_claims",
                    })

                # Check B=positive, A=negative
                pos_match_b = re.search(pos_pattern, text_b, re.IGNORECASE)
                neg_match_a = re.search(neg_pattern, text_a, re.IGNORECASE)

                if pos_match_b and neg_match_a:
                    contradictions.append({
                        "source_a": sid_b,
                        "source_b": sid_a,
                        "claim_a": pos_match_b.group(0)[:150],
                        "claim_b": neg_match_a.group(0)[:150],
                        "type": "opposing_claims",
                    })

    # Deduplicate (same source pair, same pattern)
    seen: Set[str] = set()
    unique_contradictions: List[Dict] = []
    for c in contradictions:
        key = f"{c['source_a']}|{c['source_b']}|{c['claim_a'][:50]}"
        if key not in seen:
            seen.add(key)
            unique_contradictions.append(c)

    # Penalty: each contradiction reduces confidence
    penalty = min(0.20, len(unique_contradictions) * 0.07)

    return {
        "has_contradictions": len(unique_contradictions) > 0,
        "contradiction_count": len(unique_contradictions),
        "contradictions": unique_contradictions[:10],
        "confidence_penalty": round(-penalty, 3),
    }


# ============================================================================
# UNIFIED SOURCE EVALUATION — combines all 5 systems
# ============================================================================

def evaluate_evidence_quality(
    evidences: List[Tuple[dict, str, float]],
    company_domain: str = "",
) -> Dict[str, Any]:
    """
    Run the full source evaluation pipeline on a set of evidence chunks.

    Combines all 5 evaluation systems and returns a unified quality assessment:
    1. Content-based tier analysis (already applied at fetch time)
    2. Semantic corroboration
    3. Freshness analysis
    4. Authority classification
    5. Contradiction detection

    Returns a comprehensive dict with all metrics and a final confidence_adjustment.
    """
    # 2. Semantic corroboration
    corroboration = detect_semantic_corroboration(evidences)

    # 3. Freshness
    freshness_boost, freshness_details = calculate_freshness_boost(evidences)

    # 4. Authority
    authority_boost, authority_details = calculate_authority_boost(evidences, company_domain)

    # 5. Contradictions
    contradictions = detect_contradictions(evidences)

    # --- Compute net confidence adjustment ---
    corroboration_max = float(os.getenv("VITELIS_CORROBORATION_BOOST_MAX", "0.15"))
    corr_boost = corroboration["corroboration_score"] * corroboration_max

    net_adjustment = round(
        corr_boost + freshness_boost + authority_boost + contradictions["confidence_penalty"],
        3,
    )

    return {
        "corroboration": corroboration,
        "freshness": {
            "boost": freshness_boost,
            "per_source": freshness_details,
        },
        "authority": {
            "boost": authority_boost,
            "per_source": authority_details,
        },
        "contradictions": contradictions,
        "net_confidence_adjustment": net_adjustment,
    }
