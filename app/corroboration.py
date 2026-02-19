"""
Cross-source corroboration detection for evidence validation.

Detects when multiple independent sources provide supporting evidence for a claim,
increasing confidence in KPI assessments.
"""

import os
from typing import List, Dict, Tuple, Set
from collections import Counter


def extract_keywords(text: str, min_length: int = 4) -> Set[str]:
    """
    Extract meaningful keywords from text.

    Args:
        text: Input text string
        min_length: Minimum word length to consider

    Returns:
        Set of lowercase keywords
    """
    # Simple tokenization and filtering
    words = text.lower().split()

    # Filter out common stop words and short words
    stop_words = {
        "the", "and", "for", "are", "with", "this", "that", "from", "have",
        "has", "been", "will", "would", "could", "should", "about", "into",
        "through", "over", "after", "before", "between", "under", "also"
    }

    keywords = {
        word.strip(".,!?:;\"'()[]{}")
        for word in words
        if len(word) >= min_length and word not in stop_words
    }

    return keywords


def calculate_source_diversity(evidences: List[Tuple[dict, str, float]]) -> int:
    """
    Count number of unique source_ids in evidence set.

    Args:
        evidences: List of (metadata, document, score) tuples

    Returns:
        Count of unique sources
    """
    source_ids = {meta.get("source_id", "") for meta, _, _ in evidences if meta.get("source_id")}
    return len(source_ids)


def find_keyword_overlap(evidences: List[Tuple[dict, str, float]], min_sources: int = 2) -> List[str]:
    """
    Find keywords that appear in multiple sources.

    Args:
        evidences: List of (metadata, document, score) tuples
        min_sources: Minimum number of sources for a keyword to be considered shared

    Returns:
        List of keywords appearing in min_sources or more different sources
    """
    # Group keywords by source
    source_keywords: Dict[str, Set[str]] = {}

    for meta, doc, _ in evidences:
        source_id = meta.get("source_id", "")
        if not source_id:
            continue

        keywords = extract_keywords(doc)

        if source_id not in source_keywords:
            source_keywords[source_id] = set()
        source_keywords[source_id].update(keywords)

    # Count how many sources each keyword appears in
    keyword_source_count: Dict[str, int] = {}

    for keywords in source_keywords.values():
        for keyword in keywords:
            keyword_source_count[keyword] = keyword_source_count.get(keyword, 0) + 1

    # Return keywords appearing in multiple sources
    shared_keywords = [
        keyword for keyword, count in keyword_source_count.items()
        if count >= min_sources
    ]

    return shared_keywords


def detect_corroboration(
    evidences: List[Tuple[dict, str, float]],
    min_sources: int = 2
) -> float:
    """
    Detect cross-source corroboration in evidence set.

    This function analyzes:
    1. Source diversity (multiple independent sources)
    2. Keyword overlap (similar concepts across sources)

    Args:
        evidences: List of (metadata, document, score) tuples
        min_sources: Minimum sources required for corroboration

    Returns:
        Corroboration score from 0.0 (no corroboration) to 1.0 (strong corroboration)
    """
    if len(evidences) < min_sources:
        return 0.0

    # Calculate source diversity
    unique_sources = calculate_source_diversity(evidences)

    if unique_sources < min_sources:
        # All evidence from same source = no corroboration
        return 0.0

    # Find shared keywords across sources
    shared_keywords = find_keyword_overlap(evidences, min_sources=min_sources)

    # Calculate corroboration score
    # Components:
    # 1. Source diversity score: more sources = higher score
    # 2. Keyword overlap score: more shared keywords = higher score

    # Source diversity component (0.0 - 0.5)
    # 2 sources = 0.25, 3 sources = 0.35, 4+ sources = 0.5
    diversity_score = min(0.5, 0.15 + (unique_sources * 0.1))

    # Keyword overlap component (0.0 - 0.5)
    # Scale based on number of shared keywords (5+ keywords = max score)
    overlap_score = min(0.5, len(shared_keywords) * 0.1)

    total_score = diversity_score + overlap_score

    return round(min(1.0, total_score), 2)


def get_corroboration_details(
    evidences: List[Tuple[dict, str, float]],
    min_sources: int = 2
) -> Dict[str, object]:
    """
    Get detailed corroboration metrics for debugging/reporting.

    Args:
        evidences: List of (metadata, document, score) tuples
        min_sources: Minimum sources for corroboration

    Returns:
        Dictionary with detailed corroboration metrics
    """
    unique_sources = calculate_source_diversity(evidences)
    shared_keywords = find_keyword_overlap(evidences, min_sources=min_sources)
    corroboration_score = detect_corroboration(evidences, min_sources=min_sources)

    return {
        "unique_sources": unique_sources,
        "shared_keywords_count": len(shared_keywords),
        "shared_keywords": shared_keywords[:10],  # First 10 for brevity
        "corroboration_score": corroboration_score,
        "has_corroboration": corroboration_score >= 0.3,
    }
