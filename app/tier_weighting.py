"""
Tier-based source weighting for enhanced retrieval quality.

Applies post-retrieval boosting based on source tier metadata to prioritize
high-credibility sources (investor reports, press releases) over lower-tier content.
"""
import os
from typing import List, Tuple, Dict, Any


def get_tier_boost_config() -> Dict[str, float]:
    """Get tier boost configuration from environment variables."""
    return {
        "tier1": float(os.getenv("VITELIS_TIER1_BOOST", "1.5")),
        "tier2": float(os.getenv("VITELIS_TIER2_BOOST", "1.2")),
        "tier3": 1.0,  # Baseline
        "oversampling_factor": float(os.getenv("VITELIS_TIER_OVERSAMPLING_FACTOR", "2")),
    }


def retrieve_evidence_weighted(
    collection,
    query: str,
    k: int = 6,
    apply_tier_boost: bool = True
) -> List[Tuple[dict, str, float]]:
    """
    Retrieve evidence chunks with tier-based relevance boosting.

    Args:
        collection: ChromaDB collection
        query: Search query string
        k: Number of results to return
        apply_tier_boost: Whether to apply tier-based boosting

    Returns:
        List of (metadata, document, weighted_score) tuples
    """
    if not query.strip():
        return []

    config = get_tier_boost_config()

    # Over-fetch to allow for re-ranking
    fetch_k = int(k * config["oversampling_factor"]) if apply_tier_boost else k

    # Query ChromaDB
    results = collection.query(query_texts=[query], n_results=fetch_k)

    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not metadatas:
        return []

    # Apply tier boosting to scores
    weighted_results = []
    for metadata, document, distance in zip(metadatas, documents, distances):
        tier = metadata.get("tier", 3)

        # Convert distance to similarity score (lower distance = higher similarity)
        # Normalize to [0, 1] range assuming max distance ~2.0 for normalized vectors
        similarity = max(0.0, 1.0 - (distance / 2.0))

        if apply_tier_boost:
            # Apply tier boost
            if tier == 1:
                boost = config["tier1"]
            elif tier == 2:
                boost = config["tier2"]
            else:
                boost = config["tier3"]

            weighted_score = similarity * boost
        else:
            weighted_score = similarity

        weighted_results.append((metadata, document, weighted_score))

    # Sort by weighted score (descending) and return top k
    weighted_results.sort(key=lambda x: x[2], reverse=True)
    return weighted_results[:k]


def calculate_tier_quality(evidences: List[Tuple[dict, str, float]]) -> float:
    """
    Calculate confidence boost based on source tier quality.

    Args:
        evidences: List of (metadata, document, score) tuples

    Returns:
        Confidence boost value (0.0 to 0.15)
    """
    if not evidences:
        return 0.0

    max_boost = float(os.getenv("VITELIS_TIER_BOOST_MAX", "0.15"))

    # Calculate average tier (lower tier number = higher quality)
    tiers = [meta.get("tier", 3) for meta, _, _ in evidences]
    avg_tier = sum(tiers) / len(tiers)

    # Convert to boost: tier 1 avg = max boost, tier 3 avg = 0 boost
    # Formula: max_boost - (avg_tier - 1) * (max_boost / 2)
    tier_boost = max(0.0, max_boost - (avg_tier - 1.0) * (max_boost / 2.0))

    return round(tier_boost, 3)


def get_tier_distribution(evidences: List[Tuple[dict, str, float]]) -> Dict[str, Any]:
    """
    Get distribution statistics of tiers in evidence set.

    Args:
        evidences: List of (metadata, document, score) tuples

    Returns:
        Dictionary with tier counts and average
    """
    if not evidences:
        return {"tier1": 0, "tier2": 0, "tier3": 0, "avg": 0.0}

    tiers = [meta.get("tier", 3) for meta, _, _ in evidences]

    tier_counts = {
        "tier1": tiers.count(1),
        "tier2": tiers.count(2),
        "tier3": tiers.count(3),
    }

    tier_counts["avg"] = round(sum(tiers) / len(tiers), 2)

    return tier_counts
