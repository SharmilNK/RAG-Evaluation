"""
Dynamic k-parameter tuning for retrieval based on KPI type and complexity.

Adjusts the number of chunks retrieved based on KPI characteristics to balance
quality and efficiency.
"""

import os
from typing import Dict


def determine_optimal_k(kpi, default_k: int = 6) -> int:
    """
    Determine optimal k parameter based on KPI type and characteristics.

    Args:
        kpi: KPIDefinition object with type, question, rubric
        default_k: Fallback k value

    Returns:
        Optimal k value for this KPI
    """
    # Check if dynamic k is enabled
    enabled = os.getenv("VITELIS_ENABLE_DYNAMIC_K", "false").lower() in {"1", "true"}

    if not enabled:
        return default_k

    kpi_type = getattr(kpi, "type", "").lower()
    question = getattr(kpi, "question", "").lower()
    rubric = getattr(kpi, "rubric", []) or []

    # KPI-type based rules
    if kpi_type == "rubric":
        # Rubric KPIs with 5-point scales need more context
        # Check complexity of rubric
        rubric_text = " ".join(rubric).lower() if rubric else ""
        rubric_length = len(rubric_text)

        if rubric_length > 500:  # Complex rubric
            return 10
        elif rubric_length > 200:  # Medium rubric
            return 8
        else:
            return 6

    elif kpi_type == "quant":
        # Quantitative KPIs (counts, mentions)
        # Check if it's a counting/mention type task
        if any(keyword in question for keyword in ["count", "mention", "how many", "number of"]):
            # Need broader search for counting tasks
            return 8
        else:
            return 6

    elif kpi_type == "recency":
        # Recency checks need more results to find recent dates
        return 12

    # Default
    return default_k


def retrieve_with_quality_threshold(
    collection,
    query: str,
    retrieve_fn,
    initial_k: int = 6,
    quality_threshold: float = 0.6,
    max_k: int = 15,
    increment: int = 3
) -> tuple:
    """
    Iteratively retrieve evidence until quality threshold is met.

    This is an optional advanced feature for adaptive retrieval.

    Args:
        collection: ChromaDB collection
        query: Search query
        retrieve_fn: Retrieval function to use (e.g., retrieve_evidence_weighted)
        initial_k: Starting k value
        quality_threshold: Minimum average similarity score required
        max_k: Maximum k to try
        increment: How much to increase k each iteration

    Returns:
        Tuple of (evidences, k_used)
    """
    current_k = initial_k

    while current_k <= max_k:
        evidences = retrieve_fn(collection, query, k=current_k)

        if not evidences:
            return evidences, current_k

        # Check quality - look at average score
        if len(evidences) >= 3:
            scores = [score for _, _, score in evidences]
            avg_score = sum(scores) / len(scores)

            if avg_score >= quality_threshold:
                # Quality threshold met
                return evidences, current_k

        # Try with more results
        current_k += increment

    # Return results at max_k
    evidences = retrieve_fn(collection, query, k=max_k)
    return evidences, max_k


def get_k_recommendation(kpi_type: str, complexity: str = "medium") -> Dict[str, int]:
    """
    Get k recommendations for different scenarios.

    Args:
        kpi_type: Type of KPI (rubric, quant, recency)
        complexity: Complexity level (low, medium, high)

    Returns:
        Dictionary with k recommendations
    """
    recommendations = {
        "rubric": {
            "low": 6,
            "medium": 8,
            "high": 10,
        },
        "quant": {
            "low": 6,
            "medium": 8,
            "high": 10,
        },
        "recency": {
            "low": 10,
            "medium": 12,
            "high": 15,
        },
    }

    kpi_recommendations = recommendations.get(kpi_type, recommendations["rubric"])
    return {
        "recommended_k": kpi_recommendations.get(complexity, 8),
        "min_k": 6,
        "max_k": 15,
    }
