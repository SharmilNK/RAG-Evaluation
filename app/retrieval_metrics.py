"""
Retrieval quality metrics for the Vitelis KPI benchmarking pipeline.

Computes three standard IR metrics for each KPI retrieval call, comparing
the returned chunk IDs against a golden (ground-truth) set loaded from YAML:

  - Hit Rate  : fraction of golden chunks that appear anywhere in top-k results
  - MRR       : reciprocal rank of the first golden chunk in the ranked list
  - nDCG      : normalised discounted cumulative gain using per-chunk relevance labels

Golden chunks YAML path is read from the GOLDEN_CHUNKS_PATH environment variable.
If the variable is unset or the file is missing/malformed, all three metrics
return None silently — the rest of the pipeline is unaffected.

Expected YAML format
--------------------
    kpi_001:
      - chunk_id: "abc123::chunk_0"
        relevance: 2          # optional; defaults to 1 if omitted
      - chunk_id: "def456::chunk_2"
        relevance: 1
    kpi_002:
      - chunk_id: "ghi789::chunk_1"
        relevance: 1

The chunk_id values must match the IDs stored in ChromaDB, which follow the
pattern  "{source_id}::chunk_{idx}"  (see vectorstore.index_sources).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # already in requirements.txt (pyyaml)

# Module-level cache so the YAML file is read at most once per process.
_golden_cache: Optional[Dict[str, Any]] = None
_golden_load_attempted: bool = False


def _load_golden_chunks() -> Dict[str, Any]:
    """
    Load the golden-chunks YAML file and cache the result.

    Returns:
        Dict mapping kpi_id -> list of {chunk_id, relevance} dicts.
        Returns an empty dict if GOLDEN_CHUNKS_PATH is unset, the file is
        absent, or parsing fails.

    Side effects:
        Sets the module-level cache on first call.
    """
    global _golden_cache, _golden_load_attempted

    if _golden_load_attempted:
        return _golden_cache or {}

    _golden_load_attempted = True
    path = (os.getenv("GOLDEN_CHUNKS_PATH") or "").strip()
    if not path:
        # Default to repo-level golden_chunks.yaml so the feature works out-of-box.
        # Expected layout: <repo>/app/retrieval_metrics.py → <repo>/golden_chunks.yaml
        try:
            repo_root = Path(__file__).resolve().parents[1]
            candidate = repo_root / "golden_chunks.yaml"
            if candidate.exists():
                path = str(candidate)
        except Exception:
            path = ""
    if not path:
        _golden_cache = {}
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _golden_cache = data
        return data
    except Exception:
        _golden_cache = {}
        return {}


def get_golden_ids_for_kpi(kpi_id: str) -> Optional[Dict[str, float]]:
    """
    Return the golden chunk relevance map for a KPI.

    Args:
        kpi_id: KPI identifier used as the YAML top-level key.

    Returns:
        Dict mapping chunk_id -> relevance score, or None if the KPI
        has no golden set defined.
    """
    data = _load_golden_chunks()
    entries = data.get(kpi_id, [])
    if not entries:
        return None
    return {
        e["chunk_id"]: float(e.get("relevance", 1))
        for e in entries
        if "chunk_id" in e
    }


def compute_hit_rate(kpi_id: str, retrieved_ids: List[str]) -> Optional[float]:
    """
    Hit Rate: fraction of golden chunks that appear anywhere in the top-k list.

    A hit is counted for each unique golden chunk that appears at any rank
    in the retrieved list. The denominator is the total number of golden
    chunks for this KPI.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_ids: Ordered list of chunk IDs returned by retrieval
            (index 0 = rank 1).

    Returns:
        Float in [0.0, 1.0], or None if no golden set is available.
    """
    golden = get_golden_ids_for_kpi(kpi_id)
    if golden is None:
        return None

    retrieved_set = set(retrieved_ids)
    hits = sum(1 for cid in golden if cid in retrieved_set)
    return round(hits / len(golden), 4)


def compute_mrr(kpi_id: str, retrieved_ids: List[str]) -> Optional[float]:
    """
    MRR: reciprocal rank of the *first* golden chunk in the ranked list.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_ids: Ordered list of chunk IDs (index 0 = rank 1).

    Returns:
        Float in (0.0, 1.0], 0.0 if no golden chunk appears in the list,
        or None if no golden set is available.
    """
    golden = get_golden_ids_for_kpi(kpi_id)
    if golden is None:
        return None

    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in golden:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg(kpi_id: str, retrieved_ids: List[str]) -> Optional[float]:
    """
    nDCG: normalised discounted cumulative gain using golden relevance labels.

    The ideal ranking places all golden chunks (sorted by descending relevance)
    at the top. Retrieved chunks not in the golden set contribute 0 relevance.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_ids: Ordered list of chunk IDs (index 0 = rank 1).

    Returns:
        Float in [0.0, 1.0], or None if no golden set is available.
    """
    golden = get_golden_ids_for_kpi(kpi_id)
    if golden is None:
        return None

    # DCG of the actual retrieved ranking
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        rel = golden.get(cid, 0.0)
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    # Ideal DCG: golden chunks sorted by relevance desc, placed at ranks 1..n
    ideal_rels = sorted(golden.values(), reverse=True)
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, start=1)
        if rel > 0
    )

    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


def compute_all_retrieval_metrics(
    kpi_id: str,
    retrieved_evidences: List[Any],
) -> Dict[str, Optional[float]]:
    """
    Compute Hit Rate, MRR, and nDCG in a single call.

    Extracts chunk IDs from a list of (metadata, document, score) tuples
    as returned by vectorstore.retrieve_evidence / reranker.rerank.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_evidences: List of (metadata_dict, doc_str, score_float)
            tuples in retrieval rank order.

    Returns:
        Dict with keys "hit_rate", "mrr", "ndcg", each a float or None.
    """
    # Build the ordered list of chunk IDs.
    # ChromaDB stores individual chunk IDs in the query result "ids" field,
    # but retrieve_evidence returns (metadata, doc, score) without the raw ID.
    # The chunk ID follows the pattern "{source_id}::chunk_{idx}"; we
    # reconstruct it from metadata where possible, falling back to source_id.
    chunk_ids: List[str] = []
    for item in retrieved_evidences:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        meta = item[0] if isinstance(item[0], dict) else {}
        # Prefer a stored chunk_id field; fall back to source_id
        cid = meta.get("chunk_id") or meta.get("source_id", "")
        chunk_ids.append(str(cid))

    return {
        "hit_rate": compute_hit_rate(kpi_id, chunk_ids),
        "mrr": compute_mrr(kpi_id, chunk_ids),
        "ndcg": compute_ndcg(kpi_id, chunk_ids),
    }
