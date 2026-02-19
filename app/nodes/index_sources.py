from __future__ import annotations

from typing import Dict, List

from app.vectorstore import build_collection, index_sources


def index_sources_node(state: Dict) -> Dict:
    run_id = state["run_id"]
    sources: List[Dict] = state.get("sources", [])

    collection = build_collection(run_id)
    index_sources(collection, sources)

    return {"collection_id": f"collection_{run_id}"}
