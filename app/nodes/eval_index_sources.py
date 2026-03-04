from __future__ import annotations

from typing import Dict, List

from app.vectorstore import build_collection, chunk_text

# ChromaDB's default max batch size — stay safely below it.
_CHROMA_BATCH = 4000


def eval_index_sources_node(state: Dict) -> Dict:
    """Identical to index_sources_node but upserts in batches to avoid ChromaDB
    batch-size limits when sources contain very long content."""
    run_id = state["run_id"]
    sources: List[Dict] = state.get("sources", [])

    collection = build_collection(run_id)

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[dict] = []

    for source in sources:
        chunks = chunk_text(source["text"])
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{source['source_id']}::chunk_{idx}"
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append(
                {
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "title": source.get("title", ""),
                    "domain": source.get("domain", ""),
                    "retrieved_at": source.get("retrieved_at", ""),
                    "tier": source["tier"],
                }
            )

    for start in range(0, len(ids), _CHROMA_BATCH):
        batch_ids = ids[start : start + _CHROMA_BATCH]
        batch_docs = docs[start : start + _CHROMA_BATCH]
        batch_meta = metadatas[start : start + _CHROMA_BATCH]
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)

    return {"collection_id": f"collection_{run_id}"}
