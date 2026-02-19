from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Tuple

import chromadb
import numpy as np


class SimpleHashEmbeddingFunction:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, input: Iterable[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: str | List[str]) -> List[List[float]]:
        if isinstance(input, list):
            return self._embed(input)
        return self._embed([input])

    def _embed(self, input: Iterable[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in input:
            vector = np.zeros(self.dim, dtype=np.float32)
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest[:8], 16) % self.dim
                vector[idx] += 1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            embeddings.append(vector.tolist())
        return embeddings

    def name(self) -> str:
        return "simple-hash-embedding"

    def get_config(self) -> dict:
        return {"dim": self.dim}


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    chunks = []
    cursor = 0
    while cursor < len(text):
        chunk = text[cursor : cursor + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        cursor += max_chars
    return chunks


def build_collection(run_id: str):
    persist_dir = os.path.join(os.path.dirname(__file__), "data", f"chroma_{run_id}")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = SimpleHashEmbeddingFunction()
    collection = client.get_or_create_collection("sources", embedding_function=embedding_fn)
    return collection


def index_sources(collection, sources: List[dict]) -> None:
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

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)


def retrieve_evidence(collection, query: str, k: int = 6) -> List[Tuple[dict, str]]:
    if not query.strip():
        return []
    results = collection.query(query_texts=[query], n_results=k)
    evidences: List[Tuple[dict, str]] = []
    for metadata, document in zip(results.get("metadatas", [[]])[0], results.get("documents", [[]])[0]):
        evidences.append((metadata, document))
    return evidences
