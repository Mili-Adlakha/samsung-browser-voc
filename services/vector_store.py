"""ChromaDB persistence wrapper for VOC review chunks."""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class VocVectorStore:
    """Local persistent vector store for review chunks."""

    def __init__(self, persist_dir: str, collection_name: str):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name
        )

    @classmethod
    def from_env(cls) -> VocVectorStore:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        collection_name = os.getenv("COLLECTION_NAME", "samsung_browser_voc")
        return cls(persist_dir=persist_dir, collection_name=collection_name)

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [_normalize_metadata(chunk["metadata"]) for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return _format_query_results(results)

    def get_all(self, top_k: int = 100) -> list[dict]:
        if self.count() == 0:
            return []

        results = self._collection.get(
            include=["documents", "metadatas"],
        )

        items: list[dict] = []
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        for doc_id, document, metadata in zip(ids, documents, metadatas):
            meta = metadata or {}
            items.append(
                {
                    "id": doc_id,
                    "text": document or "",
                    "metadata": meta,
                    "distance": None,
                }
            )

        items.sort(
            key=lambda item: int((item.get("metadata") or {}).get("upvotes", 0)),
            reverse=True,
        )
        return items[:top_k]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )


def _normalize_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
    """Ensure metadata values are Chroma-compatible scalars."""
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def _format_query_results(results: dict) -> list[dict]:
    formatted: list[dict] = []
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for doc_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        formatted.append(
            {
                "id": doc_id,
                "text": document or "",
                "metadata": metadata or {},
                "distance": distance,
            }
        )
    return formatted
