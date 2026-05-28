"""Semantic retrieval and context formatting for RAG chat."""

from __future__ import annotations

from services.embedder import get_embedding
from services.vector_store import VocVectorStore

MAX_CONTEXT_CHUNKS = 15
HIGH_UPVOTE_THRESHOLD = 10


class VocRetriever:
    """Retrieve review chunks and build LLM context strings."""

    def __init__(self, vector_store: VocVectorStore | None = None):
        self._store = vector_store or VocVectorStore.from_env()

    async def retrieve(
        self,
        question: str,
        top_k: int = 20,
        version_filter: str = "",
    ) -> tuple[list[dict], dict]:
        query_embedding = await get_embedding(question)

        where = None
        if version_filter.strip():
            where = {"app_version": {"$eq": version_filter.strip()}}

        results = self._store.query(query_embedding, top_k=top_k, where=where)
        results.sort(
            key=lambda item: (
                item.get("distance")
                if item.get("distance") is not None
                else float("inf"),
                -int((item.get("metadata") or {}).get("upvotes", 0)),
            )
        )

        stats = _compute_stats(results)
        return results, stats

    def build_context(self, chunks: list[dict]) -> str:
        sections: list[str] = []
        for index, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
            metadata = chunk.get("metadata") or {}
            rating = metadata.get("rating", 0)
            upvotes = metadata.get("upvotes", 0)
            version = metadata.get("app_version", "")
            date = metadata.get("date", "")
            text = chunk.get("text", "")

            sections.append(
                f"[Review {index}] Rating: {rating}/5 | Upvotes: {upvotes} | "
                f"Version: {version} | Date: {date}\n{text}\n---"
            )
        return "\n".join(sections)


def _compute_stats(results: list[dict]) -> dict:
    if not results:
        return {
            "total": 0,
            "avg_rating": 0.0,
            "high_upvote_count": 0,
        }

    ratings: list[float] = []
    high_upvote_count = 0
    for item in results:
        metadata = item.get("metadata") or {}
        ratings.append(float(metadata.get("rating", 0)))
        if int(metadata.get("upvotes", 0)) >= HIGH_UPVOTE_THRESHOLD:
            high_upvote_count += 1

    return {
        "total": len(results),
        "avg_rating": sum(ratings) / len(ratings),
        "high_upvote_count": high_upvote_count,
    }
