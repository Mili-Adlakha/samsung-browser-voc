"""Review chunking and OpenAI embedding helpers."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from models.schemas import ReviewRecord

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100
SINGLE_CHUNK_MAX_LEN = 500
MULTI_CHUNK_MAX_LEN = 450

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file before embedding."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


def chunk_review(review: ReviewRecord) -> list[dict]:
    """Split a review into chunks with embedding prefixes and metadata."""
    text = review.review_text.strip()
    if len(text) <= SINGLE_CHUNK_MAX_LEN:
        parts = [text]
    else:
        parts = _split_at_sentences(text, MULTI_CHUNK_MAX_LEN)

    chunks: list[dict] = []
    for index, chunk_text in enumerate(parts):
        chunks.append(
            {
                "chunk_id": f"{review.review_id}_chunk_{index}",
                "text": chunk_text,
                "embedding_text": (
                    f"[Rating:{review.rating}/5][Upvotes:{review.thumbs_up_count}]"
                    f"[Version:{review.app_version}] {chunk_text}"
                ),
                "metadata": {
                    "review_id": review.review_id,
                    "app_version": review.app_version,
                    "author": review.author_name,
                    "rating": review.rating,
                    "upvotes": review.thumbs_up_count,
                    "date": review.review_date,
                    "chunk_index": index,
                },
            }
        )
    return chunks


def _split_at_sentences(text: str, max_len: int) -> list[str]:
    """Split long text on sentence boundaries; each chunk at most max_len characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text[:max_len]]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_len:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_by_words(sentence, max_len))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks or [text[:max_len]]


def _split_by_words(text: str, max_len: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_len:
            current.append(word)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word] if len(word) <= max_len else []
            if len(word) > max_len:
                for i in range(0, len(word), max_len):
                    chunks.append(word[i : i + max_len])
    if current:
        chunks.append(" ".join(current))
    return chunks


async def get_embedding(text: str) -> list[float]:
    """Embed a single string using OpenAI text-embedding-3-small."""
    embeddings = await get_embeddings([text])
    return embeddings[0]


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed up to 100 texts per API request; returns vectors in input order."""
    if not texts:
        return []

    client = _get_client()
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        all_embeddings.extend(item.embedding for item in ordered)

    return all_embeddings
