"""Tests for review chunking (no OpenAI calls)."""

from datetime import datetime, timezone

from models.schemas import ReviewRecord
from services.embedder import (
    MULTI_CHUNK_MAX_LEN,
    chunk_review,
    _split_at_sentences,
)


def _review(text: str, review_id: str = "v30.XX_abc12345_0") -> ReviewRecord:
    return ReviewRecord(
        review_id=review_id,
        app_version="30.XX",
        author_name="Test User",
        rating=4,
        review_text=text,
        thumbs_up_count=12,
        review_date="2026-05-22",
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )


def test_short_review_single_chunk():
    text = "Great browser, fast and clean. Ad blocking works perfectly."
    chunks = chunk_review(_review(text))
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "v30.XX_abc12345_0_chunk_0"
    assert chunks[0]["text"] == text


def test_embedding_text_prefix():
    text = "Great browser, fast and clean. Ad blocking works."
    chunks = chunk_review(_review(text))
    assert chunks[0]["embedding_text"].startswith(
        "[Rating:4/5][Upvotes:12][Version:30.XX] "
    )
    assert chunks[0]["embedding_text"].endswith(text)


def test_metadata_fields():
    text = "Great browser, fast and clean. Ad blocking works."
    meta = chunk_review(_review(text))[0]["metadata"]
    assert meta["review_id"] == "v30.XX_abc12345_0"
    assert meta["app_version"] == "30.XX"
    assert meta["author"] == "Test User"
    assert meta["rating"] == 4
    assert meta["upvotes"] == 12
    assert meta["date"] == "2026-05-22"
    assert meta["chunk_index"] == 0


def test_long_review_multiple_chunks():
    sentence = "This is a detailed sentence about browser performance. "
    text = sentence * 20
    assert len(text) > 500
    chunks = chunk_review(_review(text))
    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk["metadata"]["chunk_index"] == index
        assert len(chunk["text"]) <= MULTI_CHUNK_MAX_LEN


def test_split_at_sentences_respects_max_length():
    text = "First sentence here. " + ("Second part grows. " * 40)
    parts = _split_at_sentences(text, MULTI_CHUNK_MAX_LEN)
    assert len(parts) > 1
    assert all(len(part) <= MULTI_CHUNK_MAX_LEN for part in parts)


def test_chunk_ids_are_unique_per_review():
    text = ("Long review sentence. " * 60).strip()
    chunks = chunk_review(_review(text))
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
