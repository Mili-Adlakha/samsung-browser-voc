"""Tests for ChromaDB vector store (no OpenAI calls)."""

import tempfile

import pytest

from services.vector_store import VocVectorStore

DIM = 1536


def _fake_embedding(seed: float) -> list[float]:
    return [seed] * DIM


def _sample_chunk(chunk_id: str, text: str, upvotes: int, version: str = "30.XX") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding_text": f"[Rating:4/5][Upvotes:{upvotes}][Version:{version}] {text}",
        "metadata": {
            "review_id": f"review_{chunk_id}",
            "app_version": version,
            "author": "Author",
            "rating": 4,
            "upvotes": upvotes,
            "date": "2026-05-22",
            "chunk_index": 0,
        },
    }


@pytest.fixture
def store(tmp_path):
    vs = VocVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_voc",
    )
    yield vs
    vs.reset()


def test_upsert_and_count(store):
    chunks = [
        _sample_chunk(
            "c1",
            "Great browser, fast and clean. Ad blocking works perfectly.",
            upvotes=3,
        )
    ]
    stored = store.upsert_chunks(chunks, [_fake_embedding(0.1)])
    assert stored == 1
    assert store.count() == 1


def test_upsert_does_not_duplicate_on_reingest(store):
    chunk = _sample_chunk(
        "c1",
        "Great browser, fast and clean. Ad blocking works perfectly.",
        upvotes=3,
    )
    store.upsert_chunks([chunk], [_fake_embedding(0.1)])
    store.upsert_chunks([chunk], [_fake_embedding(0.2)])
    assert store.count() == 1


def test_query_returns_text_metadata_distance(store):
    chunks = [
        _sample_chunk(
            "c1",
            "Tab sync lost all my open tabs after the update.",
            upvotes=50,
        ),
        _sample_chunk(
            "c2",
            "Great browser, fast and clean. Ad blocking works perfectly.",
            upvotes=3,
        ),
    ]
    store.upsert_chunks(chunks, [_fake_embedding(1.0), _fake_embedding(2.0)])

    results = store.query(_fake_embedding(1.0), top_k=2)
    assert len(results) == 2
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "distance" in results[0]
    assert results[0]["metadata"]["upvotes"] in (3, 50)


def test_query_with_version_filter(store):
    chunks = [
        _sample_chunk(
            "c1",
            "Tab sync lost all my open tabs after the update.",
            upvotes=10,
            version="30.XX",
        ),
        _sample_chunk(
            "c2",
            "Older version review about crashes and freezes daily.",
            upvotes=5,
            version="29.XX",
        ),
    ]
    store.upsert_chunks(chunks, [_fake_embedding(1.0), _fake_embedding(2.0)])

    filtered = store.query(
        _fake_embedding(1.0),
        top_k=10,
        where={"app_version": {"$eq": "30.XX"}},
    )
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["app_version"] == "30.XX"


def test_get_all_sorted_by_upvotes(store):
    chunks = [
        _sample_chunk(
            "low",
            "Minor UI complaint about spacing in the toolbar area.",
            upvotes=2,
        ),
        _sample_chunk(
            "high",
            "Tab sync lost all my open tabs after the latest update.",
            upvotes=99,
        ),
        _sample_chunk(
            "mid",
            "Password autofill stopped working after the last release.",
            upvotes=20,
        ),
    ]
    store.upsert_chunks(
        chunks,
        [_fake_embedding(1.0), _fake_embedding(2.0), _fake_embedding(3.0)],
    )

    results = store.get_all(top_k=3)
    upvotes = [item["metadata"]["upvotes"] for item in results]
    assert upvotes == [99, 20, 2]


def test_reset_clears_collection(store):
    chunk = _sample_chunk(
        "c1",
        "Great browser, fast and clean. Ad blocking works perfectly.",
        upvotes=1,
    )
    store.upsert_chunks([chunk], [_fake_embedding(0.5)])
    assert store.count() == 1
    store.reset()
    assert store.count() == 0
