"""Tests for POST /api/v1/ingest."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from services.vector_store import VocVectorStore
from tests.conftest import APP_VERSION, EMBEDDING_DIM, FORMAT_A_SAMPLE


@pytest.fixture
def chroma_dir(chroma_env, monkeypatch):
    monkeypatch.setenv("COLLECTION_NAME", "test_ingest")
    return chroma_env


@pytest.fixture
def client(chroma_dir, mock_embeddings):
    with TestClient(app) as test_client:
        yield test_client
    store = VocVectorStore(
        persist_dir=str(chroma_dir),
        collection_name="test_ingest",
    )
    store.reset()


def test_ingest_format_a_success(client):
    response = client.post(
        "/api/v1/ingest",
        json={
            "reviews": FORMAT_A_SAMPLE,
            "app_version": APP_VERSION,
            "date_range": "20-25 May 2026",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["reviews_parsed"] == 2
    assert data["chunks_stored"] == 2
    assert data["avg_rating"] == 3.0
    assert len(data["top_upvoted"]) == 2
    assert data["top_upvoted"][0]["upvotes"] == 74


def test_ingest_empty_reviews_validation_error(client):
    response = client.post(
        "/api/v1/ingest",
        json={"reviews": "", "app_version": "30.XX"},
    )
    assert response.status_code == 422


def test_ingest_unparseable_text_returns_422(client):
    response = client.post(
        "/api/v1/ingest",
        json={
            "reviews": "random notes without star ratings or structure",
            "app_version": "30.XX",
        },
    )
    assert response.status_code == 422
    assert "No reviews could be parsed" in response.json()["detail"]


def test_ingest_stores_app_version_in_metadata(client, chroma_dir):
    client.post(
        "/api/v1/ingest",
        json={"reviews": FORMAT_A_SAMPLE, "app_version": "30.XX"},
    )

    store = VocVectorStore(
        persist_dir=str(chroma_dir),
        collection_name="test_ingest",
    )
    chunks = store.get_all(top_k=10)
    assert len(chunks) == 2
    assert all(chunk["metadata"]["app_version"] == "30.XX" for chunk in chunks)


def test_ingest_reingest_does_not_duplicate_chunks(client):
    payload = {"reviews": FORMAT_A_SAMPLE, "app_version": "30.XX"}
    first = client.post("/api/v1/ingest", json=payload)
    second = client.post("/api/v1/ingest", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["chunks_stored"] == 2
    assert second.json()["chunks_stored"] == 2


def test_ingest_embedding_failure_returns_500(client, chroma_dir):
    with patch(
        "routers.ingest.get_embeddings",
        new=AsyncMock(side_effect=RuntimeError("OpenAI unavailable")),
    ):
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/ingest",
                json={"reviews": FORMAT_A_SAMPLE, "app_version": "30.XX"},
            )
    assert response.status_code == 500
    assert response.json()["error"] == "Embedding service unavailable"
    assert "OpenAI unavailable" in response.json()["detail"]
