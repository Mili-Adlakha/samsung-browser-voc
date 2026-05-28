"""Tests for POST /api/v1/chat."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from services.retriever import VocRetriever
from services.vector_store import VocVectorStore
from tests.conftest import EMBEDDING_DIM, FORMAT_A_SAMPLE

CHAT_MODEL = "claude-sonnet-4-5"


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    path = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(path))
    monkeypatch.setenv("COLLECTION_NAME", "test_chat")
    return path


@pytest.fixture
def seeded_store(chroma_dir):
    with patch(
        "routers.ingest.get_embeddings",
        new=AsyncMock(
            side_effect=lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts]
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/ingest",
                json={"reviews": FORMAT_A_SAMPLE, "app_version": "30.XX"},
            )
            assert response.status_code == 200
    return chroma_dir


@pytest.fixture
def client(seeded_store):
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value="**Finding:** Tab sync is a top issue.")

    with patch(
        "services.retriever.get_embedding",
        new=AsyncMock(return_value=[0.2] * EMBEDDING_DIM),
    ):
        with patch("routers.chat.get_llm_client", return_value=mock_llm):
            with TestClient(app) as test_client:
                yield test_client, mock_llm


def test_chat_no_data_returns_400(chroma_dir):
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/chat",
            json={"question": "What are the top complaints?"},
        )
    assert response.status_code == 400
    assert "Ingest reviews first" in response.json()["detail"]


def test_chat_empty_question_returns_422(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/chat",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_chat_success_returns_answer(client):
    test_client, mock_llm = client
    response = test_client.post(
        "/api/v1/chat",
        json={"question": "What are the top complaints about tab sync?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What are the top complaints about tab sync?"
    assert "Finding" in data["answer"]
    assert data["retrieved_chunks"] > 0
    assert data["model"] == CHAT_MODEL
    assert "timestamp" in data
    mock_llm.chat.assert_awaited_once()


def test_chat_llm_failure_returns_503(seeded_store):
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(side_effect=RuntimeError("Anthropic unavailable"))

    with patch(
        "services.retriever.get_embedding",
        new=AsyncMock(return_value=[0.2] * EMBEDDING_DIM),
    ):
        with patch("routers.chat.get_llm_client", return_value=mock_llm):
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/api/v1/chat",
                    json={"question": "What are the top complaints?"},
                )
    assert response.status_code == 503
    assert response.json()["error"] == "LLM unavailable"


@pytest.mark.asyncio
async def test_retriever_version_filter(seeded_store):
    store = VocVectorStore(
        persist_dir=str(seeded_store),
        collection_name="test_chat",
    )
    extra_chunk = {
        "chunk_id": "v29.XX_extra_0",
        "text": "Older version had crashes and freezes on startup daily.",
        "metadata": {
            "review_id": "v29.XX_extra_0",
            "app_version": "29.XX",
            "author": "Legacy User",
            "rating": 2,
            "upvotes": 8,
            "date": "2026-04-01",
            "chunk_index": 0,
        },
    }
    store.upsert_chunks([extra_chunk], [[0.9] * EMBEDDING_DIM])

    retriever = VocRetriever(store)

    with patch(
        "services.retriever.get_embedding",
        new=AsyncMock(return_value=[0.2] * EMBEDDING_DIM),
    ):
        results, _ = await retriever.retrieve(
            "crashes and freezes",
            top_k=10,
            version_filter="29.XX",
        )

    assert len(results) == 1
    assert results[0]["metadata"]["app_version"] == "29.XX"


@pytest.mark.asyncio
async def test_build_context_limits_to_fifteen_chunks():
    chunks = [
        {
            "text": f"Review body number {index} with enough length here.",
            "metadata": {
                "rating": 3,
                "upvotes": index,
                "app_version": "30.XX",
                "date": "2026-05-22",
            },
        }
        for index in range(20)
    ]
    context = VocRetriever().build_context(chunks)
    assert context.count("[Review 15]") == 1
    assert "[Review 16]" not in context
