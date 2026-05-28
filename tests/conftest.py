"""Shared pytest fixtures and sample data."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app

EMBEDDING_DIM = 1536
APP_VERSION = "30.XX"

FORMAT_A_SAMPLE = """\
John Doe
★★★★★
Great browser, fast and clean. Ad blocking works perfectly.
3 people found this review helpful
Did you find this helpful? Yes No

Jane Smith
★
Latest update broke everything. Lost all my tabs after sync.
74 people found this review helpful
Did you find this helpful? Yes No
"""


@pytest.fixture
def chroma_env(tmp_path, monkeypatch):
    """Isolate ChromaDB to a temp directory for each test module."""
    path = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(path))
    monkeypatch.setenv("COLLECTION_NAME", "test_voc")
    return path


@pytest.fixture
def mock_embeddings():
    """Patch OpenAI embedding calls with deterministic fake vectors."""
    with patch(
        "routers.ingest.get_embeddings",
        new=AsyncMock(
            side_effect=lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts]
        ),
    ):
        yield


@pytest.fixture
def ingested_client(chroma_env, mock_embeddings):
    """FastAPI client with sample reviews already ingested."""
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ingest",
            json={"reviews": FORMAT_A_SAMPLE, "app_version": APP_VERSION},
        )
        assert response.status_code == 200
        yield client
