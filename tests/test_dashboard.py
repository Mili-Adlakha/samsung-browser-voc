"""Tests for dashboard analytics and POST /api/v1/dashboard."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from routers.dashboard import compute_analytics
from services.dashboard_renderer import render_dashboard_html
from tests.conftest import EMBEDDING_DIM, FORMAT_A_SAMPLE


def _chunk(
    review_id: str,
    text: str,
    rating: int,
    upvotes: int,
    author: str = "User",
    date: str = "2026-05-22",
) -> dict:
    return {
        "chunk_id": f"{review_id}_chunk_0",
        "text": text,
        "metadata": {
            "review_id": review_id,
            "app_version": "30.XX",
            "author": author,
            "rating": rating,
            "upvotes": upvotes,
            "date": date,
            "chunk_index": 0,
        },
    }


class TestComputeAnalytics:
    def test_theme_counts_from_known_reviews(self):
        chunks = [
            _chunk(
                "r1",
                "Latest update broke sync and lost all my tabs after login.",
                rating=1,
                upvotes=74,
                author="Jane Smith",
            ),
            _chunk(
                "r2",
                "The new UI design looks horrible and ugly, please revert classic layout.",
                rating=2,
                upvotes=12,
                author="John Doe",
            ),
            _chunk(
                "r3",
                "Great browser, fast and clean. Ad blocking works perfectly for me.",
                rating=5,
                upvotes=8,
                author="Alex Kim",
            ),
        ]

        analytics = compute_analytics(chunks, "30.XX", "20-25 May 2026")

        assert analytics["metrics"]["total_reviews"] == 3
        assert analytics["metrics"]["negative_count"] == 2
        assert analytics["metrics"]["positive_count"] == 1
        assert analytics["metrics"]["top_upvote"] == 74

        theme_counts = {theme["name"]: theme["count"] for theme in analytics["themes"]}
        assert theme_counts["Tab sync / data loss"] >= 1
        assert theme_counts["UI overhaul rejection"] >= 1
        assert theme_counts["Ad blocker degraded"] >= 1

        assert analytics["top_upvoted_reviews"][0]["upvotes"] == 74
        assert len(analytics["positive_signals"]) >= 1

    def test_empty_chunks_returns_zero_metrics(self):
        analytics = compute_analytics([], "30.XX", "Recent")
        assert analytics["metrics"]["total_reviews"] == 0
        assert analytics["metrics"]["avg_rating"] == 0.0


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    path = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(path))
    monkeypatch.setenv("COLLECTION_NAME", "test_dashboard")
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
def dashboard_client(seeded_store):
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_no_data_returns_400(chroma_dir):
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/dashboard",
            json={"app_version": "30.XX", "date_range": "May 2026"},
        )
    assert response.status_code == 400
    assert "Ingest reviews first" in response.json()["detail"]


def test_dashboard_success_returns_html(dashboard_client):
    response = dashboard_client.post(
        "/api/v1/dashboard",
        json={
            "app_version": "30.XX",
            "date_range": "20-25 May 2026",
            "top_k": 100,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["html"].startswith("<!DOCTYPE")
    assert "Chart.js" in data["html"] or "chart" in data["html"].lower()
    assert data["app_version"] == "30.XX"
    assert data["total_reviews"] == 2
    assert data["filename"].endswith("_voc_dashboard.html")


def test_template_renderer_produces_valid_html():
    chunks = [
        _chunk(
            "r1",
            "Tab sync lost all my open tabs after the latest browser update.",
            rating=1,
            upvotes=74,
            author="Jane Smith",
        ),
    ]
    analytics = compute_analytics(chunks, "30.XX", "May 2026")
    html = render_dashboard_html(analytics)
    assert html.startswith("<!DOCTYPE")
    assert "Jane Smith" in html or "Tab sync" in html


def test_dashboard_llm_failure_falls_back_to_template(seeded_store):
    mock_llm = MagicMock()
    mock_llm.generate_dashboard = AsyncMock(
        side_effect=RuntimeError("Anthropic unavailable")
    )
    with patch("routers.dashboard.USE_LLM_DASHBOARD", True):
        with patch("routers.dashboard.get_llm_client", return_value=mock_llm):
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/api/v1/dashboard",
                    json={"app_version": "30.XX", "date_range": "May 2026"},
                )
    assert response.status_code == 200
    assert response.json()["html"].startswith("<!DOCTYPE")


def test_sanitize_html_strips_fences():
    from routers.dashboard import _sanitize_html

    raw = "```html\n<!DOCTYPE html><html><body>x</body></html>\n```"
    cleaned = _sanitize_html(raw)
    assert cleaned.startswith("<!DOCTYPE")
