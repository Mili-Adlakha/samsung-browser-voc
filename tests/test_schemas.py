"""Phase 2: validate schemas against API contract samples."""

import json

from models.schemas import (
    ChatRequest,
    ChatResponse,
    DashboardRequest,
    DashboardResponse,
    IngestRequest,
    IngestResponse,
    TopUpvotedReview,
)


def test_ingest_request_from_spec():
    req = IngestRequest.model_validate(
        {
            "reviews": "John Doe\n★\nTabs lost after sync.",
            "app_version": "30.XX",
            "date_range": "20-25 May 2026",
        }
    )
    assert req.app_version == "30.XX"
    assert req.date_range == "20-25 May 2026"


def test_ingest_response_from_spec():
    resp = IngestResponse.model_validate(
        {
            "status": "success",
            "reviews_parsed": 2,
            "chunks_stored": 2,
            "avg_rating": 3.0,
            "top_upvoted": [
                {"text": "Tabs lost after sync.", "upvotes": 74, "rating": 1}
            ],
        }
    )
    assert resp.reviews_parsed == 2
    assert resp.top_upvoted[0].upvotes == 74


def test_chat_request_defaults():
    req = ChatRequest(question="What are top tab sync complaints?")
    assert req.top_k == 20
    assert req.version_filter == ""


def test_chat_response_from_spec():
    resp = ChatResponse.model_validate(
        {
            "question": "What are the top complaints about tab sync?",
            "answer": "**Finding:** Tab sync data loss...",
            "retrieved_chunks": 20,
            "avg_rating_in_context": 1.8,
            "high_upvote_count": 1,
            "model": "claude-sonnet-4-5",
            "timestamp": "2026-05-28T10:30:00Z",
        }
    )
    assert resp.model == "claude-sonnet-4-5"


def test_dashboard_request_defaults():
    req = DashboardRequest()
    assert req.app_version == "30.XX"
    assert req.top_k == 100


def test_dashboard_response_from_spec():
    resp = DashboardResponse.model_validate(
        {
            "html": "<!DOCTYPE html><html lang=\"en\">...</html>",
            "filename": "samsung_browser_30_XX_voc_dashboard.html",
            "app_version": "30.XX",
            "date_range": "20-25 May 2026",
            "total_reviews": 127,
            "generated_at": "2026-05-28T10:31:45Z",
        }
    )
    assert resp.total_reviews == 127
    assert resp.html.startswith("<!DOCTYPE")


def test_ingest_request_rejects_empty_reviews():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"reviews": "", "app_version": "30.XX"})


def test_top_upvoted_serializes_as_dict():
    item = TopUpvotedReview(text="x", upvotes=3, rating=5)
    assert json.loads(item.model_dump_json()) == {
        "text": "x",
        "upvotes": 3,
        "rating": 5,
    }
