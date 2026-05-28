"""Smoke tests for app routes and prompt assets."""

from pathlib import Path

from fastapi.testclient import TestClient

from main import app
from routers.chat import VOC_ANALYST_PROMPT_PATH, load_voc_analyst_prompt
from routers.dashboard import DASHBOARD_PROMPT_PATH, load_dashboard_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_serves_ui():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "VOC Intelligence" in response.text


def test_api_index_lists_endpoints():
    with TestClient(app) as client:
        response = client.get("/api")
    assert response.status_code == 200
    endpoints = response.json()["endpoints"]
    assert "POST /api/v1/ingest" in endpoints["ingest"]
    assert "/" in endpoints["ui"]


def test_prompt_files_exist_and_load():
    assert VOC_ANALYST_PROMPT_PATH.is_file()
    assert DASHBOARD_PROMPT_PATH.is_file()
    assert "Voice of Customer" in load_voc_analyst_prompt()
    assert "valid HTML" in load_dashboard_prompt()
