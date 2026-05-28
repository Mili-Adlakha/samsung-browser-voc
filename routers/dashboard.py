"""VOC analytics dashboard generation."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from models.schemas import DashboardRequest, DashboardResponse
from routers.ingest import get_vector_store
from services.dashboard_renderer import render_dashboard_html
from services.llm import AnthropicClient, DEFAULT_DASHBOARD_MODEL

logger = logging.getLogger(__name__)

router = APIRouter()

DASHBOARD_TIMEOUT_SECONDS = int(os.getenv("DASHBOARD_TIMEOUT_SECONDS", "180"))
USE_LLM_DASHBOARD = os.getenv("USE_LLM_DASHBOARD", "false").lower() in (
    "1",
    "true",
    "yes",
)
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DASHBOARD_PROMPT_PATH = PROMPTS_DIR / "dashboard_gen.txt"

THEME_KEYWORDS = {
    "UI overhaul rejection": [
        "ui",
        "design",
        "layout",
        "look",
        "interface",
        "changed",
        "horrible",
        "ugly",
        "old ui",
        "previous",
        "revert",
        "classic",
        "worse",
    ],
    "Tab group / switcher lag": [
        "tab",
        "group",
        "switcher",
        "tabs",
        "thumbnail",
        "stack",
        "stacked",
        "tab bar",
    ],
    "Crashes & freezes": [
        "crash",
        "crashes",
        "freeze",
        "freezes",
        "hang",
        "force close",
        "restart",
        "slow",
        "lag",
    ],
    "Password / autofill broken": [
        "password",
        "autofill",
        "auto-fill",
        "login",
        "credential",
        "1password",
        "bitwarden",
        "lastpass",
        "samsung pass",
    ],
    "Tab sync / data loss": [
        "sync",
        "lost",
        "data loss",
        "lost tabs",
        "tabs gone",
        "history",
        "erased",
        "deleted",
        "auto-close",
        "auto close",
    ],
    "Ad blocker degraded": [
        "ad block",
        "adblock",
        "ads",
        "advertisement",
        "blocker",
        "tracker",
    ],
    "PDF download failure": [
        "pdf",
        "download",
        "save file",
        "file manager",
    ],
    "Netflix / streaming broken": [
        "netflix",
        "stream",
        "video",
        "drm",
        "widevine",
        "hulu",
        "prime video",
    ],
    "Dark mode / wallpaper bug": [
        "dark mode",
        "wallpaper",
        "background",
        "theme",
        "night mode",
    ],
}

COMPETITOR_KEYWORDS = {
    "Chrome": ["chrome", "google chrome"],
    "Brave": ["brave browser", "brave"],
    "Firefox": ["firefox", "mozilla"],
    "iPhone/iOS": ["iphone", "ios", "apple", "safari"],
    "Google Pixel": ["pixel", "google phone", "switch to google"],
}


def get_llm_client() -> AnthropicClient:
    return AnthropicClient()


def load_dashboard_prompt() -> str:
    return DASHBOARD_PROMPT_PATH.read_text(encoding="utf-8")


def compute_analytics(
    chunks: list[dict],
    app_version: str,
    date_range: str,
) -> dict:
    reviews = _dedupe_reviews(chunks)
    total_reviews = len(reviews)

    ratings = [int((review.get("metadata") or {}).get("rating", 0)) for review in reviews]
    upvotes = [int((review.get("metadata") or {}).get("upvotes", 0)) for review in reviews]

    negative_count = sum(1 for rating in ratings if rating <= 2)
    positive_count = sum(1 for rating in ratings if rating >= 4)
    neutral_count = sum(1 for rating in ratings if rating == 3)

    def pct(count: int) -> int:
        if total_reviews == 0:
            return 0
        return int(round((count / total_reviews) * 100))

    avg_rating = round(sum(ratings) / total_reviews, 1) if total_reviews else 0.0

    themes: list[dict] = []
    for theme_name, keywords in THEME_KEYWORDS.items():
        matching_texts: list[str] = []
        for review in reviews:
            text = review.get("text", "")
            if _matches_keywords(text, keywords):
                matching_texts.append(text)
        themes.append(
            {
                "name": theme_name,
                "count": len(matching_texts),
                "pct": pct(len(matching_texts)),
                "sample_reviews": [
                    _truncate(text, 150) for text in matching_texts[:3]
                ],
            }
        )
    themes.sort(key=lambda theme: theme["count"], reverse=True)

    competitor_mentions: list[dict] = []
    for competitor, keywords in COMPETITOR_KEYWORDS.items():
        count = sum(
            1
            for review in reviews
            if _matches_keywords(review.get("text", ""), keywords)
        )
        if count:
            competitor_mentions.append({"competitor": competitor, "count": count})
    competitor_mentions.sort(key=lambda item: item["count"], reverse=True)

    positive_reviews = [
        review
        for review in reviews
        if int((review.get("metadata") or {}).get("rating", 0)) >= 4
    ]
    positive_signals = [
        _truncate(review.get("text", ""), 150) for review in positive_reviews[:5]
    ]

    sorted_by_upvotes = sorted(
        reviews,
        key=lambda review: int((review.get("metadata") or {}).get("upvotes", 0)),
        reverse=True,
    )
    top_upvoted_reviews = []
    for review in sorted_by_upvotes[:5]:
        metadata = review.get("metadata") or {}
        top_upvoted_reviews.append(
            {
                "text": _truncate(review.get("text", ""), 300),
                "upvotes": int(metadata.get("upvotes", 0)),
                "rating": int(metadata.get("rating", 0)),
                "author": str(metadata.get("author", "")),
                "date": str(metadata.get("date", "")),
            }
        )

    all_review_texts = "\n---\n".join(
        chunk.get("text", "") for chunk in chunks[:80] if chunk.get("text")
    )

    return {
        "app_version": app_version,
        "date_range": date_range,
        "metrics": {
            "total_reviews": total_reviews,
            "avg_rating": avg_rating,
            "negative_pct": pct(negative_count),
            "positive_pct": pct(positive_count),
            "neutral_pct": pct(neutral_count),
            "negative_count": negative_count,
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "top_upvote": max(upvotes) if upvotes else 0,
        },
        "themes": themes,
        "top_upvoted_reviews": top_upvoted_reviews,
        "competitor_mentions": competitor_mentions,
        "positive_signals": positive_signals,
        "all_review_texts": all_review_texts,
    }


def _dedupe_reviews(chunks: list[dict]) -> list[dict]:
    by_review_id: dict[str, dict] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        review_id = str(metadata.get("review_id") or chunk.get("id") or "")
        if not review_id:
            review_id = chunk.get("text", "")[:80]

        existing = by_review_id.get(review_id)
        if existing is None:
            by_review_id[review_id] = chunk
            continue

        existing_meta = existing.get("metadata") or {}
        new_upvotes = int(metadata.get("upvotes", 0))
        existing_upvotes = int(existing_meta.get("upvotes", 0))
        if new_upvotes > existing_upvotes or len(chunk.get("text", "")) > len(
            existing.get("text", "")
        ):
            by_review_id[review_id] = chunk
    return list(by_review_id.values())


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _truncate(text: str, max_len: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _format_analytics_message(analytics: dict) -> str:
    """Slim analytics payload for LLM (avoids huge all_review_texts timeouts)."""
    slim = copy.deepcopy(analytics)
    texts = slim.get("all_review_texts", "")
    if isinstance(texts, str) and len(texts) > 4000:
        slim["all_review_texts"] = texts[:4000] + "\n… [truncated]"
    return (
        "Generate the Samsung Browser VOC dashboard HTML using this analytics data.\n\n"
        f"{json.dumps(slim, indent=2)}"
    )


async def _generate_dashboard_html_llm(analytics: dict) -> str:
    system_prompt = load_dashboard_prompt()
    user_message = _format_analytics_message(analytics)
    return await asyncio.wait_for(
        get_llm_client().generate_dashboard(
            system_prompt=system_prompt,
            user_message=user_message,
            model=os.getenv("DASHBOARD_MODEL", DEFAULT_DASHBOARD_MODEL),
        ),
        timeout=DASHBOARD_TIMEOUT_SECONDS,
    )


def _sanitize_html(html: str) -> str:
    cleaned = html.strip()
    cleaned = re.sub(r"^```(?:html)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

    lower = cleaned.lower()
    if lower.startswith("<!doctype") or lower.startswith("<html"):
        return cleaned

    for marker in ("<!doctype", "<html"):
        index = lower.find(marker)
        if index >= 0:
            return cleaned[index:]
    return cleaned


def _dashboard_filename(app_version: str) -> str:
    safe_version = re.sub(r"[^\w.-]+", "_", app_version).replace(".", "_")
    return f"samsung_browser_{safe_version}_voc_dashboard.html"


@router.post("/dashboard", response_model=DashboardResponse)
async def generate_dashboard(request: DashboardRequest) -> DashboardResponse:
    store = get_vector_store()
    if store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Ingest reviews first",
        )

    chunks = store.get_all(top_k=request.top_k)
    analytics = compute_analytics(chunks, request.app_version, request.date_range)

    try:
        if USE_LLM_DASHBOARD:
            logger.info("Generating dashboard via LLM (timeout=%ss)", DASHBOARD_TIMEOUT_SECONDS)
            html = await _generate_dashboard_html_llm(analytics)
        else:
            logger.info("Generating dashboard via template renderer")
            html = render_dashboard_html(analytics)
    except asyncio.TimeoutError:
        logger.warning("LLM dashboard timed out; falling back to template renderer")
        try:
            html = render_dashboard_html(analytics)
        except Exception as fallback_exc:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "LLM unavailable",
                    "detail": (
                        f"Dashboard generation exceeded {DASHBOARD_TIMEOUT_SECONDS}s "
                        f"and template fallback failed: {fallback_exc}"
                    ),
                },
            )
    except Exception as exc:
        if USE_LLM_DASHBOARD:
            logger.warning("LLM dashboard failed (%s); using template fallback", exc)
            try:
                html = render_dashboard_html(analytics)
            except Exception as fallback_exc:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "LLM unavailable",
                        "detail": f"{exc}; fallback failed: {fallback_exc}",
                    },
                )
        else:
            logger.exception("Template dashboard generation failed")
            return JSONResponse(
                status_code=503,
                content={"error": "Dashboard unavailable", "detail": str(exc)},
            )

    html = _sanitize_html(html)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return DashboardResponse(
        html=html,
        filename=_dashboard_filename(request.app_version),
        app_version=request.app_version,
        date_range=request.date_range,
        total_reviews=analytics["metrics"]["total_reviews"],
        generated_at=generated_at,
    )
