import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from models.schemas import IngestRequest, IngestResponse, TopUpvotedReview
from services.embedder import chunk_review, get_embeddings
from services.parser import parse_play_store_text
from services.vector_store import VocVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


def get_vector_store() -> VocVectorStore:
    return VocVectorStore.from_env()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_reviews(request: IngestRequest) -> IngestResponse:
    if not request.reviews.strip():
        raise HTTPException(status_code=400, detail="reviews field must not be empty")

    records = parse_play_store_text(request.reviews, request.app_version)

    all_chunks: list[dict] = []
    for record in records:
        all_chunks.extend(chunk_review(record))

    if not records:
        logger.warning("Ingest completed but no reviews were parsed from input")
        raise HTTPException(
            status_code=422,
            detail=(
                "No reviews could be parsed. Paste Play Store reviews (author, date, text) "
                "or format with ★ ratings, Play Console CSV, or numbered blocks. "
                "Each review body needs 15+ characters. Excel (.xlsx) is not supported."
            ),
        )

    try:
        embeddings = await get_embeddings(
            [chunk["embedding_text"] for chunk in all_chunks]
        )
    except Exception as exc:
        logger.exception("Embedding service failed during ingest")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Embedding service unavailable",
                "detail": str(exc),
            },
        )

    store = get_vector_store()
    chunks_stored = store.upsert_chunks(all_chunks, embeddings)

    avg_rating = sum(record.rating for record in records) / len(records)
    top_records = sorted(
        records,
        key=lambda record: record.thumbs_up_count,
        reverse=True,
    )[:3]
    top_upvoted = [
        TopUpvotedReview(
            text=record.review_text,
            upvotes=record.thumbs_up_count,
            rating=record.rating,
        )
        for record in top_records
    ]

    logger.info(
        "Ingested %s reviews → %s chunks stored",
        len(records),
        chunks_stored,
    )

    return IngestResponse(
        status="success",
        reviews_parsed=len(records),
        chunks_stored=chunks_stored,
        avg_rating=round(avg_rating, 2),
        top_upvoted=top_upvoted,
        message=f"Parsed {len(records)} reviews into {chunks_stored} chunks.",
    )
