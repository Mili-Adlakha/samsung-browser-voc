import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from models.schemas import ChatRequest, ChatResponse
from routers.ingest import get_vector_store
from services.llm import AnthropicClient, DEFAULT_CHAT_MODEL
from services.retriever import VocRetriever

logger = logging.getLogger(__name__)

router = APIRouter()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
VOC_ANALYST_PROMPT_PATH = PROMPTS_DIR / "voc_analyst.txt"


def get_llm_client() -> AnthropicClient:
    return AnthropicClient()


def load_voc_analyst_prompt() -> str:
    return VOC_ANALYST_PROMPT_PATH.read_text(encoding="utf-8")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    store = get_vector_store()
    if store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Ingest reviews first",
        )

    retriever = VocRetriever(store)
    chunks, stats = await retriever.retrieve(
        request.question,
        top_k=request.top_k,
        version_filter=request.version_filter,
    )
    context_string = retriever.build_context(chunks)
    system_prompt = load_voc_analyst_prompt()

    user_message = (
        f"## REVIEW CORPUS ({stats['total']} chunks retrieved)\n"
        f"Grounding stats: avg rating {stats['avg_rating']:.1f}/5, "
        f"high-upvote reviews: {stats['high_upvote_count']}\n\n"
        f"{context_string}\n\n"
        f"## PM QUESTION\n"
        f"{request.question}"
    )

    try:
        answer = await get_llm_client().chat(
            system_prompt=system_prompt,
            user_message=user_message,
            model=DEFAULT_CHAT_MODEL,
        )
    except Exception as exc:
        logger.exception("Anthropic chat request failed")
        return JSONResponse(
            status_code=503,
            content={
                "error": "LLM unavailable",
                "detail": str(exc),
            },
        )

    return ChatResponse(
        question=request.question,
        answer=answer,
        retrieved_chunks=stats["total"],
        avg_rating_in_context=round(stats["avg_rating"], 2),
        high_upvote_count=stats["high_upvote_count"],
        model=DEFAULT_CHAT_MODEL,
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
