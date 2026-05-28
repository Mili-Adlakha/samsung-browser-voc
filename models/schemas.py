from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    """Request body for Play Store review ingestion."""

    model_config = ConfigDict(str_strip_whitespace=True)

    reviews: str = Field(
        ...,
        min_length=1,
        description="Raw Play Store review text (copy-pasted from the store or export)",
    )
    app_version: str = Field(
        ...,
        description='App version label for this batch, e.g. "30.XX"',
    )
    date_range: str = Field(
        default="",
        description='Optional date window label for dashboards, e.g. "20-25 May 2026"',
    )


class ReviewRecord(BaseModel):
    """Structured review parsed from raw Play Store text."""

    model_config = ConfigDict(str_strip_whitespace=True)

    review_id: str = Field(
        ...,
        description="Deterministic id, e.g. v{app_version}_{md5[:8]}_{index}",
    )
    app_version: str = Field(
        ...,
        description="App version associated with this review batch",
    )
    author_name: str = Field(
        ...,
        description="Reviewer display name from Play Store",
    )
    rating: int = Field(
        ...,
        ge=0,
        le=5,
        description="Star rating 1-5; 0 if not parseable",
    )
    review_text: str = Field(
        ...,
        description="Cleaned review body text",
    )
    thumbs_up_count: int = Field(
        ...,
        ge=0,
        description="Community helpful/upvote count; 0 if absent",
    )
    review_date: str = Field(
        ...,
        description="Review date as ISO string or raw parsed date text",
    )
    language: str = Field(
        default="en",
        description="Detected or assumed review language code",
    )
    ingested_at: str = Field(
        ...,
        description="ISO datetime when this record was ingested",
    )


class TopUpvotedReview(BaseModel):
    """Summary of a highly upvoted review for ingest response stats."""

    text: str = Field(..., description="Review text snippet")
    upvotes: int = Field(..., ge=0, description="Thumbs up / helpful count")
    rating: int = Field(..., ge=0, le=5, description="Star rating 1-5")


class IngestResponse(BaseModel):
    """Result of ingesting and embedding a batch of reviews."""

    status: str = Field(
        ...,
        description='Ingest outcome, e.g. "success"',
    )
    reviews_parsed: int = Field(
        ...,
        ge=0,
        description="Number of reviews successfully parsed",
    )
    chunks_stored: int = Field(
        ...,
        ge=0,
        description="Number of vector chunks upserted into ChromaDB",
    )
    avg_rating: float = Field(
        ...,
        description="Mean star rating across parsed reviews",
    )
    top_upvoted: list[TopUpvotedReview] = Field(
        ...,
        description="Top 3 reviews by upvote count",
    )
    message: str = Field(
        default="",
        description="Optional warning or hint when ingest completes with issues",
    )


class ChatRequest(BaseModel):
    """Natural-language question against the review corpus."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        description="PM question to answer using retrieved review evidence",
    )
    version_filter: str = Field(
        default="",
        description="Optional app_version metadata filter; empty means all versions",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of chunks to retrieve for RAG context",
    )


class ChatResponse(BaseModel):
    """RAG chat answer with retrieval metadata."""

    question: str = Field(..., description="Echo of the submitted question")
    answer: str = Field(
        ...,
        description="LLM-generated answer with review citations",
    )
    retrieved_chunks: int = Field(
        ...,
        ge=0,
        description="Number of chunks included in retrieval",
    )
    avg_rating_in_context: float = Field(
        ...,
        description="Mean rating across retrieved chunk metadata",
    )
    high_upvote_count: int = Field(
        ...,
        ge=0,
        description="Count of retrieved chunks with upvotes >= 10",
    )
    model: str = Field(
        ...,
        description="Anthropic model id used for generation",
    )
    timestamp: str = Field(
        ...,
        description="ISO datetime when the response was generated",
    )


class DashboardRequest(BaseModel):
    """Parameters for VOC analytics dashboard generation."""

    app_version: str = Field(
        default="30.XX",
        description="App version label shown on the dashboard",
    )
    date_range: str = Field(
        default="Recent window",
        description="Date range label shown on the dashboard header",
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum chunks to load from the vector store for analytics",
    )


class DashboardResponse(BaseModel):
    """Generated HTML dashboard and export metadata."""

    html: str = Field(
        ...,
        description="Complete self-contained HTML document string",
    )
    filename: str = Field(
        ...,
        description="Suggested filename for saving the dashboard HTML",
    )
    app_version: str = Field(
        ...,
        description="App version label used in the dashboard",
    )
    date_range: str = Field(
        ...,
        description="Date range label used in the dashboard",
    )
    total_reviews: int = Field(
        ...,
        ge=0,
        description="Total distinct reviews represented in analytics",
    )
    generated_at: str = Field(
        ...,
        description="ISO datetime when the dashboard was generated",
    )
