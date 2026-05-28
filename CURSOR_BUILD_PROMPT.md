# Samsung Browser — VOC Intelligence System
## Cursor Build Prompt · Software Requirements Specification

---

> **How to use this file in Cursor**
> 1. Drop this file into your project root as `CURSOR_BUILD_PROMPT.md`
> 2. Open Cursor → press `Cmd+Shift+P` → `Cursor: Use as Context`
> 3. Or reference it directly in chat: `@CURSOR_BUILD_PROMPT.md build Pipeline 1`
> 4. Each section is self-contained — you can feed individual sections to Cursor for incremental builds

---

## 0. Project Overview

Build a **VOC (Voice of Customer) Intelligence System** for Samsung Browser product management. The system ingests raw Play Store review text, provides a RAG-based chatbot for natural language querying, and auto-generates an HTML analytics dashboard — all powered by LLMs.

### Tech Stack (non-negotiable)
```
Runtime:       Python 3.11+
Web framework: FastAPI
Vector store:  ChromaDB (local, persistent to ./chroma_db/)
Embeddings:    OpenAI text-embedding-3-small
LLM:           Anthropic Claude API (claude-sonnet-4-5 for chat, claude-sonnet-4-6 for dashboard)
Frontend:      Vanilla HTML/CSS/JS (no React — dashboard is a generated static file)
Config:        python-dotenv (.env file)
Package mgr:   pip + requirements.txt
```

### Environment variables required (`.env`)
```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=samsung_browser_voc
PORT=8000
```

### Project structure to generate
```
samsung-voc/
├── main.py                  # FastAPI app, route registration
├── .env                     # secrets (gitignored)
├── .env.example             # committed template
├── requirements.txt
├── README.md
├── chroma_db/               # auto-created by ChromaDB
├── routers/
│   ├── __init__.py
│   ├── ingest.py            # Pipeline 1: POST /ingest
│   ├── chat.py              # Pipeline 2: POST /chat
│   └── dashboard.py         # Pipeline 3: POST /dashboard
├── services/
│   ├── __init__.py
│   ├── parser.py            # Play Store text → structured reviews
│   ├── embedder.py          # OpenAI embedding wrapper
│   ├── vector_store.py      # ChromaDB CRUD wrapper
│   ├── retriever.py         # Semantic search + metadata filter
│   └── llm.py               # Anthropic API wrapper (chat + dashboard)
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic request/response models
├── prompts/
│   ├── voc_analyst.txt      # System prompt for RAG chatbot
│   └── dashboard_gen.txt    # System prompt for dashboard HTML generation
└── tests/
    ├── test_parser.py
    ├── test_ingest.py
    ├── test_chat.py
    └── test_dashboard.py
```

---

## 1. Data Models — `models/schemas.py`

Generate all Pydantic v2 models. Every field must have a description string.

```python
# ── Ingest ────────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    reviews: str          # Raw Play Store review text (copy-pasted)
    app_version: str      # e.g. "30.XX"
    date_range: str = ""  # e.g. "20-25 May 2026" (optional, for dashboard labelling)

class ReviewRecord(BaseModel):
    review_id: str         # Generated: f"review_{hash}_{timestamp}"
    app_version: str
    author_name: str
    rating: int            # 1-5, 0 if not parseable
    review_text: str       # Cleaned body text
    thumbs_up_count: int   # Community upvotes, 0 if absent
    review_date: str       # ISO date string or raw parsed date
    language: str = "en"
    ingested_at: str       # ISO datetime

class IngestResponse(BaseModel):
    status: str
    reviews_parsed: int
    chunks_stored: int
    avg_rating: float
    top_upvoted: list[dict]  # [{text, upvotes, rating}] top 3

# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    version_filter: str = ""   # Optional: filter to specific app version
    top_k: int = 20            # Number of chunks to retrieve

class ChatResponse(BaseModel):
    question: str
    answer: str                 # LLM-generated, cited response
    retrieved_chunks: int
    avg_rating_in_context: float
    high_upvote_count: int      # Chunks with upvotes >= 10
    model: str
    timestamp: str

# ── Dashboard ─────────────────────────────────────────────────────────────────
class DashboardRequest(BaseModel):
    app_version: str = "30.XX"
    date_range: str = "Recent window"
    top_k: int = 100           # Chunks to retrieve for analytics

class DashboardResponse(BaseModel):
    html: str                  # Complete self-contained HTML string
    filename: str              # Suggested save filename
    app_version: str
    date_range: str
    total_reviews: int
    generated_at: str
```

---

## 2. Pipeline 1 — Review Ingestion

### 2.1 Parser — `services/parser.py`

**Function signature:**
```python
def parse_play_store_text(raw_text: str, app_version: str) -> list[ReviewRecord]:
```

**Parser must handle these three Play Store copy-paste formats:**

**Format A** (most common — browser copy-paste):
```
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
```

**Format B** (Play Console CSV export style):
```
reviewId,authorName,reviewText,starRating,thumbsUpCount,reviewCreatedVersion,at
abc123,John Doe,Great browser,5,3,30.0.2.30,2026-05-22 10:30:00
def456,Jane Smith,Lost all tabs,1,74,30.0.0.63,2026-05-23 14:15:00
```

**Format C** (plain numbered blocks):
```
1. Rating: 5/5 | John Doe | 22 May 2026
Great browser, fast and clean.
Upvotes: 3

2. Rating: 1/5 | Jane Smith | 23 May 2026
Lost all tabs after update. Terrible sync.
Upvotes: 74
```

**Parser logic requirements:**
- Auto-detect format from first 200 chars (CSV header check, numbered block check, or default to Format A)
- Extract: author, rating (stars/digits/★ count), review body, upvote count, date
- Strip noise: "Did you find this helpful? Yes No", "people found this review helpful", Play Store UI chrome
- Minimum review body length: 15 characters (skip shorter)
- Generate deterministic `review_id`: `f"v{app_version}_{hashlib.md5(review_text.encode()).hexdigest()[:8]}_{i}"`
- Return empty list (not error) if no reviews parseable
- Log warnings for skipped blocks

### 2.2 Chunking — `services/embedder.py`

**Function signature:**
```python
def chunk_review(review: ReviewRecord) -> list[dict]:
```

**Chunking rules:**
- Reviews ≤ 500 chars → single chunk
- Reviews > 500 chars → split at sentence boundaries (`[.!?]` + space), max 450 chars per chunk
- Each chunk dict contains:
  ```python
  {
    "chunk_id": f"{review.review_id}_chunk_{i}",
    "text": chunk_text,
    "embedding_text": f"[Rating:{review.rating}/5][Upvotes:{review.thumbs_up_count}][Version:{review.app_version}] {chunk_text}",
    "metadata": {
      "review_id": review.review_id,
      "app_version": review.app_version,
      "author": review.author_name,
      "rating": review.rating,
      "upvotes": review.thumbs_up_count,
      "date": review.review_date,
      "chunk_index": i
    }
  }
  ```
- The `embedding_text` prefix anchors semantic search to upvote-weighted, version-filtered results

**Embedding function:**
```python
async def get_embedding(text: str) -> list[float]:
    # Call OpenAI text-embedding-3-small
    # Model: text-embedding-3-small (1536 dimensions)
    # Batch up to 100 texts per API call
    # Return list[float]
```

### 2.3 Vector Store — `services/vector_store.py`

Use **ChromaDB** with persistent local storage.

```python
class VocVectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        # chromadb.PersistentClient(path=persist_dir)
        # collection: get_or_create_collection(collection_name)

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        # Use chunk_id as ChromaDB document id
        # Upsert (not insert) — handles re-ingestion of same reviews
        # Return count of chunks stored

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: dict | None = None   # ChromaDB metadata filter
    ) -> list[dict]:
        # Returns list of {text, metadata, distance} dicts
        # Apply where filter for version_filter if provided

    def get_all(self, top_k: int = 100) -> list[dict]:
        # Returns all stored chunks (for dashboard analytics)
        # Sorted by upvotes desc

    def count(self) -> int:
        # Total chunks in collection

    def reset(self) -> None:
        # Delete and recreate collection (for testing)
```

### 2.4 Ingest Router — `routers/ingest.py`

```
POST /ingest
Content-Type: application/json
Body: IngestRequest
Response: IngestResponse
```

**Full flow:**
1. Validate request (reviews field not empty)
2. `parser.parse_play_store_text()` → list of ReviewRecord
3. For each ReviewRecord: `embedder.chunk_review()` → list of chunk dicts
4. Batch all `embedding_text` values → `embedder.get_embedding()` (batch call)
5. `vector_store.upsert_chunks()` with chunks + embeddings
6. Compute and return IngestResponse stats
7. Log: `f"Ingested {reviews_parsed} reviews → {chunks_stored} chunks stored"`

**Error handling:**
- 400: empty reviews field
- 422: Pydantic validation error (auto)
- 500: OpenAI API error → return `{"error": "Embedding service unavailable", "detail": str(e)}`

---

## 3. Pipeline 2 — RAG Chatbot

### 3.1 Retriever — `services/retriever.py`

```python
class VocRetriever:
    def retrieve(
        self,
        question: str,
        top_k: int = 20,
        version_filter: str = ""
    ) -> tuple[list[dict], dict]:
        # 1. Embed the question using embedder.get_embedding()
        # 2. Build ChromaDB where filter if version_filter provided:
        #    {"app_version": {"$eq": version_filter}}
        # 3. vector_store.query(query_embedding, top_k, where)
        # 4. Sort results: upvotes DESC as tiebreaker after distance
        # 5. Compute context stats:
        stats = {
            "total": len(results),
            "avg_rating": ...,           # mean of metadata.rating
            "high_upvote_count": ...,    # count where upvotes >= 10
        }
        # 6. Return (results, stats)

    def build_context(self, chunks: list[dict]) -> str:
        # Format each chunk as:
        # [Review N] Rating: X/5 | Upvotes: Y | Version: Z | Date: D
        # <review text>
        # ---
        # Max 15 chunks in context (token budget)
        # Return joined string
```

### 3.2 LLM Wrapper — `services/llm.py`

```python
class AnthropicClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 1500
    ) -> str:
        # Call anthropic messages.create()
        # Return response text content
        # Raise on API error with descriptive message

    async def generate_dashboard(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8000
    ) -> str:
        # Same but higher token limit for HTML generation
        # Strip markdown fences if present: ```html ... ```
```

### 3.3 VOC Analyst System Prompt — `prompts/voc_analyst.txt`

```
You are a senior Samsung Browser product analyst specializing in Voice of Customer (VOC) analysis. You have deep expertise in mobile browser UX, Android app quality, and user sentiment analysis.

Your role is to answer PM questions about Samsung Browser user reviews with precision, evidence, and actionable insight.

## CORE RULES
1. Answer ONLY using the review evidence provided in the context. Never use general knowledge about Samsung Browser.
2. For EVERY factual claim, cite the Review number (e.g., [Review 3]) and include the upvote count if > 0.
3. For numerical claims (counts, percentages), base them strictly on the evidence provided — never extrapolate.
4. If the context lacks sufficient evidence, respond: "Insufficient evidence in the current corpus for this query."
5. Never hallucinate specific user names, dates, or review text not present in the context.

## RESPONSE STRUCTURE
**Finding:** [1-2 sentence direct answer]

**Evidence:** [Specific quotes with citation numbers and upvote counts]

**PM Implication:** [What this means for product roadmap or immediate action]

**Caveats:** [Limitations in the evidence — e.g., "Based on X reviews in corpus"]

## QUERY TYPE HANDLING
- Quantitative ("how many"): Count from evidence, state the number, cite reviews
- Thematic ("what are top issues"): Rank by frequency + upvote weight
- Comparative ("v29 vs v30"): Only compare if both versions in context
- Investigative ("show me reviews about X"): List matching reviews with metadata
- Feature requests: Extract specific asks, note recurrence count

## TONE
PM-grade: specific, structured, actionable. No filler. Flag P1 implications explicitly.
```

### 3.4 Chat Router — `routers/chat.py`

```
POST /chat
Content-Type: application/json
Body: ChatRequest
Response: ChatResponse
```

**Full flow:**
1. Validate question not empty
2. Check vector store has data: `vector_store.count() > 0` → 400 if empty ("Ingest reviews first")
3. `retriever.retrieve(question, top_k, version_filter)` → (chunks, stats)
4. `retriever.build_context(chunks)` → context_string
5. Load system prompt from `prompts/voc_analyst.txt`
6. Build user message:
   ```
   ## REVIEW CORPUS ({stats['total']} chunks retrieved)
   Grounding stats: avg rating {stats['avg_rating']}/5, high-upvote reviews: {stats['high_upvote_count']}

   {context_string}

   ## PM QUESTION
   {question}
   ```
7. `llm.chat(system_prompt, user_message)` → answer
8. Return ChatResponse

**Error handling:**
- 400: empty question, or no data in vector store
- 503: Anthropic API error → `{"error": "LLM unavailable", "detail": str(e)}`

---

## 4. Pipeline 3 — Dashboard Generator

### 4.1 Analytics Computation — inline in `routers/dashboard.py`

```python
def compute_analytics(chunks: list[dict], app_version: str, date_range: str) -> dict:
```

**Must compute:**

```python
{
    "app_version": app_version,
    "date_range": date_range,
    "metrics": {
        "total_reviews": int,
        "avg_rating": float,          # 1 decimal place
        "negative_pct": int,          # rating <= 2
        "positive_pct": int,          # rating >= 4
        "neutral_pct": int,           # rating == 3
        "negative_count": int,
        "positive_count": int,
        "neutral_count": int,
        "top_upvote": int,            # highest single upvote count
    },
    "themes": [                       # sorted by count desc
        {
            "name": str,
            "count": int,
            "pct": int,               # % of total reviews
            "sample_reviews": list[str]  # up to 3 text snippets, max 150 chars each
        }
    ],
    "top_upvoted_reviews": [          # top 5 by upvotes
        {
            "text": str,              # full text, max 300 chars
            "upvotes": int,
            "rating": int,
            "author": str,
            "date": str
        }
    ],
    "competitor_mentions": [          # sorted by count desc
        {"competitor": str, "count": int}
    ],
    "positive_signals": list[str],    # up to 5 text snippets from 4-5 star reviews
    "all_review_texts": str           # first 80 chunks joined by \n---\n (for LLM feature extraction)
}
```

**Theme keyword taxonomy (use exactly these labels and keywords):**

```python
THEME_KEYWORDS = {
    "UI overhaul rejection":       ["ui", "design", "layout", "look", "interface", "changed", "horrible", "ugly", "old ui", "previous", "revert", "classic", "worse"],
    "Tab group / switcher lag":    ["tab", "group", "switcher", "tabs", "thumbnail", "stack", "stacked", "tab bar"],
    "Crashes & freezes":           ["crash", "crashes", "freeze", "freezes", "hang", "force close", "restart", "slow", "lag"],
    "Password / autofill broken":  ["password", "autofill", "auto-fill", "login", "credential", "1password", "bitwarden", "lastpass", "samsung pass"],
    "Tab sync / data loss":        ["sync", "lost", "data loss", "lost tabs", "tabs gone", "history", "erased", "deleted", "auto-close", "auto close"],
    "Ad blocker degraded":         ["ad block", "adblock", "ads", "advertisement", "blocker", "tracker"],
    "PDF download failure":        ["pdf", "download", "save file", "file manager"],
    "Netflix / streaming broken":  ["netflix", "stream", "video", "drm", "widevine", "hulu", "prime video"],
    "Dark mode / wallpaper bug":   ["dark mode", "wallpaper", "background", "theme", "night mode"],
}

COMPETITOR_KEYWORDS = {
    "Chrome":      ["chrome", "google chrome"],
    "Brave":       ["brave browser", "brave"],
    "Firefox":     ["firefox", "mozilla"],
    "iPhone/iOS":  ["iphone", "ios", "apple", "safari"],
    "Google Pixel":["pixel", "google phone", "switch to google"],
}
```

### 4.2 Dashboard System Prompt — `prompts/dashboard_gen.txt`

```
You are a Samsung Browser product analytics expert. Generate a complete, self-contained HTML analytics dashboard.

## CRITICAL OUTPUT RULE
Return ONLY valid HTML — no markdown, no explanations, no code fences.
The entire response must be one self-contained HTML file that renders in a browser.

## VISUAL DESIGN SPECIFICATION

### CSS Variables (use exactly these)
--bg: #f4f3ef
--surface: #ffffff
--surface-2: #f9f8f5
--border: rgba(0,0,0,0.08)
--text-primary: #1a1a18
--text-secondary: #5a5a55
--text-tertiary: #9a9990
--samsung: #1428A0
--dark-blue: #0D1F7A
--mid-blue: #2A52B5
--light-blue: #D6E4F7
--accent-blue: #EEF4FC
--red: #d94040
--red-light: #fdf0f0
--amber: #c47a18
--amber-light: #fef5e7
--green: #3a7c3a
--green-light: #eef6ee
--grey: #b4b2a9
--radius-md: 10px
--radius-lg: 14px

### Typography
Load from Google Fonts CDN:
- DM Sans (weights 300, 400, 500) — body and UI
- DM Mono (weights 400, 500) — labels, numbers, section headers

### Chart.js
CDN: https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js
Config for all charts: responsive:true, maintainAspectRatio:false, legend:{display:false}
Doughnut charts: cutout: "68%"
Colors: red #d94040, green #3a7c3a, grey #b4b2a9, amber #d87a30, blue #2a6db5, purple #6b52b8, teal #1a8a6b

## REQUIRED SECTIONS — generate in this order

### 1. HEADER (background: #1a1a18)
- Eyebrow: "SAMSUNG BROWSER · Play Store VOC"
- Title: "Samsung Browser {version} — VOC Analytics"
- Subtitle: "{date_range} · {total} reviews · Post-release monitoring"
- If negative_pct > 50%: red alert pill "⚠ High complaint volume"

### 2. KPI STRIP — 5 metric cards, top-colored left border
- Total reviews (blue border)
- Negative % (red border, red value)
- Positive % (green border, green value)
- Neutral % (grey border)
- Top upvote count (blue border)

### 3. VOLUME & SENTIMENT (2-col grid: 1.4fr 1fr)
- Left: Stacked bar chart — daily volume (neg=red, pos=green). Distribute total across available dates proportionally.
- Right: Sentiment doughnut chart + legend

### 4. ISSUE BREAKDOWN (2-col grid: 1.4fr 1fr)
- Left: Horizontal bar chart — themes with count and % labels
- Right: Category doughnut (UI/UX, Stability, Feature, Data/Sync, Privacy — derive from theme data)

### 5. PRIORITY ACTION PLAN (3-col grid)
- P1 column (red top border): Issues with count > 5, plus highest-upvote reviews
- P2 column (amber top border): Issues count 3-5, UX regressions
- P3 column (green top border): Roadmap items extracted from review text

### 6. HIGH-SIGNAL REVIEWS + COMPETITIVE RISK (2-col grid)
- Left: Top 5 upvoted reviews — quote block style, left border color by rating severity, upvote badge
- Right top: Competitor mentions with counts
- Right bottom: Positive signals to protect

### 7. FEATURE REQUESTS (full width)
- Pill-tag style, each with P1/P2/P3 badge
- Extract from review text + sample_reviews

### 8. FOOTER
- Left: "Samsung Browser PM · VOC Analysis · {date_range} · Source: Google Play Store reviews"
- Right: Negative/Positive/Neutral legend squares

## CONTENT RULES
- All numbers from analytics data provided — never invent
- P1 = theme count > 5 OR upvote > 30; P2 = count 3-5; P3 = feature requests
- Quote reviews verbatim (max 200 chars) with attribution
- Section labels: DM Mono, 10px, uppercase, 0.1em spacing, #9a9990, horizontal rule after
- Cards: white bg, 1px border rgba(0,0,0,0.08), 14px border-radius, 18-20px padding
- Generate real, specific insight text — no placeholder copy, no "lorem ipsum"
```

### 4.3 Dashboard Router — `routers/dashboard.py`

```
POST /dashboard
Content-Type: application/json
Body: DashboardRequest
Response: DashboardResponse
```

**Full flow:**
1. Check vector store count > 0 → 400 if empty
2. `vector_store.get_all(top_k)` → all chunks
3. `compute_analytics(chunks, app_version, date_range)` → analytics dict
4. Load system prompt from `prompts/dashboard_gen.txt`
5. Build user message with full analytics dict formatted as structured text
6. `llm.generate_dashboard(system_prompt, user_message)` → html string
7. Clean HTML: strip markdown fences, ensure starts with `<!DOCTYPE` or `<html`
8. Return DashboardResponse with html + metadata

**Error handling:**
- 400: no data in vector store
- 503: Anthropic API unavailable
- Timeout: set 60-second timeout on dashboard generation (long LLM call)

---

## 5. Main App — `main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ingest, chat, dashboard

app = FastAPI(
    title="Samsung Browser VOC Intelligence System",
    description="RAG-based VOC analysis pipeline for Samsung Browser Play Store reviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router,   prefix="/api/v1", tags=["Chat"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "Samsung Browser VOC Intelligence System"}

@app.get("/")
def root():
    return {
        "endpoints": {
            "ingest":    "POST /api/v1/ingest",
            "chat":      "POST /api/v1/chat",
            "dashboard": "POST /api/v1/dashboard",
            "docs":      "/docs",
            "health":    "/health"
        }
    }
```

---

## 6. Requirements — `requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.7.0
python-dotenv==1.0.1
anthropic==0.28.0
openai==1.35.0
chromadb==0.5.3
httpx==0.27.0
pytest==8.2.0
pytest-asyncio==0.23.7
httpx==0.27.0
```

---

## 7. API Contract — Complete Reference

### POST `/api/v1/ingest`

**Request:**
```json
{
  "reviews": "John Doe\n★\nTabs lost after sync.\n74 people found this review helpful\n\nJane Smith\n★★★★★\nBest browser ever!",
  "app_version": "30.XX",
  "date_range": "20-25 May 2026"
}
```

**Response:**
```json
{
  "status": "success",
  "reviews_parsed": 2,
  "chunks_stored": 2,
  "avg_rating": 3.0,
  "top_upvoted": [
    {"text": "Tabs lost after sync.", "upvotes": 74, "rating": 1}
  ]
}
```

---

### POST `/api/v1/chat`

**Request:**
```json
{
  "question": "What are the top complaints about tab sync?",
  "version_filter": "30.XX",
  "top_k": 20
}
```

**Response:**
```json
{
  "question": "What are the top complaints about tab sync?",
  "answer": "**Finding:** Tab sync data loss is the highest-signal issue...\n\n**Evidence:** [Review 1] (74 upvotes): \"Tabs lost after sync\"...\n\n**PM Implication:** P1 — requires hotfix in v30.1...\n\n**Caveats:** Based on 20 chunks retrieved from 2-review corpus.",
  "retrieved_chunks": 20,
  "avg_rating_in_context": 1.8,
  "high_upvote_count": 1,
  "model": "claude-sonnet-4-5",
  "timestamp": "2026-05-28T10:30:00Z"
}
```

---

### POST `/api/v1/dashboard`

**Request:**
```json
{
  "app_version": "30.XX",
  "date_range": "20-25 May 2026",
  "top_k": 100
}
```

**Response:**
```json
{
  "html": "<!DOCTYPE html><html lang=\"en\">...",
  "filename": "samsung_browser_30_XX_voc_dashboard.html",
  "app_version": "30.XX",
  "date_range": "20-25 May 2026",
  "total_reviews": 127,
  "generated_at": "2026-05-28T10:31:45Z"
}
```

---

## 8. Tests — Cursor, generate these

### `tests/test_parser.py`
```python
# Test parse_play_store_text() with:
# - Format A (star + upvotes)
# - Format B (CSV)
# - Format C (numbered)
# - Mixed quality input (some blocks too short → skipped)
# - Empty string → returns []
# - Single review → returns list of 1
```

### `tests/test_ingest.py`
```python
# Use FastAPI TestClient
# Test POST /api/v1/ingest with:
# - Valid Format A text → 200 + correct counts
# - Empty reviews field → 400
# - Version filter stored correctly in metadata
```

### `tests/test_chat.py`
```python
# Mock Anthropic client (no real API calls in tests)
# Test POST /api/v1/chat:
# - No data in store → 400
# - Empty question → 400
# - Valid question → 200 with answer field present
# - version_filter applied → only matching chunks retrieved
```

### `tests/test_dashboard.py`
```python
# Mock Anthropic client
# Test POST /api/v1/dashboard:
# - No data in store → 400
# - Valid request → 200, html field starts with <!DOCTYPE
# - compute_analytics() unit test: correct theme counts from known review set
```

---

## 9. README — `README.md`

Generate a README with:
1. One-paragraph what this is
2. Setup (clone → pip install → .env → uvicorn main:app)
3. Usage examples with curl for all 3 endpoints
4. How to save dashboard HTML: `curl ... | python3 -c "import sys,json; print(json.load(sys.stdin)['html'])" > dashboard.html`
5. Architecture diagram in ASCII
6. Known limitations (in-memory ChromaDB resets on restart — persist_dir solves this)

---

## 10. Cursor-specific build instructions

When building this project in Cursor, use these commands in order:

```
Phase 1 — Scaffold
@CURSOR_BUILD_PROMPT.md Create the project structure, requirements.txt, .env.example, and main.py

Phase 2 — Models
@CURSOR_BUILD_PROMPT.md Build models/schemas.py with all Pydantic models from Section 1

Phase 3 — Parser
@CURSOR_BUILD_PROMPT.md Build services/parser.py. Handle all three formats from Section 2.1. Include unit tests in tests/test_parser.py

Phase 4 — Embedder + Vector Store
@CURSOR_BUILD_PROMPT.md Build services/embedder.py and services/vector_store.py from Section 2.2 and 2.3

Phase 5 — Pipeline 1
@CURSOR_BUILD_PROMPT.md Build routers/ingest.py implementing the full flow from Section 2.4

Phase 6 — Pipeline 2
@CURSOR_BUILD_PROMPT.md Build services/retriever.py, services/llm.py, prompts/voc_analyst.txt, and routers/chat.py from Sections 3.1–3.4

Phase 7 — Pipeline 3
@CURSOR_BUILD_PROMPT.md Build routers/dashboard.py with compute_analytics() and prompts/dashboard_gen.txt from Section 4

Phase 8 — Tests
@CURSOR_BUILD_PROMPT.md Generate all tests from Section 8 with mocked Anthropic client

Phase 9 — README
@CURSOR_BUILD_PROMPT.md Generate README.md from Section 9
```

---

## 11. Non-negotiable constraints for Cursor

1. **No mock data** — parser must work on real Play Store copy-paste text, not synthetic clean data
2. **ChromaDB upsert not insert** — re-running ingest on same reviews must not duplicate chunks
3. **Embedding text prefix** — always prepend `[Rating:X/5][Upvotes:Y][Version:Z]` to chunk text before embedding; this is load-bearing for retrieval quality
4. **LLM system prompt from file** — load `prompts/voc_analyst.txt` at request time, not hardcoded in router
5. **Dashboard HTML cleaning** — always strip markdown code fences from LLM response before returning
6. **No streaming** — all responses are synchronous JSON; no SSE or websockets
7. **Async throughout** — all route handlers and LLM calls must be `async def`
8. **Pydantic v2** — use `model_config = ConfigDict(...)` not `class Config`; use `model_validate` not `parse_obj`
9. **Error messages are actionable** — never return generic "Internal Server Error"; always include what failed and what to try
10. **One collection, all versions** — do not create separate ChromaDB collections per version; use metadata filtering

---

*Build prompt version 1.0 · Samsung Browser VOC Intelligence System · May 2026*
