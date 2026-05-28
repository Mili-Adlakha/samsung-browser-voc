# Samsung Browser VOC Intelligence System

A FastAPI backend that ingests Google Play Store review text for Samsung Browser, stores semantic embeddings in local ChromaDB, answers product-manager questions via a RAG chatbot (Claude), and generates a self-contained HTML analytics dashboard from the same corpus.

## Requirements

- Python **3.11** or **3.12** (3.13 is not supported by pinned `pydantic==2.7.0`)
- OpenAI API key (embeddings)
- Anthropic API key (chat + dashboard)

## Setup

```bash
cd "VOC analyser"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and OPENAI_API_KEY
uvicorn main:app --reload --port 8000
```

- **Web UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — upload VOCs, chat, generate dashboard
- **API docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for `/chat` and `/dashboard` |
| `OPENAI_API_KEY` | OpenAI key for `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | Chroma persistence path (default `./chroma_db`) |
| `COLLECTION_NAME` | Collection name (default `samsung_browser_voc`) |
| `PORT` | Uvicorn port (default `8000`) |

## Usage

### 1. Ingest Play Store reviews

Paste raw review text (browser copy-paste, CSV export, or numbered format):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": "John Doe\n★★★★★\nGreat browser, fast and clean. Ad blocking works perfectly.\n3 people found this review helpful\n\nJane Smith\n★\nLatest update broke everything. Lost all my tabs after sync.\n74 people found this review helpful",
    "app_version": "30.XX",
    "date_range": "20-25 May 2026"
  }'
```

### 2. Ask PM questions (RAG chat)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top complaints about tab sync?",
    "version_filter": "30.XX",
    "top_k": 20
  }'
```

### 3. Generate VOC dashboard HTML

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/dashboard \
  -H "Content-Type: application/json" \
  -d '{
    "app_version": "30.XX",
    "date_range": "20-25 May 2026",
    "top_k": 100
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['html'])" > dashboard.html
```

Open `dashboard.html` in a browser.

## Architecture

```
Play Store paste / CSV
        │
        ▼
   POST /ingest ──► parser ──► chunker ──► OpenAI embeddings ──► ChromaDB
                                                                    │
                    ┌───────────────────────────────────────────────┤
                    ▼                                               ▼
             POST /chat                                    POST /dashboard
         retriever + Claude                           analytics + Claude
         (claude-sonnet-4-5)                          (claude-sonnet-4-6)
                    │                                               │
                    ▼                                               ▼
              JSON answer                                    HTML file
```

## Project layout

```
├── main.py                 # FastAPI app
├── routers/                # ingest, chat, dashboard
├── services/               # parser, embedder, vector_store, retriever, llm
├── models/schemas.py       # Pydantic models
├── prompts/                # voc_analyst.txt, dashboard_gen.txt
├── tests/                  # pytest suite (no live LLM calls)
└── chroma_db/              # created at runtime (gitignored)
```

## Tests

```bash
pytest
```

All tests mock external LLM and embedding APIs.

## Known limitations

- **Single collection** — all app versions share one Chroma collection; filter by `version_filter` / metadata, not separate DBs.
- **Dashboard** — Uses a fast local HTML template by default (seconds). Set `USE_LLM_DASHBOARD=true` in `.env` for Claude-generated HTML (slower; timeout `DASHBOARD_TIMEOUT_SECONDS`, default 180).
- **Parser heuristics** — Play Store paste formats vary; unsupported layouts may be skipped with warnings.
- **No auth** — API is open on localhost; add authentication before any public deployment.
- **Chroma persistence** — data survives restarts when `CHROMA_PERSIST_DIR` is set; deleting that folder clears the corpus.

## License

Internal Samsung Browser PM tooling — adjust as needed for your org.
