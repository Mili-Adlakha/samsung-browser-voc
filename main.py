import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routers import chat, dashboard, ingest

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Samsung Browser VOC Intelligence System",
    description="RAG-based VOC analysis pipeline for Samsung Browser Play Store reviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "service": "Samsung Browser VOC Intelligence System"}


@app.get("/")
def ui():
    """Serve the interactive VOC web UI."""
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "message": "UI not found. Static files missing.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api")
def api_index():
    return {
        "endpoints": {
            "ingest": "POST /api/v1/ingest",
            "chat": "POST /api/v1/chat",
            "dashboard": "POST /api/v1/dashboard",
            "docs": "/docs",
            "health": "/health",
            "ui": "/",
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
