"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .routes import query_router, index_router, sources_router
from .routes import query, index, sources
from ..config import get_settings
from ..rag_engine import RAGEngine

# Global RAG engine instance
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Get the global RAG engine instance."""
    if _rag_engine is None:
        raise RuntimeError("RAG engine not initialized")
    return _rag_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _rag_engine

    settings = get_settings()

    # Initialize RAG engine
    _rag_engine = RAGEngine(
        working_dir=settings.lightrag_working_dir,
        openai_api_key=settings.openai_api_key,
        model_name=settings.openai_model,
        embedding_model=settings.embedding_model,
    )

    print(f"RAG API Server initialized")
    print(f"  Working directory: {settings.lightrag_working_dir}")
    print(f"  Local files path: {settings.local_files_path}")
    print(f"  Model: {settings.openai_model}")

    yield

    # Cleanup
    _rag_engine = None
    print("RAG API Server shutdown")


# Create FastAPI application
app = FastAPI(
    title="LlamaHub + LightRAG API",
    description="""
Universal RAG API server powered by LlamaHub connectors and LightRAG engine.

## Features
- **Multi-source ingestion**: Local files, Google Drive, Slack, Notion
- **Knowledge Graph RAG**: LightRAG combines vector search with knowledge graphs
- **Multiple query modes**: naive, local, global, hybrid

## Getting Started
1. Index documents using `/index/local` endpoint
2. Query using `/query` endpoint with your questions
""",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use FastAPI's dependency override mechanism
app.dependency_overrides[query.get_rag_engine] = get_rag_engine
app.dependency_overrides[index.get_rag_engine] = get_rag_engine
app.dependency_overrides[sources.get_rag_engine] = get_rag_engine

# Include routers
app.include_router(query_router)
app.include_router(index_router)
app.include_router(sources_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LlamaHub + LightRAG API",
        "version": "0.2.0",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "endpoints": {
            "query": "POST /query",
            "index_local": "POST /index/local",
            "index_sync": "POST /index/local/sync",
            "index_files": "GET /index/files",
            "index_gdrive": "POST /index/google-drive",
            "index_slack": "POST /index/slack",
            "index_notion": "POST /index/notion",
            "sources": "GET /sources",
            "stats": "GET /stats",
            "health": "GET /health",
        }
    }


@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard HTML page."""
    # Try multiple possible locations for dashboard.html
    possible_paths = [
        Path(__file__).parent.parent.parent / "dashboard.html",  # Project root
        Path("/app/dashboard.html"),  # Docker container
        Path("dashboard.html"),  # Current directory
    ]

    for dashboard_path in possible_paths:
        if dashboard_path.exists():
            return FileResponse(dashboard_path, media_type="text/html")

    return {"error": "Dashboard not found", "tried_paths": [str(p) for p in possible_paths]}


def run_server():
    """Run the API server."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    run_server()
