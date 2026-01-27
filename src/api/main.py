"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .routes import query_router, index_router, sources_router, embedding_router, analytics_router
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
    title="OKT-RAG API",
    description="""
**Open Korea Tech RAG Platform** - 모델에 종속되지 않고, 실사용 로그가 곧 연구 데이터가 되는 한국형 차세대 RAG 플랫폼

## 핵심 기능

### Model-Agnostic Design
- **Multi-Embedding**: 문서 1개를 여러 임베딩 공간에 동시 저장
- **Matryoshka Support**: 품질/속도 트레이드오프를 위한 차원 축소
- **Pluggable LLM**: OpenAI, Anthropic, Ollama 등 자유로운 모델 교체

### Retrieval Observability
- **완전 로그화**: 모든 검색 의사결정 과정 기록
- **실시간 분석**: 슬롯별, 전략별 성능 비교
- **A/B 실험**: 검색 전략 실험 인프라

### Adaptive Retrieval
- **질문 유형 분류**: Factual, Analytical, Comparative 자동 분류
- **전략 자동 선택**: 질문 유형에 따른 최적 검색 전략

## 시작하기
1. `/index/local` 엔드포인트로 문서 인덱싱
2. `/query` 엔드포인트로 질문
3. `/embedding/config`에서 멀티 임베딩 설정 확인
4. `/analytics/retrieval`에서 성능 분석
""",
    version="0.4.0",
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
app.include_router(embedding_router)
app.include_router(analytics_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "OKT-RAG API",
        "version": "0.3.0",
        "description": "Open Korea Tech RAG Platform - Model-Agnostic, Multi-Embedding, Observable",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "endpoints": {
            "query": "POST /query",
            "query_verified": "POST /query/verified",
            "index_local": "POST /index/local",
            "index_sync": "POST /index/local/sync",
            "index_files": "GET /index/files",
            "index_gdrive": "POST /index/google-drive",
            "index_slack": "POST /index/slack",
            "index_notion": "POST /index/notion",
            "sources": "GET /sources",
            "embedding_config": "GET /embedding/config",
            "embedding_slots": "GET /embedding/slots",
            "analytics_retrieval": "GET /analytics/retrieval",
            "analytics_slots": "GET /analytics/slots/performance",
            "analytics_cost": "GET /analytics/cost",
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
