"""
Query endpoints for RAG operations.
"""

import time
from fastapi import APIRouter, Depends, HTTPException

from ..models import QueryRequest, QueryResponse
from ...rag_engine import RAGEngine, QueryMode

router = APIRouter(prefix="/query", tags=["Query"])


def get_rag_engine():
    """Dependency to get RAG engine instance."""
    # This will be overridden in main.py
    raise NotImplementedError("RAG engine not configured")


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Query the RAG system.

    Supports multiple query modes:
    - **naive**: Simple vector similarity search
    - **local**: Local context from knowledge graph relationships
    - **global**: Global patterns across the knowledge graph
    - **hybrid**: Combines local and global (recommended)
    """
    start_time = time.time()

    try:
        result = await rag_engine.query(
            question=request.question,
            mode=request.mode,
        )

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=result.answer,
            mode=result.mode.value,
            sources=result.sources,
            metadata=result.metadata,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/modes")
async def list_query_modes():
    """List available query modes with descriptions."""
    return {
        "modes": [
            {
                "name": QueryMode.NAIVE.value,
                "description": "Simple vector similarity search. Fast but less context-aware.",
            },
            {
                "name": QueryMode.LOCAL.value,
                "description": "Uses local context from knowledge graph entity relationships.",
            },
            {
                "name": QueryMode.GLOBAL.value,
                "description": "Uses global patterns across the entire knowledge graph.",
            },
            {
                "name": QueryMode.HYBRID.value,
                "description": "Combines local and global modes. Recommended for most queries.",
            },
        ]
    }
