"""
Query endpoints for RAG operations.
"""

import time
from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    QueryRequest, QueryResponse,
    VerifyRequest, VerifyResponse, CitationVerificationResult,
    VerifiedQueryRequest, VerifiedQueryResponse, VerificationLogEntry,
)
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


@router.post("/verify", response_model=VerifyResponse)
async def verify_citations(
    request: VerifyRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Verify that citations in an answer accurately reflect the source content.

    This endpoint analyzes each citation in the answer and checks if:
    - The statement is supported by the source content
    - Facts are not distorted or exaggerated
    - No information is added that doesn't exist in the source
    """
    start_time = time.time()

    try:
        # Convert SourceReference to dict format, filtering out empty excerpts
        sources_dict = [
            {
                "file_name": s.file_name,
                "page": s.page,
                "excerpt": s.excerpt or "",
            }
            for s in request.sources
            if s.excerpt and len(s.excerpt) >= 20  # Filter out empty or too short excerpts
        ]

        if not sources_dict:
            # If all sources were filtered out, return empty result
            processing_time = (time.time() - start_time) * 1000
            return VerifyResponse(
                total_citations=0,
                verified_count=0,
                accurate_count=0,
                inaccurate_count=0,
                uncertain_count=0,
                accuracy_rate=0.0,
                verifications=[],
                processing_time_ms=round(processing_time, 2),
            )

        result = await rag_engine.verify_citations(
            answer=request.answer,
            sources=sources_dict,
        )

        processing_time = (time.time() - start_time) * 1000

        return VerifyResponse(
            total_citations=result.total_citations,
            verified_count=result.verified_count,
            accurate_count=result.accurate_count,
            inaccurate_count=result.inaccurate_count,
            uncertain_count=result.uncertain_count,
            accuracy_rate=result.accuracy_rate,
            verifications=[
                CitationVerificationResult(
                    citation_number=v.citation_number,
                    statement=v.statement,
                    source_file=v.source_file,
                    source_page=v.source_page,
                    source_excerpt=v.source_excerpt,
                    is_accurate=v.is_accurate,
                    confidence=v.confidence,
                    explanation=v.explanation,
                )
                for v in result.verifications
            ],
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@router.post("/verified", response_model=VerifiedQueryResponse)
async def query_with_verification(
    request: VerifiedQueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Query the RAG system with automatic citation verification.

    This endpoint:
    1. Runs the normal RAG query
    2. Verifies each citation against source content
    3. Removes inaccurate citations from the answer
    4. Returns the corrected answer with verification log

    Use this for higher accuracy answers with automatic fact-checking.
    """
    start_time = time.time()

    try:
        result = await rag_engine.query_with_verification(
            question=request.question,
            mode=request.mode,
            confidence_threshold=request.confidence_threshold,
        )

        processing_time = (time.time() - start_time) * 1000

        # Convert verification log to response format
        verification_log = []
        for log_entry in result.verification_log:
            status = "accurate" if log_entry.get("is_accurate") else "inaccurate"
            if log_entry.get("confidence", 0) < request.confidence_threshold:
                status = "uncertain"

            verification_log.append(VerificationLogEntry(
                citation_number=log_entry.get("citation_number", 0),
                statement=log_entry.get("statement", ""),
                source_file=log_entry.get("source_file"),
                source_page=log_entry.get("source_page"),
                is_accurate=log_entry.get("is_accurate", False),
                confidence=log_entry.get("confidence", 0.0),
                explanation=log_entry.get("explanation", ""),
                status=status,
            ))

        # Convert sources to SourceReference format
        from ..models import SourceReference
        sources = [
            SourceReference(
                file_name=s.get("file_name"),
                page=s.get("page"),
                excerpt=s.get("excerpt"),
                relevance_score=s.get("relevance_score"),
            )
            for s in result.sources
        ]

        return VerifiedQueryResponse(
            original_answer=result.original_answer,
            corrected_answer=result.corrected_answer,
            mode=result.mode.value,
            sources=sources,
            removed_citations=result.removed_citations,
            accuracy_rate=result.accuracy_rate,
            verification_log=verification_log,
            metadata=result.metadata,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verified query failed: {str(e)}"
        )
