"""
Pydantic models for API request/response validation.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field

from ..rag_engine import QueryMode


# ============== Query Models ==============

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=1, description="The question to ask")
    mode: QueryMode = Field(
        default=QueryMode.HYBRID,
        description="Query mode: naive, local, global, or hybrid"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "데이터 엔지니어링이란 무엇인가요?",
                    "mode": "hybrid"
                }
            ]
        }
    }


class SourceReference(BaseModel):
    """Reference to a source document."""
    file_name: Optional[str] = Field(None, description="Name of the source file")
    page: Optional[int] = Field(None, description="Page number in the document")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the source")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0.0-1.0)")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    mode: str
    sources: list[SourceReference] = []
    metadata: dict[str, Any] = {}
    processing_time_ms: float


# ============== Index Models ==============

class IndexLocalRequest(BaseModel):
    """Request to index local files."""
    path: str = Field(..., description="Path to directory containing files")
    recursive: bool = Field(default=True, description="Scan subdirectories")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "path": "./rag_raw_pdfs",
                    "recursive": True
                }
            ]
        }
    }


class IndexGoogleDriveRequest(BaseModel):
    """Request to index Google Drive folder."""
    folder_id: str = Field(..., description="Google Drive folder ID")
    credentials_path: Optional[str] = Field(
        default=None,
        description="Path to credentials JSON (uses env var if not provided)"
    )


class IndexSlackRequest(BaseModel):
    """Request to index Slack channels."""
    channel_ids: list[str] = Field(..., description="List of Slack channel IDs")
    token: Optional[str] = Field(
        default=None,
        description="Slack bot token (uses env var if not provided)"
    )


class IndexNotionRequest(BaseModel):
    """Request to index Notion databases."""
    database_ids: list[str] = Field(
        default=[],
        description="List of Notion database IDs"
    )
    page_ids: list[str] = Field(
        default=[],
        description="List of specific Notion page IDs"
    )
    token: Optional[str] = Field(
        default=None,
        description="Notion integration token (uses env var if not provided)"
    )


class IndexResponse(BaseModel):
    """Response for indexing operations."""
    success: bool
    source_type: str
    documents_indexed: int
    documents_skipped: int = 0
    documents_failed: int
    error_details: list[dict[str, Any]] = []
    message: str


class IndexedFileInfo(BaseModel):
    """Information about an indexed file."""
    file_name: str
    file_path: str
    file_size: int
    indexed_at: str
    doc_count: int


class IndexedFilesResponse(BaseModel):
    """Response listing indexed files."""
    files: list[IndexedFileInfo]
    total: int
    total_size_bytes: int


class FileStatusRequest(BaseModel):
    """Request to check file indexing status."""
    file_paths: list[str] = Field(..., description="List of file paths to check")


class FileStatusInfo(BaseModel):
    """Status information for a single file."""
    file_path: str
    is_indexed: bool
    needs_reindex: bool
    reason: Optional[str] = None


class FileStatusResponse(BaseModel):
    """Response with file indexing status."""
    files: list[FileStatusInfo]
    total: int
    indexed_count: int
    pending_count: int


# ============== Source Models ==============

class SourceInfo(BaseModel):
    """Information about a data source connector."""
    name: str
    source_type: str
    description: str
    is_configured: bool
    config_requirements: list[str]


class SourcesResponse(BaseModel):
    """Response listing available sources."""
    sources: list[SourceInfo]
    total: int


class FileInfo(BaseModel):
    """Information about a local file."""
    name: str
    path: str
    size: int
    extension: str


class LocalFilesResponse(BaseModel):
    """Response listing local files."""
    files: list[FileInfo]
    total: int
    base_path: str


# ============== Stats Models ==============

class StatsResponse(BaseModel):
    """Response with system statistics."""
    indexed_documents: int
    working_dir: str
    model_name: str
    embedding_model: str
    is_initialized: bool
    available_sources: list[str]
    tracked_files: int = 0
    tracked_size_bytes: int = 0


# ============== Health Models ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    components: dict[str, bool]


# ============== Verification Models ==============

class VerifyRequest(BaseModel):
    """Request to verify citations in an answer."""
    answer: str = Field(..., description="The answer text with inline citations")
    sources: list[SourceReference] = Field(..., description="List of sources referenced")


class CitationVerificationResult(BaseModel):
    """Verification result for a single citation."""
    citation_number: int
    statement: str
    source_file: Optional[str]
    source_page: Optional[int]
    source_excerpt: str
    is_accurate: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class VerifyResponse(BaseModel):
    """Response from citation verification."""
    total_citations: int
    verified_count: int
    accurate_count: int
    inaccurate_count: int
    uncertain_count: int
    accuracy_rate: float = Field(..., ge=0.0, le=1.0)
    verifications: list[CitationVerificationResult]
    processing_time_ms: float


# ============== Verified Query Models ==============

class VerifiedQueryRequest(BaseModel):
    """Request model for RAG queries with automatic verification."""
    question: str = Field(..., min_length=1, description="The question to ask")
    mode: QueryMode = Field(
        default=QueryMode.HYBRID,
        description="Query mode: naive, local, global, or hybrid"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to consider a citation inaccurate"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "데이터 엔지니어링이란 무엇인가요?",
                    "mode": "hybrid",
                    "confidence_threshold": 0.7
                }
            ]
        }
    }


class VerificationLogEntry(BaseModel):
    """Log entry for a single citation verification."""
    citation_number: int
    statement: str
    source_file: Optional[str]
    source_page: Optional[int]
    is_accurate: bool
    confidence: float
    explanation: str
    status: str = Field(..., description="accurate, inaccurate, or uncertain")


class VerifiedQueryResponse(BaseModel):
    """Response model for RAG queries with automatic verification and correction."""
    original_answer: str = Field(..., description="Original answer before verification")
    corrected_answer: str = Field(..., description="Answer with inaccurate citations removed")
    mode: str
    sources: list[SourceReference] = []
    removed_citations: list[int] = Field(
        default=[],
        description="List of citation numbers that were removed due to inaccuracy"
    )
    accuracy_rate: float = Field(..., ge=0.0, le=1.0)
    verification_log: list[VerificationLogEntry] = Field(
        default=[],
        description="Detailed verification results for each citation"
    )
    metadata: dict[str, Any] = {}
    processing_time_ms: float
