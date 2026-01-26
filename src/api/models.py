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


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    mode: str
    sources: list[dict[str, Any]] = []
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
    documents_failed: int
    error_details: list[dict[str, Any]] = []
    message: str


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


# ============== Health Models ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    components: dict[str, bool]
