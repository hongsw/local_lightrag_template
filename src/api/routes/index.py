"""
Indexing endpoints for various data sources.
"""

from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    IndexLocalRequest,
    IndexGoogleDriveRequest,
    IndexSlackRequest,
    IndexNotionRequest,
    IndexResponse,
)
from ...rag_engine import RAGEngine
from ...processor import DocumentProcessor
from ...connectors.local_files import LocalFilesConnector
from ...connectors.google_drive import GoogleDriveConnector
from ...connectors.slack import SlackConnector
from ...connectors.notion import NotionConnector
from ...config import get_settings

router = APIRouter(prefix="/index", tags=["Indexing"])


def get_rag_engine():
    """Dependency to get RAG engine instance."""
    raise NotImplementedError("RAG engine not configured")


def get_processor():
    """Dependency to get document processor."""
    settings = get_settings()
    from ...processor import ChunkConfig
    return DocumentProcessor(ChunkConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ))


@router.post("/local", response_model=IndexResponse)
async def index_local_files(
    request: IndexLocalRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    processor: DocumentProcessor = Depends(get_processor),
):
    """
    Index local files (PDF, TXT, MD).

    Scans the specified directory for supported files and indexes them.
    """
    try:
        connector = LocalFilesConnector(base_path=request.path)

        # Validate configuration
        is_valid, error = connector.validate_config()
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        # Load documents
        documents = connector.load_documents()

        if not documents:
            return IndexResponse(
                success=True,
                source_type="local_files",
                documents_indexed=0,
                documents_failed=0,
                message="No documents found in the specified path",
            )

        # Process (chunk) documents
        processed_docs = processor.process_documents(documents)

        # Index into LightRAG
        result = await rag_engine.index_documents(processed_docs)

        return IndexResponse(
            success=True,
            source_type="local_files",
            documents_indexed=result["indexed"],
            documents_failed=result["errors"],
            error_details=result.get("error_details", []),
            message=f"Successfully indexed {result['indexed']} document chunks from {len(documents)} files",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
        )


@router.post("/google-drive", response_model=IndexResponse)
async def index_google_drive(
    request: IndexGoogleDriveRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    processor: DocumentProcessor = Depends(get_processor),
):
    """
    Index documents from Google Drive.

    Requires Google Cloud credentials and folder access.
    """
    settings = get_settings()

    try:
        credentials_path = request.credentials_path or settings.google_credentials_path

        connector = GoogleDriveConnector(
            credentials_path=credentials_path,
            folder_id=request.folder_id,
        )

        is_valid, error = connector.validate_config()
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        documents = connector.load_documents()

        if not documents:
            return IndexResponse(
                success=True,
                source_type="google_drive",
                documents_indexed=0,
                documents_failed=0,
                message="No documents found in the Google Drive folder",
            )

        processed_docs = processor.process_documents(documents)
        result = await rag_engine.index_documents(processed_docs)

        return IndexResponse(
            success=True,
            source_type="google_drive",
            documents_indexed=result["indexed"],
            documents_failed=result["errors"],
            error_details=result.get("error_details", []),
            message=f"Successfully indexed {result['indexed']} chunks from Google Drive",
        )

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive indexing failed: {str(e)}"
        )


@router.post("/slack", response_model=IndexResponse)
async def index_slack(
    request: IndexSlackRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    processor: DocumentProcessor = Depends(get_processor),
):
    """
    Index messages from Slack channels.

    Requires Slack Bot Token with appropriate permissions.
    """
    settings = get_settings()

    try:
        token = request.token or settings.slack_bot_token

        connector = SlackConnector(
            slack_token=token,
            channel_ids=request.channel_ids,
        )

        is_valid, error = connector.validate_config()
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        documents = connector.load_documents()

        if not documents:
            return IndexResponse(
                success=True,
                source_type="slack",
                documents_indexed=0,
                documents_failed=0,
                message="No messages found in the specified channels",
            )

        processed_docs = processor.process_documents(documents)
        result = await rag_engine.index_documents(processed_docs)

        return IndexResponse(
            success=True,
            source_type="slack",
            documents_indexed=result["indexed"],
            documents_failed=result["errors"],
            error_details=result.get("error_details", []),
            message=f"Successfully indexed {result['indexed']} chunks from Slack",
        )

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Slack indexing failed: {str(e)}"
        )


@router.post("/notion", response_model=IndexResponse)
async def index_notion(
    request: IndexNotionRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    processor: DocumentProcessor = Depends(get_processor),
):
    """
    Index pages from Notion.

    Requires Notion Integration Token with page access.
    """
    settings = get_settings()

    try:
        token = request.token or settings.notion_api_key

        connector = NotionConnector(
            notion_token=token,
            database_ids=request.database_ids,
            page_ids=request.page_ids,
        )

        is_valid, error = connector.validate_config()
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        documents = connector.load_documents()

        if not documents:
            return IndexResponse(
                success=True,
                source_type="notion",
                documents_indexed=0,
                documents_failed=0,
                message="No pages found in the specified databases/pages",
            )

        processed_docs = processor.process_documents(documents)
        result = await rag_engine.index_documents(processed_docs)

        return IndexResponse(
            success=True,
            source_type="notion",
            documents_indexed=result["indexed"],
            documents_failed=result["errors"],
            error_details=result.get("error_details", []),
            message=f"Successfully indexed {result['indexed']} chunks from Notion",
        )

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Notion indexing failed: {str(e)}"
        )


@router.delete("/clear")
async def clear_index(
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Clear all indexed data.

    Warning: This permanently deletes all indexed documents.
    """
    success = rag_engine.clear_index()

    if success:
        return {"success": True, "message": "Index cleared successfully"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to clear index"
        )
