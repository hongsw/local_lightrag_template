"""
Indexing endpoints for various data sources.
"""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    IndexLocalRequest,
    IndexGoogleDriveRequest,
    IndexSlackRequest,
    IndexNotionRequest,
    IndexResponse,
    IndexedFilesResponse,
    IndexedFileInfo,
    FileStatusRequest,
    FileStatusResponse,
    FileStatusInfo,
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

        skipped = result.get("skipped", 0)
        message = f"Indexed {result['indexed']} chunks from {len(documents)} files"
        if skipped > 0:
            message += f" (skipped {skipped} already indexed)"

        return IndexResponse(
            success=True,
            source_type="local_files",
            documents_indexed=result["indexed"],
            documents_skipped=skipped,
            documents_failed=result["errors"],
            error_details=result.get("error_details", []),
            message=message,
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


@router.get("/files", response_model=IndexedFilesResponse)
async def list_indexed_files(
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    List all indexed files.

    Returns information about files that have been indexed,
    including file name, path, size, and indexing timestamp.
    """
    files = rag_engine.get_indexed_files()
    total_size = sum(f["file_size"] for f in files)

    return IndexedFilesResponse(
        files=[IndexedFileInfo(**f) for f in files],
        total=len(files),
        total_size_bytes=total_size,
    )


@router.post("/status", response_model=FileStatusResponse)
async def check_file_status(
    request: FileStatusRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Check indexing status for specific files.

    Returns whether each file is indexed, needs re-indexing,
    and the reason (new_file, content_changed, etc.).
    """
    results = []
    indexed_count = 0
    pending_count = 0

    for file_path_str in request.file_paths:
        file_path = Path(file_path_str)
        is_indexed = rag_engine.is_file_indexed(file_path)
        needs_reindex, reason = rag_engine.tracker.needs_reindex(file_path)

        if is_indexed:
            indexed_count += 1
        if needs_reindex:
            pending_count += 1

        results.append(FileStatusInfo(
            file_path=file_path_str,
            is_indexed=is_indexed,
            needs_reindex=needs_reindex,
            reason=reason,
        ))

    return FileStatusResponse(
        files=results,
        total=len(results),
        indexed_count=indexed_count,
        pending_count=pending_count,
    )


@router.post("/local/sync", response_model=IndexResponse)
async def sync_local_files(
    request: IndexLocalRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    processor: DocumentProcessor = Depends(get_processor),
):
    """
    Sync local files - only index new or changed files.

    Scans the directory and indexes only files that:
    - Have not been indexed before (new files)
    - Have changed since last indexing (content_changed)

    This is more efficient than full re-indexing.
    """
    try:
        connector = LocalFilesConnector(base_path=request.path)

        is_valid, error = connector.validate_config()
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        # Load all documents
        documents = connector.load_documents()

        if not documents:
            return IndexResponse(
                success=True,
                source_type="local_files",
                documents_indexed=0,
                documents_skipped=0,
                documents_failed=0,
                message="No documents found in the specified path",
            )

        # Filter to only new/changed files
        new_docs = []
        skipped_files = 0
        for doc in documents:
            if doc.source_path:
                needs_index, _ = rag_engine.tracker.needs_reindex(Path(doc.source_path))
                if needs_index:
                    new_docs.append(doc)
                else:
                    skipped_files += 1
            else:
                new_docs.append(doc)

        if not new_docs:
            return IndexResponse(
                success=True,
                source_type="local_files",
                documents_indexed=0,
                documents_skipped=skipped_files,
                documents_failed=0,
                message=f"All {skipped_files} files are already indexed",
            )

        # Process and index only new documents
        processed_docs = processor.process_documents(new_docs)
        result = await rag_engine.index_documents(processed_docs, skip_tracked=False)

        return IndexResponse(
            success=True,
            source_type="local_files",
            documents_indexed=result["indexed"],
            documents_skipped=skipped_files,
            documents_failed=result["errors"],
            error_details=result.get("error_details", []),
            message=f"Synced: indexed {result['indexed']} chunks from {len(new_docs)} new files, skipped {skipped_files} already indexed",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sync failed: {str(e)}"
        )
