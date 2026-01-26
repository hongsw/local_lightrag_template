"""
Source management endpoints.
"""

from fastapi import APIRouter, Depends

from ..models import (
    SourceInfo,
    SourcesResponse,
    FileInfo,
    LocalFilesResponse,
    StatsResponse,
    HealthResponse,
)
from ...rag_engine import RAGEngine
from ...connectors.local_files import LocalFilesConnector
from ...connectors.google_drive import GoogleDriveConnector
from ...connectors.slack import SlackConnector
from ...connectors.notion import NotionConnector
from ...connectors.base import SourceType
from ...config import get_settings
from ... import __version__
from datetime import datetime

router = APIRouter(tags=["Sources & Management"])


def get_rag_engine():
    """Dependency to get RAG engine instance."""
    raise NotImplementedError("RAG engine not configured")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Health check endpoint.

    Returns system status and component availability.
    """
    settings = get_settings()

    # Check components
    components = {
        "rag_engine": True,
        "openai_configured": bool(settings.openai_api_key),
        "local_files_path": LocalFilesConnector(settings.local_files_path).validate_config()[0],
    }

    # Check optional connectors
    if settings.google_credentials_path:
        components["google_drive"] = GoogleDriveConnector(
            credentials_path=settings.google_credentials_path,
            folder_id=settings.google_drive_folder_id,
        ).validate_config()[0]

    if settings.slack_bot_token:
        components["slack"] = SlackConnector(
            slack_token=settings.slack_bot_token,
            channel_ids=settings.slack_channels,
        ).validate_config()[0]

    if settings.notion_api_key:
        components["notion"] = NotionConnector(
            notion_token=settings.notion_api_key,
            database_ids=settings.notion_databases,
        ).validate_config()[0]

    all_healthy = all([
        components.get("rag_engine", False),
        components.get("openai_configured", False),
    ])

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now(),
        version=__version__,
        components=components,
    )


@router.get("/sources", response_model=SourcesResponse)
async def list_sources():
    """
    List all available data source connectors.

    Shows configuration status for each connector.
    """
    settings = get_settings()

    sources = []

    # Local files (always available)
    local_connector = LocalFilesConnector(settings.local_files_path)
    local_info = local_connector.get_info()
    sources.append(SourceInfo(
        name=local_info.name,
        source_type=local_info.source_type.value,
        description=local_info.description,
        is_configured=local_info.is_configured,
        config_requirements=local_info.config_requirements,
    ))

    # Google Drive
    gdrive_connector = GoogleDriveConnector(
        credentials_path=settings.google_credentials_path,
        folder_id=settings.google_drive_folder_id,
    )
    gdrive_info = gdrive_connector.get_info()
    sources.append(SourceInfo(
        name=gdrive_info.name,
        source_type=gdrive_info.source_type.value,
        description=gdrive_info.description,
        is_configured=gdrive_info.is_configured,
        config_requirements=gdrive_info.config_requirements,
    ))

    # Slack
    slack_connector = SlackConnector(
        slack_token=settings.slack_bot_token,
        channel_ids=settings.slack_channels,
    )
    slack_info = slack_connector.get_info()
    sources.append(SourceInfo(
        name=slack_info.name,
        source_type=slack_info.source_type.value,
        description=slack_info.description,
        is_configured=slack_info.is_configured,
        config_requirements=slack_info.config_requirements,
    ))

    # Notion
    notion_connector = NotionConnector(
        notion_token=settings.notion_api_key,
        database_ids=settings.notion_databases,
    )
    notion_info = notion_connector.get_info()
    sources.append(SourceInfo(
        name=notion_info.name,
        source_type=notion_info.source_type.value,
        description=notion_info.description,
        is_configured=notion_info.is_configured,
        config_requirements=notion_info.config_requirements,
    ))

    return SourcesResponse(
        sources=sources,
        total=len(sources),
    )


@router.get("/sources/local/files", response_model=LocalFilesResponse)
async def list_local_files():
    """
    List all supported files in the local files directory.
    """
    settings = get_settings()

    connector = LocalFilesConnector(settings.local_files_path)

    is_valid, error = connector.validate_config()
    if not is_valid:
        return LocalFilesResponse(
            files=[],
            total=0,
            base_path=settings.local_files_path,
        )

    files = connector.list_files()

    return LocalFilesResponse(
        files=[FileInfo(**f) for f in files],
        total=len(files),
        base_path=settings.local_files_path,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Get system statistics.

    Returns information about indexed documents and configuration.
    """
    stats = rag_engine.get_stats()

    available_sources = [SourceType.LOCAL_FILES.value]

    settings = get_settings()
    if settings.google_credentials_path:
        available_sources.append(SourceType.GOOGLE_DRIVE.value)
    if settings.slack_bot_token:
        available_sources.append(SourceType.SLACK.value)
    if settings.notion_api_key:
        available_sources.append(SourceType.NOTION.value)

    return StatsResponse(
        indexed_documents=stats["indexed_documents"],
        working_dir=stats["working_dir"],
        model_name=stats["model_name"],
        embedding_model=stats["embedding_model"],
        is_initialized=stats["is_initialized"],
        available_sources=available_sources,
        tracked_files=stats.get("tracked_files", 0),
        tracked_size_bytes=stats.get("tracked_size_bytes", 0),
    )
