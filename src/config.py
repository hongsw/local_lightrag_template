"""
Configuration management for the RAG API system.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # LightRAG Configuration
    lightrag_working_dir: str = Field(default="./lightrag_data", alias="LIGHTRAG_WORKING_DIR")

    # API Server Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")

    # Local Files Configuration
    local_files_path: str = Field(default="./rag_raw_pdfs", alias="LOCAL_FILES_PATH")

    # Optional: Google Drive Configuration
    google_credentials_path: Optional[str] = Field(default=None, alias="GOOGLE_CREDENTIALS_PATH")
    google_drive_folder_id: Optional[str] = Field(default=None, alias="GOOGLE_DRIVE_FOLDER_ID")

    # Optional: Slack Configuration
    slack_bot_token: Optional[str] = Field(default=None, alias="SLACK_BOT_TOKEN")
    slack_channel_ids: Optional[str] = Field(default=None, alias="SLACK_CHANNEL_IDS")

    # Optional: Notion Configuration
    notion_api_key: Optional[str] = Field(default=None, alias="NOTION_API_KEY")
    notion_database_ids: Optional[str] = Field(default=None, alias="NOTION_DATABASE_IDS")

    # Processing Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def slack_channels(self) -> list[str]:
        """Parse comma-separated Slack channel IDs."""
        if self.slack_channel_ids:
            return [c.strip() for c in self.slack_channel_ids.split(",")]
        return []

    @property
    def notion_databases(self) -> list[str]:
        """Parse comma-separated Notion database IDs."""
        if self.notion_database_ids:
            return [d.strip() for d in self.notion_database_ids.split(",")]
        return []


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
