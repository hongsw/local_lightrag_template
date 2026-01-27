"""
Configuration management for the OKT-RAG system.

Supports multi-embedding slots, model-agnostic configuration,
and observability settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field


class EmbeddingSlotConfig(BaseModel):
    """Configuration for an embedding slot."""

    name: str = Field(description="Unique slot identifier")
    provider: str = Field(default="openai", description="Provider type")
    model: str = Field(description="Model identifier")
    dimension: int = Field(description="Embedding dimension")
    weight: float = Field(default=1.0, description="Weight for hybrid search")
    enabled: bool = Field(default=True, description="Whether slot is active")
    description: str = Field(default="", description="Human-readable description")


class ObservabilityConfig(BaseModel):
    """Observability and logging configuration."""

    enabled: bool = Field(default=True, description="Enable retrieval logging")
    log_embeddings: bool = Field(default=True, description="Log embedding operations")
    log_searches: bool = Field(default=True, description="Log search operations")
    log_llm_calls: bool = Field(default=True, description="Log LLM operations")
    storage_path: str = Field(default="./logs/retrieval", description="Log storage path")
    buffer_size: int = Field(default=100, description="Log buffer size before flush")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # Multi-Embedding Slot Configuration
    embedding_slots: list[EmbeddingSlotConfig] = Field(
        default=[
            EmbeddingSlotConfig(
                name="semantic",
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
                weight=1.0,
                description="Primary semantic embedding (full dimension)",
            ),
            EmbeddingSlotConfig(
                name="semantic_fast",
                provider="openai",
                model="text-embedding-3-small",
                dimension=512,
                weight=0.8,
                description="Fast semantic embedding (Matryoshka 512D)",
            ),
        ],
        description="Embedding slot configurations",
    )

    # Default embedding slot for queries
    default_embedding_slot: str = Field(
        default="semantic",
        alias="DEFAULT_EMBEDDING_SLOT",
        description="Default slot for queries without explicit slot specification",
    )

    # Observability Configuration
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="Observability and logging settings",
    )

    # Enable multi-embedding mode
    multi_embedding_enabled: bool = Field(
        default=True,
        alias="MULTI_EMBEDDING_ENABLED",
        description="Enable multi-embedding store (vs legacy single embedding)",
    )

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
