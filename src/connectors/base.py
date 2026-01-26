"""
Base connector interface for all data source connectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum


class SourceType(str, Enum):
    """Supported data source types."""
    LOCAL_FILES = "local_files"
    GOOGLE_DRIVE = "google_drive"
    SLACK = "slack"
    NOTION = "notion"
    S3 = "s3"


@dataclass
class Document:
    """Unified document representation across all connectors."""

    text: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Source information
    source_type: SourceType = SourceType.LOCAL_FILES
    source_path: Optional[str] = None
    source_name: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure metadata includes source info."""
        self.metadata.update({
            "source_type": self.source_type.value,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "doc_id": self.doc_id,
        })


@dataclass
class ConnectorInfo:
    """Information about a connector."""
    name: str
    source_type: SourceType
    description: str
    is_configured: bool
    config_requirements: list[str]


class BaseConnector(ABC):
    """
    Abstract base class for all data source connectors.

    All connectors must implement:
    - load_documents(): Load and return documents from the source
    - get_info(): Return connector metadata
    """

    @abstractmethod
    def load_documents(self) -> list[Document]:
        """
        Load documents from the data source.

        Returns:
            List of Document objects with text and metadata.
        """
        pass

    @abstractmethod
    def get_info(self) -> ConnectorInfo:
        """
        Get connector metadata and configuration status.

        Returns:
            ConnectorInfo with name, type, and configuration status.
        """
        pass

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Return the source type for this connector."""
        pass

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate connector configuration.

        Returns:
            Tuple of (is_valid, error_message).
        """
        return True, None
