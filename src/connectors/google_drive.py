"""
Google Drive connector using LlamaHub GoogleDriveReader.

Prerequisites:
1. Create a Google Cloud project
2. Enable Google Drive API
3. Create Service Account and download credentials.json
4. Share target folder with Service Account email
"""

from pathlib import Path
from typing import Optional
import hashlib

from .base import BaseConnector, ConnectorInfo, Document, SourceType


class GoogleDriveConnector(BaseConnector):
    """
    Connector for loading documents from Google Drive.

    Requires:
    - llama-index-readers-google package
    - Google Cloud credentials (service account JSON)
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        folder_id: Optional[str] = None,
    ):
        """
        Initialize Google Drive connector.

        Args:
            credentials_path: Path to Google Cloud service account JSON.
            folder_id: Google Drive folder ID to read from.
        """
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self.folder_id = folder_id
        self._reader = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.GOOGLE_DRIVE

    def _get_reader(self):
        """Lazy load the Google Drive reader."""
        if self._reader is None:
            try:
                from llama_index.readers.google import GoogleDriveReader
                self._reader = GoogleDriveReader(
                    credentials_path=str(self.credentials_path)
                )
            except ImportError:
                raise ImportError(
                    "Google Drive reader not installed. "
                    "Run: pip install llama-index-readers-google"
                )
        return self._reader

    def load_documents(self) -> list[Document]:
        """
        Load documents from Google Drive folder.

        Returns:
            List of Document objects.
        """
        is_valid, error = self.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")

        reader = self._get_reader()
        llama_docs = reader.load_data(folder_id=self.folder_id)

        documents = []
        for idx, llama_doc in enumerate(llama_docs):
            doc_id = self._generate_doc_id(llama_doc, idx)

            # Extract metadata from LlamaIndex document
            metadata = dict(llama_doc.metadata) if llama_doc.metadata else {}

            documents.append(Document(
                text=llama_doc.text,
                doc_id=doc_id,
                source_type=SourceType.GOOGLE_DRIVE,
                source_path=metadata.get("file_path", f"gdrive://{self.folder_id}"),
                source_name=metadata.get("file_name", f"document_{idx}"),
                metadata=metadata,
            ))

        return documents

    def _generate_doc_id(self, llama_doc, idx: int) -> str:
        """Generate document ID from content or metadata."""
        if llama_doc.metadata and "file_id" in llama_doc.metadata:
            return f"gdrive_{llama_doc.metadata['file_id'][:12]}"

        hash_input = f"{self.folder_id}_{idx}_{llama_doc.text[:100]}".encode()
        return f"gdrive_{hashlib.md5(hash_input).hexdigest()[:12]}"

    def get_info(self) -> ConnectorInfo:
        """Get connector information."""
        is_configured = (
            self.credentials_path is not None
            and self.credentials_path.exists()
            and self.folder_id is not None
        )

        return ConnectorInfo(
            name="Google Drive",
            source_type=SourceType.GOOGLE_DRIVE,
            description="Load documents from Google Drive folders",
            is_configured=is_configured,
            config_requirements=[
                "GOOGLE_CREDENTIALS_PATH (service account JSON path)",
                "GOOGLE_DRIVE_FOLDER_ID (folder to read from)",
            ],
        )

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate Google Drive configuration."""
        if not self.credentials_path:
            return False, "Credentials path not configured"
        if not self.credentials_path.exists():
            return False, f"Credentials file not found: {self.credentials_path}"
        if not self.folder_id:
            return False, "Folder ID not configured"
        return True, None
