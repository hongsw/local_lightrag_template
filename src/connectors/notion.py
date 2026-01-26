"""
Notion connector using LlamaHub NotionPageReader.

Prerequisites:
1. Create Notion Integration at notion.so/my-integrations
2. Get Internal Integration Token
3. Share target pages/databases with the integration
"""

from typing import Optional
import hashlib

from .base import BaseConnector, ConnectorInfo, Document, SourceType


class NotionConnector(BaseConnector):
    """
    Connector for loading pages from Notion.

    Requires:
    - llama-index-readers-notion package
    - Notion Integration Token
    """

    def __init__(
        self,
        notion_token: Optional[str] = None,
        database_ids: Optional[list[str]] = None,
        page_ids: Optional[list[str]] = None,
    ):
        """
        Initialize Notion connector.

        Args:
            notion_token: Notion Internal Integration Token.
            database_ids: List of Notion database IDs to read from.
            page_ids: List of specific page IDs to read.
        """
        self.notion_token = notion_token
        self.database_ids = database_ids or []
        self.page_ids = page_ids or []
        self._reader = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.NOTION

    def _get_reader(self):
        """Lazy load the Notion reader."""
        if self._reader is None:
            try:
                from llama_index.readers.notion import NotionPageReader
                self._reader = NotionPageReader(integration_token=self.notion_token)
            except ImportError:
                raise ImportError(
                    "Notion reader not installed. "
                    "Run: pip install llama-index-readers-notion"
                )
        return self._reader

    def load_documents(self) -> list[Document]:
        """
        Load pages from Notion databases or specific pages.

        Returns:
            List of Document objects.
        """
        is_valid, error = self.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")

        reader = self._get_reader()
        all_docs = []

        # Load from databases
        for db_id in self.database_ids:
            try:
                llama_docs = reader.load_data(database_id=db_id)
                all_docs.extend(llama_docs)
            except Exception as e:
                print(f"Warning: Failed to load database {db_id}: {e}")

        # Load specific pages
        if self.page_ids:
            try:
                llama_docs = reader.load_data(page_ids=self.page_ids)
                all_docs.extend(llama_docs)
            except Exception as e:
                print(f"Warning: Failed to load pages: {e}")

        documents = []
        for idx, llama_doc in enumerate(all_docs):
            doc_id = self._generate_doc_id(llama_doc, idx)
            metadata = dict(llama_doc.metadata) if llama_doc.metadata else {}

            documents.append(Document(
                text=llama_doc.text,
                doc_id=doc_id,
                source_type=SourceType.NOTION,
                source_path=metadata.get("page_id", "notion://unknown"),
                source_name=metadata.get("title", f"notion_page_{idx}"),
                metadata=metadata,
            ))

        return documents

    def _generate_doc_id(self, llama_doc, idx: int) -> str:
        """Generate document ID."""
        if llama_doc.metadata and "page_id" in llama_doc.metadata:
            page_id = llama_doc.metadata["page_id"].replace("-", "")
            return f"notion_{page_id[:12]}"

        hash_input = f"notion_{idx}_{llama_doc.text[:100]}".encode()
        return f"notion_{hashlib.md5(hash_input).hexdigest()[:12]}"

    def get_info(self) -> ConnectorInfo:
        """Get connector information."""
        is_configured = bool(
            self.notion_token and (self.database_ids or self.page_ids)
        )

        return ConnectorInfo(
            name="Notion",
            source_type=SourceType.NOTION,
            description="Load pages from Notion databases",
            is_configured=is_configured,
            config_requirements=[
                "NOTION_API_KEY (Integration Token)",
                "NOTION_DATABASE_IDS (comma-separated database IDs)",
            ],
        )

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate Notion configuration."""
        if not self.notion_token:
            return False, "Notion token not configured"
        if not self.database_ids and not self.page_ids:
            return False, "No database IDs or page IDs configured"
        return True, None
