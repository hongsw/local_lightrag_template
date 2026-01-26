"""
Slack connector using LlamaHub SlackReader.

Prerequisites:
1. Create Slack App at api.slack.com
2. Add Bot Token Scopes: channels:history, channels:read, files:read
3. Install app to workspace
4. Get Bot User OAuth Token
"""

from typing import Optional
import hashlib

from .base import BaseConnector, ConnectorInfo, Document, SourceType


class SlackConnector(BaseConnector):
    """
    Connector for loading messages from Slack channels.

    Requires:
    - llama-index-readers-slack package
    - Slack Bot Token with appropriate scopes
    """

    def __init__(
        self,
        slack_token: Optional[str] = None,
        channel_ids: Optional[list[str]] = None,
    ):
        """
        Initialize Slack connector.

        Args:
            slack_token: Slack Bot User OAuth Token.
            channel_ids: List of Slack channel IDs to read from.
        """
        self.slack_token = slack_token
        self.channel_ids = channel_ids or []
        self._reader = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.SLACK

    def _get_reader(self):
        """Lazy load the Slack reader."""
        if self._reader is None:
            try:
                from llama_index.readers.slack import SlackReader
                self._reader = SlackReader(slack_token=self.slack_token)
            except ImportError:
                raise ImportError(
                    "Slack reader not installed. "
                    "Run: pip install llama-index-readers-slack"
                )
        return self._reader

    def load_documents(self) -> list[Document]:
        """
        Load messages from Slack channels.

        Returns:
            List of Document objects.
        """
        is_valid, error = self.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")

        reader = self._get_reader()
        llama_docs = reader.load_data(channel_ids=self.channel_ids)

        documents = []
        for idx, llama_doc in enumerate(llama_docs):
            doc_id = self._generate_doc_id(llama_doc, idx)

            metadata = dict(llama_doc.metadata) if llama_doc.metadata else {}

            documents.append(Document(
                text=llama_doc.text,
                doc_id=doc_id,
                source_type=SourceType.SLACK,
                source_path=metadata.get("channel_id", "slack://unknown"),
                source_name=metadata.get("channel_name", f"slack_message_{idx}"),
                metadata=metadata,
            ))

        return documents

    def _generate_doc_id(self, llama_doc, idx: int) -> str:
        """Generate document ID."""
        if llama_doc.metadata:
            channel_id = llama_doc.metadata.get("channel_id", "")
            ts = llama_doc.metadata.get("ts", "")
            if channel_id and ts:
                return f"slack_{channel_id}_{ts}"[:24]

        hash_input = f"slack_{idx}_{llama_doc.text[:100]}".encode()
        return f"slack_{hashlib.md5(hash_input).hexdigest()[:12]}"

    def get_info(self) -> ConnectorInfo:
        """Get connector information."""
        is_configured = bool(self.slack_token and self.channel_ids)

        return ConnectorInfo(
            name="Slack",
            source_type=SourceType.SLACK,
            description="Load messages from Slack channels",
            is_configured=is_configured,
            config_requirements=[
                "SLACK_BOT_TOKEN (Bot User OAuth Token)",
                "SLACK_CHANNEL_IDS (comma-separated channel IDs)",
            ],
        )

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate Slack configuration."""
        if not self.slack_token:
            return False, "Slack token not configured"
        if not self.channel_ids:
            return False, "No channel IDs configured"
        return True, None
