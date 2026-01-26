"""
Tests for data source connectors.
"""

import pytest
from pathlib import Path

from src.connectors.local_files import LocalFilesConnector
from src.connectors.base import SourceType, Document


class TestLocalFilesConnector:
    """Tests for LocalFilesConnector."""

    def test_init(self, tmp_path):
        """Test connector initialization."""
        connector = LocalFilesConnector(tmp_path)
        assert connector.base_path == tmp_path
        assert connector.source_type == SourceType.LOCAL_FILES

    def test_validate_config_valid(self, tmp_path):
        """Test validation with valid path."""
        connector = LocalFilesConnector(tmp_path)
        is_valid, error = connector.validate_config()
        assert is_valid is True
        assert error is None

    def test_validate_config_invalid(self):
        """Test validation with non-existent path."""
        connector = LocalFilesConnector("/nonexistent/path")
        is_valid, error = connector.validate_config()
        assert is_valid is False
        assert "does not exist" in error

    def test_load_documents_empty(self, tmp_path):
        """Test loading from empty directory."""
        connector = LocalFilesConnector(tmp_path)
        docs = connector.load_documents()
        assert docs == []

    def test_load_documents_txt(self, tmp_path):
        """Test loading text files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        connector = LocalFilesConnector(tmp_path)
        docs = connector.load_documents()

        assert len(docs) == 1
        assert "Hello, world!" in docs[0].text
        assert docs[0].source_type == SourceType.LOCAL_FILES
        assert docs[0].source_name == "test.txt"

    def test_load_documents_md(self, tmp_path):
        """Test loading markdown files."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nContent here.")

        connector = LocalFilesConnector(tmp_path)
        docs = connector.load_documents()

        assert len(docs) == 1
        assert "# Header" in docs[0].text

    def test_list_files(self, tmp_path):
        """Test listing files."""
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.md").write_text("content2")
        (tmp_path / "file3.jpg").write_text("not supported")

        connector = LocalFilesConnector(tmp_path)
        files = connector.list_files()

        assert len(files) == 2
        names = [f["name"] for f in files]
        assert "file1.txt" in names
        assert "file2.md" in names
        assert "file3.jpg" not in names

    def test_get_info(self, tmp_path):
        """Test getting connector info."""
        connector = LocalFilesConnector(tmp_path)
        info = connector.get_info()

        assert info.name == "Local Files"
        assert info.source_type == SourceType.LOCAL_FILES
        assert info.is_configured is True


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            text="Test content",
            doc_id="test123",
            source_type=SourceType.LOCAL_FILES,
            source_path="/path/to/file.txt",
            source_name="file.txt",
        )

        assert doc.text == "Test content"
        assert doc.doc_id == "test123"
        assert doc.metadata["source_type"] == "local_files"

    def test_document_metadata(self):
        """Test document metadata is populated."""
        doc = Document(
            text="Content",
            doc_id="id1",
            source_name="myfile.pdf",
        )

        assert "doc_id" in doc.metadata
        assert doc.metadata["source_name"] == "myfile.pdf"
