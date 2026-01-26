"""
Tests for document processor.
"""

import pytest

from src.processor import DocumentProcessor, ChunkConfig
from src.connectors.base import Document, SourceType


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        processor = DocumentProcessor()
        assert processor.config.chunk_size == 1000
        assert processor.config.chunk_overlap == 200

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ChunkConfig(chunk_size=500, chunk_overlap=100)
        processor = DocumentProcessor(config)
        assert processor.config.chunk_size == 500

    def test_chunk_small_document(self):
        """Test that small documents are not chunked."""
        processor = DocumentProcessor()
        doc = Document(
            text="Small text",
            doc_id="test",
            source_type=SourceType.LOCAL_FILES,
        )

        chunks = processor.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Small text"

    def test_chunk_large_document(self):
        """Test chunking a large document."""
        config = ChunkConfig(chunk_size=100, chunk_overlap=20)
        processor = DocumentProcessor(config)

        long_text = "This is a sentence. " * 50  # About 1000 chars
        doc = Document(
            text=long_text,
            doc_id="test",
            source_type=SourceType.LOCAL_FILES,
        )

        chunks = processor.chunk_document(doc)
        assert len(chunks) > 1

        # Verify chunk metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["parent_doc_id"] == "test"
            assert f"_chunk_{i}" in chunk.doc_id

    def test_process_documents(self):
        """Test processing multiple documents."""
        processor = DocumentProcessor()
        docs = [
            Document(text="Doc 1", doc_id="1", source_type=SourceType.LOCAL_FILES),
            Document(text="Doc 2", doc_id="2", source_type=SourceType.LOCAL_FILES),
        ]

        processed = processor.process_documents(docs)
        assert len(processed) == 2

    def test_clean_text(self):
        """Test text cleaning."""
        processor = DocumentProcessor()

        dirty_text = "  Multiple   spaces  \n\n\n\n\nToo many lines  "
        cleaned = processor._clean_text(dirty_text)

        assert "  " not in cleaned.split("\n")[0]  # No double spaces in first line
        # Allows up to 2 blank lines (3 newlines), but not more
        assert cleaned.count("\n\n\n\n") == 0  # No quadruple newlines

    def test_get_text_stats(self):
        """Test getting text statistics."""
        processor = DocumentProcessor()
        text = "Hello world.\nSecond line."

        stats = processor.get_text_stats(text)

        assert stats["word_count"] == 4
        assert stats["line_count"] == 2
        assert stats["character_count"] == len(text)
