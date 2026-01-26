"""
Document processor for chunking and metadata enrichment.
"""

from dataclasses import dataclass
from typing import Optional

from .connectors.base import Document


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100


class DocumentProcessor:
    """
    Process documents: chunking, cleaning, and metadata enrichment.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize the document processor.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkConfig()

    def process_documents(self, documents: list[Document]) -> list[Document]:
        """
        Process a list of documents.

        Args:
            documents: List of documents to process.

        Returns:
            List of processed (chunked) documents.
        """
        processed = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            processed.extend(chunks)
        return processed

    def chunk_document(self, document: Document) -> list[Document]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of chunked documents with updated metadata.
        """
        text = self._clean_text(document.text)

        if len(text) <= self.config.chunk_size:
            # Document is small enough, return as-is
            return [document]

        chunks = self._split_text(text)

        chunked_docs = []
        for idx, chunk_text in enumerate(chunks):
            # Create new document for each chunk
            chunk_doc = Document(
                text=chunk_text,
                doc_id=f"{document.doc_id}_chunk_{idx}",
                source_type=document.source_type,
                source_path=document.source_path,
                source_name=document.source_name,
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "parent_doc_id": document.doc_id,
                },
            )
            chunked_docs.append(chunk_doc)

        return chunked_docs

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks with overlap.

        Uses sentence-aware splitting when possible.
        """
        chunks = []
        start = 0
        prev_start = -1

        while start < len(text):
            # Ensure we make progress to avoid infinite loop
            if start == prev_start:
                start += self.config.chunk_size // 2
                if start >= len(text):
                    break

            prev_start = start
            end = start + self.config.chunk_size

            if end >= len(text):
                # Last chunk
                chunk = text[start:]
                if len(chunk) >= self.config.min_chunk_size:
                    chunks.append(chunk)
                elif chunks:
                    # Append to previous chunk if too small
                    chunks[-1] = chunks[-1] + " " + chunk
                else:
                    chunks.append(chunk)
                break

            # Try to find a sentence boundary
            boundary = self._find_boundary(text, end)
            if boundary > start + self.config.min_chunk_size:
                end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            new_start = end - self.config.chunk_overlap
            # Ensure we always make forward progress
            start = max(new_start, start + self.config.min_chunk_size)

        return chunks

    def _find_boundary(self, text: str, position: int) -> int:
        """
        Find a natural boundary (sentence end) near the position.

        Looks for sentence-ending punctuation within a reasonable range.
        """
        search_range = 100  # Look within 100 characters

        # Look backward for sentence boundaries
        search_start = max(0, position - search_range)
        search_end = min(len(text), position + search_range)
        search_text = text[search_start:search_end]

        # Find sentence boundaries
        boundaries = []
        for i, char in enumerate(search_text):
            if char in ".!?" and i + search_start < position:
                # Check if it's a real sentence end (not abbreviation)
                actual_pos = i + search_start
                if actual_pos + 1 < len(text):
                    next_char = text[actual_pos + 1]
                    if next_char in " \n\t":
                        boundaries.append(actual_pos + 1)

        # Return the closest boundary to position
        if boundaries:
            return max(boundaries)

        # Fallback: look for newlines or spaces
        for boundary_char in ["\n", " "]:
            for i in range(position, max(0, position - search_range), -1):
                if text[i] == boundary_char:
                    return i + 1

        return position

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        - Remove excessive whitespace
        - Normalize line breaks
        - Remove control characters
        """
        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Normalize spaces within line
            cleaned = " ".join(line.split())
            cleaned_lines.append(cleaned)

        # Remove excessive blank lines
        result_lines = []
        blank_count = 0
        for line in cleaned_lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        return "\n".join(result_lines).strip()

    def get_text_stats(self, text: str) -> dict:
        """Get statistics about a text."""
        words = text.split()
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": text.count("\n") + 1,
            "estimated_chunks": max(1, len(text) // self.config.chunk_size),
        }
