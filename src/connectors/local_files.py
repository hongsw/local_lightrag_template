"""
Local files connector using LlamaIndex file readers.
"""

import hashlib
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from .base import BaseConnector, ConnectorInfo, Document, SourceType


class LocalFilesConnector(BaseConnector):
    """
    Connector for loading local files (PDF, TXT, MD, etc.).

    Uses PyMuPDF for PDF extraction and plain text reading for other formats.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def __init__(self, base_path: str | Path):
        """
        Initialize the local files connector.

        Args:
            base_path: Root directory to scan for files.
        """
        self.base_path = Path(base_path)

    @property
    def source_type(self) -> SourceType:
        return SourceType.LOCAL_FILES

    def load_documents(self) -> list[Document]:
        """
        Load all supported documents from the base path.

        Returns:
            List of Document objects.
        """
        documents = []

        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        # Find all supported files
        for file_path in self._find_files():
            try:
                doc = self._load_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        return documents

    def load_single_file(self, file_path: str | Path) -> Optional[Document]:
        """
        Load a single file.

        Args:
            file_path: Path to the file.

        Returns:
            Document object or None if failed.
        """
        return self._load_file(Path(file_path))

    def _find_files(self) -> list[Path]:
        """Find all supported files in the base path."""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.base_path.rglob(f"*{ext}"))
        return sorted(files)

    def _load_file(self, file_path: Path) -> Optional[Document]:
        """Load a single file and return a Document."""
        if not file_path.exists():
            return None

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text = self._extract_pdf_text(file_path)
        elif suffix in {".txt", ".md", ".markdown"}:
            text = self._read_text_file(file_path)
        else:
            return None

        if not text or not text.strip():
            return None

        # Generate document ID from file path
        doc_id = self._generate_doc_id(file_path)

        return Document(
            text=text,
            doc_id=doc_id,
            source_type=SourceType.LOCAL_FILES,
            source_path=str(file_path),
            source_name=file_path.name,
            metadata={
                "file_name": file_path.name,
                "file_extension": suffix,
                "file_size": file_path.stat().st_size,
            }
        )

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        text_parts = []

        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        return "\n\n".join(text_parts)

    def _read_text_file(self, file_path: Path) -> str:
        """Read a plain text file."""
        encodings = ["utf-8", "cp949", "euc-kr", "latin-1"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        # Fallback: read with errors ignored
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a unique document ID."""
        # Use relative path from base for ID
        try:
            rel_path = file_path.relative_to(self.base_path)
        except ValueError:
            rel_path = file_path

        hash_input = str(rel_path).encode()
        return hashlib.md5(hash_input).hexdigest()[:12]

    def get_info(self) -> ConnectorInfo:
        """Get connector information."""
        return ConnectorInfo(
            name="Local Files",
            source_type=SourceType.LOCAL_FILES,
            description=f"Load documents from local filesystem: {self.base_path}",
            is_configured=self.base_path.exists(),
            config_requirements=["base_path (directory path)"],
        )

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate that the base path exists."""
        if not self.base_path.exists():
            return False, f"Path does not exist: {self.base_path}"
        if not self.base_path.is_dir():
            return False, f"Path is not a directory: {self.base_path}"
        return True, None

    def list_files(self) -> list[dict]:
        """List all supported files in the base path."""
        files = []
        for file_path in self._find_files():
            files.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "extension": file_path.suffix.lower(),
            })
        return files
