"""
Index tracker for preventing duplicate indexing.
Tracks indexed files by hash and modification time.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class IndexedFileInfo:
    """Information about an indexed file."""
    file_path: str
    file_name: str
    file_hash: str
    file_size: int
    modified_time: float
    indexed_at: str
    doc_count: int = 1


class IndexTracker:
    """
    Tracks indexed files to prevent duplicate indexing.

    Stores tracking data in a JSON file alongside LightRAG data.
    """

    TRACKER_FILE = "indexed_files.json"

    def __init__(self, working_dir: str | Path):
        """
        Initialize the index tracker.

        Args:
            working_dir: LightRAG working directory.
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_path = self.working_dir / self.TRACKER_FILE
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load tracking data from file."""
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Save tracking data to file."""
        with open(self.tracker_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute MD5 hash of a file.

        Args:
            file_path: Path to the file.
            chunk_size: Size of chunks to read.

        Returns:
            MD5 hash string.
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_file_key(self, file_path: Path) -> str:
        """Generate a unique key for a file based on its absolute path."""
        return str(file_path.resolve())

    def is_indexed(self, file_path: Path) -> bool:
        """
        Check if a file is already indexed.

        Args:
            file_path: Path to the file.

        Returns:
            True if file is indexed and unchanged.
        """
        key = self.get_file_key(file_path)
        if key not in self._data:
            return False

        # Check if file still exists
        if not file_path.exists():
            return False

        # Check if file has changed (by hash)
        current_hash = self.compute_file_hash(file_path)
        return self._data[key].get("file_hash") == current_hash

    def needs_reindex(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Check if a file needs (re)indexing.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (needs_indexing, reason).
        """
        if not file_path.exists():
            return False, "file_not_found"

        key = self.get_file_key(file_path)

        if key not in self._data:
            return True, "new_file"

        current_hash = self.compute_file_hash(file_path)
        if self._data[key].get("file_hash") != current_hash:
            return True, "content_changed"

        return False, None

    def mark_indexed(
        self,
        file_path: Path,
        doc_count: int = 1,
    ) -> IndexedFileInfo:
        """
        Mark a file as indexed.

        Args:
            file_path: Path to the file.
            doc_count: Number of documents/chunks indexed.

        Returns:
            IndexedFileInfo object.
        """
        file_path = Path(file_path)
        key = self.get_file_key(file_path)

        info = IndexedFileInfo(
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_hash=self.compute_file_hash(file_path),
            file_size=file_path.stat().st_size,
            modified_time=file_path.stat().st_mtime,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            doc_count=doc_count,
        )

        self._data[key] = asdict(info)
        self._save()

        return info

    def remove_file(self, file_path: Path) -> bool:
        """
        Remove a file from tracking.

        Args:
            file_path: Path to the file.

        Returns:
            True if file was removed.
        """
        key = self.get_file_key(file_path)
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def get_indexed_files(self) -> list[IndexedFileInfo]:
        """
        Get list of all indexed files.

        Returns:
            List of IndexedFileInfo objects.
        """
        return [
            IndexedFileInfo(**data)
            for data in self._data.values()
        ]

    def get_file_info(self, file_path: Path) -> Optional[IndexedFileInfo]:
        """
        Get info for a specific indexed file.

        Args:
            file_path: Path to the file.

        Returns:
            IndexedFileInfo or None.
        """
        key = self.get_file_key(file_path)
        if key in self._data:
            return IndexedFileInfo(**self._data[key])
        return None

    def filter_new_files(self, file_paths: list[Path]) -> list[Path]:
        """
        Filter list to only files that need indexing.

        Args:
            file_paths: List of file paths.

        Returns:
            List of files that need indexing.
        """
        return [
            fp for fp in file_paths
            if self.needs_reindex(fp)[0]
        ]

    def get_stats(self) -> dict:
        """Get tracking statistics."""
        files = self.get_indexed_files()
        return {
            "total_indexed_files": len(files),
            "total_size_bytes": sum(f.file_size for f in files),
            "total_doc_count": sum(f.doc_count for f in files),
            "tracker_path": str(self.tracker_path),
        }

    def clear(self) -> None:
        """Clear all tracking data."""
        self._data = {}
        self._save()
