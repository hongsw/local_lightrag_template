"""
LightRAG engine wrapper for knowledge graph + vector RAG.
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from .connectors.base import Document
from .index_tracker import IndexTracker


class QueryMode(str, Enum):
    """LightRAG query modes."""
    NAIVE = "naive"       # Simple vector similarity
    LOCAL = "local"       # Local context from knowledge graph
    GLOBAL = "global"     # Global patterns from knowledge graph
    HYBRID = "hybrid"     # Combines local and global


@dataclass
class QueryResult:
    """Result from a RAG query."""
    answer: str
    mode: QueryMode
    sources: list[dict]
    metadata: dict


class RAGEngine:
    """
    LightRAG engine wrapper providing knowledge graph + vector RAG capabilities.

    LightRAG combines:
    - Vector store for semantic similarity
    - Knowledge graph for entity relationships
    - Entity store for structured information
    """

    def __init__(
        self,
        working_dir: str | Path,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the RAG engine.

        Args:
            working_dir: Directory for LightRAG data storage.
            openai_api_key: OpenAI API key (uses env var if not provided).
            model_name: LLM model for generation.
            embedding_model: Model for text embeddings.
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Set API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self.model_name = model_name
        self.embedding_model = embedding_model
        self._rag = None
        self._initialized = False
        self._indexed_count = 0
        self._tracker = IndexTracker(self.working_dir)

    @property
    def tracker(self) -> IndexTracker:
        """Get the index tracker."""
        return self._tracker

    def _get_rag(self):
        """Lazy load the LightRAG instance."""
        if self._rag is None:
            try:
                from lightrag import LightRAG
                from lightrag.base import EmbeddingFunc
                from lightrag.llm.openai import openai_complete_if_cache, openai_embed
                from functools import partial
                import numpy as np

                async def llm_func(prompt, **kwargs):
                    return await openai_complete_if_cache(
                        self.model_name,
                        prompt,
                        **kwargs
                    )

                # Embedding dimension for text-embedding-3-small is 1536
                embedding_dim = 1536
                if "large" in self.embedding_model:
                    embedding_dim = 3072

                # Create proper EmbeddingFunc wrapper
                embedding_func = EmbeddingFunc(
                    embedding_dim=embedding_dim,
                    func=partial(openai_embed.func, model=self.embedding_model),
                    model_name=self.embedding_model,
                )

                self._rag = LightRAG(
                    working_dir=str(self.working_dir),
                    llm_model_func=llm_func,
                    llm_model_name=self.model_name,
                    embedding_func=embedding_func,
                )
            except ImportError:
                raise ImportError(
                    "LightRAG not installed. Run: pip install lightrag-hku"
                )
        return self._rag

    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized."""
        rag = self._get_rag()
        if not self._initialized:
            await rag.initialize_storages()
            self._initialized = True
        return rag

    async def index_documents(
        self,
        documents: list[Document],
        batch_size: int = 10,
        skip_tracked: bool = True,
    ) -> dict:
        """
        Index documents into LightRAG.

        Args:
            documents: List of documents to index.
            batch_size: Number of documents to process in each batch.
            skip_tracked: Skip files that are already indexed (default: True).

        Returns:
            Indexing statistics.
        """
        rag = await self._ensure_initialized()
        indexed = 0
        skipped = 0
        errors = []
        indexed_files = set()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            for doc in batch:
                try:
                    # Check if file is already indexed
                    if skip_tracked and doc.source_path:
                        file_path = Path(doc.source_path)
                        if self._tracker.is_indexed(file_path):
                            skipped += 1
                            continue

                    # Prepare text with metadata context
                    text_with_context = self._prepare_text_for_indexing(doc)
                    await rag.ainsert(text_with_context)
                    indexed += 1

                    # Track the indexed file
                    if doc.source_path:
                        indexed_files.add(doc.source_path)

                except Exception as e:
                    errors.append({
                        "doc_id": doc.doc_id,
                        "error": str(e),
                    })

        # Update tracker for newly indexed files
        for file_path in indexed_files:
            self._tracker.mark_indexed(Path(file_path))

        self._indexed_count += indexed

        return {
            "indexed": indexed,
            "skipped": skipped,
            "errors": len(errors),
            "error_details": errors[:10],  # Limit error details
            "total_indexed": self._indexed_count,
        }

    def index_documents_sync(
        self,
        documents: list[Document],
        batch_size: int = 10,
    ) -> dict:
        """Synchronous version of index_documents."""
        return asyncio.run(self.index_documents(documents, batch_size))

    async def query(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
    ) -> QueryResult:
        """
        Query the RAG system.

        Args:
            question: The question to ask.
            mode: Query mode (naive, local, global, hybrid).

        Returns:
            QueryResult with answer and metadata.
        """
        rag = await self._ensure_initialized()

        try:
            from lightrag import QueryParam

            # First, get the context to extract sources
            context_result = await rag.aquery(
                question,
                param=QueryParam(mode=mode.value, only_need_context=True)
            )

            # Extract sources from context
            sources = self._extract_sources_from_context(context_result)

            # Then get the actual answer
            result = await rag.aquery(
                question,
                param=QueryParam(mode=mode.value)
            )

            return QueryResult(
                answer=result,
                mode=mode,
                sources=sources,
                metadata={
                    "query_mode": mode.value,
                    "working_dir": str(self.working_dir),
                    "context_length": len(context_result) if context_result else 0,
                },
            )
        except Exception as e:
            return QueryResult(
                answer=f"Error processing query: {str(e)}",
                mode=mode,
                sources=[],
                metadata={"error": str(e)},
            )

    def _extract_sources_from_context(self, context: str) -> list[dict]:
        """
        Extract source information from LightRAG context.

        Parses the context to find file names, page numbers, and relevant excerpts.
        """
        if not context:
            return []

        sources = []
        seen_sources = set()

        # Split context into chunks (LightRAG separates with various delimiters)
        chunks = re.split(r'\n(?=\[Source:|-----)', context)

        for chunk in chunks:
            if not chunk.strip():
                continue

            source_info = {
                "file_name": None,
                "page": None,
                "excerpt": None,
            }

            # Extract file name from [Source: filename.pdf]
            source_match = re.search(r'\[Source:\s*([^\]]+)\]', chunk)
            if source_match:
                source_info["file_name"] = source_match.group(1).strip()

            # Extract page number from [Page X] or [Page X]
            page_match = re.search(r'\[Page\s*(\d+)\]', chunk)
            if page_match:
                source_info["page"] = int(page_match.group(1))

            # Extract excerpt - clean text without metadata tags
            excerpt = chunk
            # Remove metadata tags
            excerpt = re.sub(r'\[Source:[^\]]+\]', '', excerpt)
            excerpt = re.sub(r'\[Type:[^\]]+\]', '', excerpt)
            excerpt = re.sub(r'\[Page\s*\d+\]', '', excerpt)
            excerpt = re.sub(r'-----+', '', excerpt)
            excerpt = excerpt.strip()

            # Truncate excerpt if too long
            if len(excerpt) > 300:
                excerpt = excerpt[:300] + "..."

            source_info["excerpt"] = excerpt if excerpt else None

            # Only add if we have meaningful content
            if source_info["file_name"] and source_info["excerpt"]:
                # Create unique key to avoid duplicates
                source_key = f"{source_info['file_name']}:{source_info.get('page', 'N/A')}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources.append(source_info)

        # Limit to top 5 sources
        return sources[:5]

    def query_sync(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
    ) -> QueryResult:
        """Synchronous version of query."""
        return asyncio.run(self.query(question, mode))

    def _prepare_text_for_indexing(self, doc: Document) -> str:
        """
        Prepare document text for indexing with metadata context.

        Adds source information to help with retrieval context.
        """
        parts = []

        # Add source context
        if doc.source_name:
            parts.append(f"[Source: {doc.source_name}]")
        if doc.source_type:
            parts.append(f"[Type: {doc.source_type.value}]")

        # Add main content
        parts.append(doc.text)

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get engine statistics."""
        tracker_stats = self._tracker.get_stats()
        return {
            "working_dir": str(self.working_dir),
            "indexed_documents": self._indexed_count,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "is_initialized": self._rag is not None,
            "tracked_files": tracker_stats["total_indexed_files"],
            "tracked_size_bytes": tracker_stats["total_size_bytes"],
        }

    def clear_index(self) -> bool:
        """
        Clear all indexed data.

        Warning: This deletes all LightRAG data in the working directory.
        """
        import shutil

        try:
            if self.working_dir.exists():
                shutil.rmtree(self.working_dir)
                self.working_dir.mkdir(parents=True, exist_ok=True)
            self._rag = None
            self._indexed_count = 0
            # Reinitialize tracker (creates empty tracker file)
            self._tracker = IndexTracker(self.working_dir)
            return True
        except Exception:
            return False

    def get_indexed_files(self) -> list[dict]:
        """Get list of all indexed files."""
        files = self._tracker.get_indexed_files()
        return [
            {
                "file_name": f.file_name,
                "file_path": f.file_path,
                "file_size": f.file_size,
                "indexed_at": f.indexed_at,
                "doc_count": f.doc_count,
            }
            for f in files
        ]

    def is_file_indexed(self, file_path: str | Path) -> bool:
        """Check if a specific file is already indexed."""
        return self._tracker.is_indexed(Path(file_path))
