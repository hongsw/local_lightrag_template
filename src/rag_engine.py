"""
LightRAG engine wrapper for knowledge graph + vector RAG.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from .connectors.base import Document


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
    ) -> dict:
        """
        Index documents into LightRAG.

        Args:
            documents: List of documents to index.
            batch_size: Number of documents to process in each batch.

        Returns:
            Indexing statistics.
        """
        rag = await self._ensure_initialized()
        indexed = 0
        errors = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            for doc in batch:
                try:
                    # Prepare text with metadata context
                    text_with_context = self._prepare_text_for_indexing(doc)
                    await rag.ainsert(text_with_context)
                    indexed += 1
                except Exception as e:
                    errors.append({
                        "doc_id": doc.doc_id,
                        "error": str(e),
                    })

        self._indexed_count += indexed

        return {
            "indexed": indexed,
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

            result = await rag.aquery(
                question,
                param=QueryParam(mode=mode.value)
            )

            return QueryResult(
                answer=result,
                mode=mode,
                sources=[],  # LightRAG doesn't return sources directly
                metadata={
                    "query_mode": mode.value,
                    "working_dir": str(self.working_dir),
                },
            )
        except Exception as e:
            return QueryResult(
                answer=f"Error processing query: {str(e)}",
                mode=mode,
                sources=[],
                metadata={"error": str(e)},
            )

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
        return {
            "working_dir": str(self.working_dir),
            "indexed_documents": self._indexed_count,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "is_initialized": self._rag is not None,
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
            return True
        except Exception:
            return False
