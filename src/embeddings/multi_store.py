"""
Multi-Embedding Store for OKT-RAG.

Manages multiple embedding spaces (slots) for the same documents,
enabling different retrieval strategies per query.
"""

import json
import asyncio
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

import numpy as np

from .providers.base import EmbeddingProvider, EmbeddingResult


@dataclass
class EmbeddingSlot:
    """Configuration for an embedding slot."""

    name: str  # Unique slot identifier (e.g., "semantic", "keyword", "domain_ko")
    provider_type: str  # Provider type (e.g., "openai", "voyage", "ollama")
    model: str  # Model identifier
    dimension: int  # Embedding dimension
    weight: float = 1.0  # Weight for hybrid search
    enabled: bool = True  # Whether slot is active
    description: str = ""  # Human-readable description


@dataclass
class DocumentEmbedding:
    """Embedding for a single document in a specific slot."""

    doc_id: str
    slot_name: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SearchResult:
    """Result from a search operation."""

    doc_id: str
    score: float
    slot_name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class HybridSearchResult:
    """Result from hybrid search across multiple slots."""

    doc_id: str
    combined_score: float
    slot_scores: dict[str, float]  # slot_name -> score
    metadata: dict = field(default_factory=dict)


class MultiEmbeddingStore:
    """
    Store that manages multiple embedding spaces for documents.

    Enables storing the same document in different embedding spaces
    (slots) for flexible retrieval strategies.
    """

    def __init__(
        self,
        working_dir: Path | str,
        slots: list[EmbeddingSlot],
    ):
        """
        Initialize multi-embedding store.

        Args:
            working_dir: Directory for storing embeddings.
            slots: List of embedding slot configurations.
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.slots = {slot.name: slot for slot in slots}
        self._providers: dict[str, EmbeddingProvider] = {}
        self._embeddings: dict[str, dict[str, DocumentEmbedding]] = {
            slot.name: {} for slot in slots
        }

        # Load existing embeddings
        self._load_embeddings()

    def register_provider(self, slot_name: str, provider: EmbeddingProvider) -> None:
        """
        Register an embedding provider for a slot.

        Args:
            slot_name: Name of the slot.
            provider: EmbeddingProvider instance.
        """
        if slot_name not in self.slots:
            raise ValueError(f"Unknown slot: {slot_name}")

        slot = self.slots[slot_name]
        if provider.dimension != slot.dimension:
            raise ValueError(
                f"Provider dimension {provider.dimension} doesn't match "
                f"slot dimension {slot.dimension}"
            )

        self._providers[slot_name] = provider

    def get_slot(self, slot_name: str) -> EmbeddingSlot:
        """Get slot configuration by name."""
        if slot_name not in self.slots:
            raise ValueError(f"Unknown slot: {slot_name}")
        return self.slots[slot_name]

    def list_slots(self) -> list[EmbeddingSlot]:
        """List all configured slots."""
        return list(self.slots.values())

    def get_enabled_slots(self) -> list[EmbeddingSlot]:
        """List all enabled slots."""
        return [slot for slot in self.slots.values() if slot.enabled]

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict] = None,
        slot_names: Optional[list[str]] = None,
    ) -> dict[str, EmbeddingResult]:
        """
        Add document to specified slots (or all enabled slots).

        Args:
            doc_id: Unique document identifier.
            content: Document text content.
            metadata: Optional metadata to store with embeddings.
            slot_names: Specific slots to use (None = all enabled slots).

        Returns:
            Dict mapping slot_name to EmbeddingResult.
        """
        metadata = metadata or {}
        target_slots = slot_names or [s.name for s in self.get_enabled_slots()]

        results = {}

        # Embed in parallel across slots
        async def embed_for_slot(slot_name: str) -> tuple[str, EmbeddingResult]:
            if slot_name not in self._providers:
                raise ValueError(f"No provider registered for slot: {slot_name}")

            provider = self._providers[slot_name]
            result = await provider.embed([content])

            # Store embedding
            doc_embedding = DocumentEmbedding(
                doc_id=doc_id,
                slot_name=slot_name,
                embedding=result.embeddings[0],
                metadata=metadata,
            )
            self._embeddings[slot_name][doc_id] = doc_embedding

            return slot_name, result

        tasks = [embed_for_slot(name) for name in target_slots]
        slot_results = await asyncio.gather(*tasks)

        for slot_name, result in slot_results:
            results[slot_name] = result

        # Persist embeddings
        self._save_embeddings()

        return results

    async def add_documents(
        self,
        documents: list[tuple[str, str, Optional[dict]]],
        slot_names: Optional[list[str]] = None,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """
        Add multiple documents to slots.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
            slot_names: Specific slots to use.
            batch_size: Batch size for embedding operations.

        Returns:
            Dict mapping slot_name to count of documents added.
        """
        target_slots = slot_names or [s.name for s in self.get_enabled_slots()]
        counts = {name: 0 for name in target_slots}

        for slot_name in target_slots:
            if slot_name not in self._providers:
                continue

            provider = self._providers[slot_name]

            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                doc_ids = [d[0] for d in batch]
                contents = [d[1] for d in batch]
                metadatas = [d[2] or {} for d in batch]

                result = await provider.embed(contents)

                for j, embedding in enumerate(result.embeddings):
                    doc_embedding = DocumentEmbedding(
                        doc_id=doc_ids[j],
                        slot_name=slot_name,
                        embedding=embedding,
                        metadata=metadatas[j],
                    )
                    self._embeddings[slot_name][doc_ids[j]] = doc_embedding
                    counts[slot_name] += 1

        self._save_embeddings()
        return counts

    async def search(
        self,
        query: str,
        slot_name: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Search in a specific slot.

        Args:
            query: Query text.
            slot_name: Slot to search in.
            top_k: Number of results to return.

        Returns:
            List of SearchResult sorted by score descending.
        """
        if slot_name not in self._providers:
            raise ValueError(f"No provider registered for slot: {slot_name}")

        provider = self._providers[slot_name]
        query_embedding = await provider.embed_single(query)

        # Compute similarities
        results = []
        for doc_id, doc_emb in self._embeddings[slot_name].items():
            score = self._cosine_similarity(query_embedding, doc_emb.embedding)
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=score,
                    slot_name=slot_name,
                    metadata=doc_emb.metadata,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def hybrid_search(
        self,
        query: str,
        slot_names: Optional[list[str]] = None,
        top_k: int = 10,
        strategy: str = "weighted_sum",
    ) -> list[HybridSearchResult]:
        """
        Search across multiple slots with result fusion.

        Args:
            query: Query text.
            slot_names: Slots to search (None = all enabled).
            top_k: Number of results to return.
            strategy: Fusion strategy ("weighted_sum", "rrf").

        Returns:
            List of HybridSearchResult sorted by combined score.
        """
        target_slots = slot_names or [s.name for s in self.get_enabled_slots()]

        # Search each slot in parallel
        async def search_slot(name: str) -> tuple[str, list[SearchResult]]:
            results = await self.search(query, name, top_k=top_k * 2)
            return name, results

        tasks = [search_slot(name) for name in target_slots if name in self._providers]
        slot_results = await asyncio.gather(*tasks)

        # Collect scores per document
        doc_scores: dict[str, dict[str, float]] = {}
        doc_metadata: dict[str, dict] = {}

        for slot_name, results in slot_results:
            for result in results:
                if result.doc_id not in doc_scores:
                    doc_scores[result.doc_id] = {}
                    doc_metadata[result.doc_id] = result.metadata
                doc_scores[result.doc_id][slot_name] = result.score

        # Fuse scores
        hybrid_results = []
        for doc_id, scores in doc_scores.items():
            if strategy == "weighted_sum":
                combined = self._weighted_sum_fusion(scores)
            elif strategy == "rrf":
                combined = self._rrf_fusion(scores, slot_results)
            else:
                combined = sum(scores.values()) / len(scores)

            hybrid_results.append(
                HybridSearchResult(
                    doc_id=doc_id,
                    combined_score=combined,
                    slot_scores=scores,
                    metadata=doc_metadata[doc_id],
                )
            )

        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        return hybrid_results[:top_k]

    def _weighted_sum_fusion(self, scores: dict[str, float]) -> float:
        """Compute weighted sum of scores."""
        total_weight = 0.0
        weighted_sum = 0.0

        for slot_name, score in scores.items():
            weight = self.slots[slot_name].weight
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _rrf_fusion(
        self,
        scores: dict[str, float],
        slot_results: list[tuple[str, list[SearchResult]]],
        k: int = 60,
    ) -> float:
        """
        Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across all lists.
        """
        rrf_score = 0.0

        for slot_name, results in slot_results:
            for rank, result in enumerate(results, 1):
                if scores.get(slot_name) and result.doc_id in scores:
                    rrf_score += 1.0 / (k + rank)
                    break

        return rrf_score

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_document(
        self,
        doc_id: str,
        slot_name: Optional[str] = None,
    ) -> dict[str, DocumentEmbedding]:
        """
        Get document embeddings.

        Args:
            doc_id: Document ID.
            slot_name: Specific slot (None = all slots).

        Returns:
            Dict mapping slot_name to DocumentEmbedding.
        """
        if slot_name:
            emb = self._embeddings.get(slot_name, {}).get(doc_id)
            return {slot_name: emb} if emb else {}

        return {
            name: embeddings[doc_id]
            for name, embeddings in self._embeddings.items()
            if doc_id in embeddings
        }

    def remove_document(self, doc_id: str) -> int:
        """
        Remove document from all slots.

        Returns:
            Number of slots document was removed from.
        """
        removed = 0
        for slot_name, embeddings in self._embeddings.items():
            if doc_id in embeddings:
                del embeddings[doc_id]
                removed += 1

        if removed > 0:
            self._save_embeddings()

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        slot_stats = {}
        for slot_name, embeddings in self._embeddings.items():
            slot = self.slots[slot_name]
            slot_stats[slot_name] = {
                "document_count": len(embeddings),
                "dimension": slot.dimension,
                "provider_type": slot.provider_type,
                "model": slot.model,
                "weight": slot.weight,
                "enabled": slot.enabled,
                "has_provider": slot_name in self._providers,
            }

        return {
            "working_dir": str(self.working_dir),
            "total_slots": len(self.slots),
            "enabled_slots": len(self.get_enabled_slots()),
            "slots": slot_stats,
        }

    def _get_storage_path(self, slot_name: str) -> Path:
        """Get storage path for a slot."""
        return self.working_dir / f"embeddings_{slot_name}.json"

    def _save_embeddings(self) -> None:
        """Save embeddings to disk."""
        for slot_name, embeddings in self._embeddings.items():
            path = self._get_storage_path(slot_name)
            data = {
                doc_id: asdict(emb) for doc_id, emb in embeddings.items()
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

    def _load_embeddings(self) -> None:
        """Load embeddings from disk."""
        for slot_name in self.slots:
            path = self._get_storage_path(slot_name)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._embeddings[slot_name] = {
                    doc_id: DocumentEmbedding(**emb_data)
                    for doc_id, emb_data in data.items()
                }

    def clear(self, slot_name: Optional[str] = None) -> int:
        """
        Clear embeddings.

        Args:
            slot_name: Specific slot to clear (None = all slots).

        Returns:
            Number of documents cleared.
        """
        cleared = 0

        if slot_name:
            if slot_name in self._embeddings:
                cleared = len(self._embeddings[slot_name])
                self._embeddings[slot_name] = {}
                path = self._get_storage_path(slot_name)
                if path.exists():
                    path.unlink()
        else:
            for name, embeddings in self._embeddings.items():
                cleared += len(embeddings)
                self._embeddings[name] = {}
                path = self._get_storage_path(name)
                if path.exists():
                    path.unlink()

        return cleared
