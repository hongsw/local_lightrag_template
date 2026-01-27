"""
Base embedding provider interface for OKT-RAG.

Defines the protocol that all embedding providers must implement,
enabling model-agnostic embedding operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class EmbeddingResult:
    """Result from an embedding operation."""

    embeddings: list[list[float]]
    model: str
    dimension: int
    tokens_used: int
    cost_usd: float


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    All embedding providers must implement this interface to ensure
    consistent behavior across different models and services.
    """

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size for embedding requests."""
        ...

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbeddingResult with embeddings and metadata.
        """
        ...

    async def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector as list of floats.
        """
        ...


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Provides common functionality and enforces interface implementation.
    """

    def __init__(
        self,
        model: str,
        dimension: int,
        max_batch_size: int = 100,
    ):
        self._model = model
        self._dimension = dimension
        self._max_batch_size = max_batch_size

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        pass

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = await self.embed([text])
        return result.embeddings[0]

    async def embed_batched(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Optional batch size (uses max_batch_size if not specified).

        Returns:
            Combined EmbeddingResult from all batches.
        """
        batch_size = batch_size or self._max_batch_size
        all_embeddings: list[list[float]] = []
        total_tokens = 0
        total_cost = 0.0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = await self.embed(batch)
            all_embeddings.extend(result.embeddings)
            total_tokens += result.tokens_used
            total_cost += result.cost_usd

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self._model,
            dimension=self._dimension,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )
