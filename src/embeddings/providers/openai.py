"""
OpenAI embedding provider with Matryoshka support.

Supports text-embedding-3-small and text-embedding-3-large models
with configurable dimensions via Matryoshka embeddings.
"""

import os
from typing import Optional

import openai

from .base import BaseEmbeddingProvider, EmbeddingResult


# Pricing per 1M tokens (as of 2024)
OPENAI_EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
    "text-embedding-ada-002": 0.10,  # $0.10 per 1M tokens (legacy)
}

# Default dimensions for each model
OPENAI_DEFAULT_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Matryoshka supported dimensions
MATRYOSHKA_DIMENSIONS = {
    "text-embedding-3-small": [256, 512, 1024, 1536],
    "text-embedding-3-large": [256, 512, 1024, 2048, 3072],
}


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider with Matryoshka dimension support.

    Supports configurable embedding dimensions for text-embedding-3-* models,
    allowing tradeoffs between quality and storage/speed.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        max_batch_size: int = 100,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            model: OpenAI embedding model name.
            dimension: Embedding dimension (uses Matryoshka if different from default).
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).
            max_batch_size: Maximum texts per API call.
        """
        # Validate model
        if model not in OPENAI_DEFAULT_DIMENSIONS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported: {list(OPENAI_DEFAULT_DIMENSIONS.keys())}"
            )

        # Set dimension (use default if not specified)
        default_dim = OPENAI_DEFAULT_DIMENSIONS[model]
        final_dimension = dimension or default_dim

        # Validate dimension for Matryoshka models
        if model in MATRYOSHKA_DIMENSIONS:
            valid_dims = MATRYOSHKA_DIMENSIONS[model]
            if final_dimension not in valid_dims:
                raise ValueError(
                    f"Invalid dimension {final_dimension} for {model}. "
                    f"Valid dimensions: {valid_dims}"
                )

        super().__init__(
            model=model,
            dimension=final_dimension,
            max_batch_size=max_batch_size,
        )

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

        self._client = openai.AsyncOpenAI(api_key=self._api_key)
        self._uses_matryoshka = (
            model in MATRYOSHKA_DIMENSIONS
            and final_dimension != OPENAI_DEFAULT_DIMENSIONS[model]
        )

    @property
    def uses_matryoshka(self) -> bool:
        """Return whether Matryoshka dimension reduction is being used."""
        return self._uses_matryoshka

    @property
    def cost_per_million_tokens(self) -> float:
        """Return cost per 1M tokens for this model."""
        return OPENAI_EMBEDDING_PRICING.get(self._model, 0.0)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for texts using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            EmbeddingResult with embeddings and cost information.
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self._model,
                dimension=self._dimension,
                tokens_used=0,
                cost_usd=0.0,
            )

        # Build request parameters
        request_params = {
            "model": self._model,
            "input": texts,
        }

        # Add dimensions parameter for Matryoshka
        if self._uses_matryoshka:
            request_params["dimensions"] = self._dimension

        response = await self._client.embeddings.create(**request_params)

        # Extract embeddings (already sorted by index)
        embeddings = [item.embedding for item in response.data]

        # Calculate cost
        tokens_used = response.usage.total_tokens
        cost_usd = (tokens_used / 1_000_000) * self.cost_per_million_tokens

        return EmbeddingResult(
            embeddings=embeddings,
            model=self._model,
            dimension=self._dimension,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )

    def __repr__(self) -> str:
        matryoshka_info = " (Matryoshka)" if self._uses_matryoshka else ""
        return (
            f"OpenAIEmbeddingProvider(model={self._model}, "
            f"dimension={self._dimension}{matryoshka_info})"
        )
