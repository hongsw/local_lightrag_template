"""
OKT-RAG Multi-Embedding System.

Provides model-agnostic embedding infrastructure with support for:
- Multiple embedding providers (OpenAI, Voyage, Ollama, Korean-specific)
- Matryoshka embeddings for dimension flexibility
- Multi-slot storage for hybrid retrieval strategies
"""

from .providers.base import EmbeddingProvider
from .providers.openai import OpenAIEmbeddingProvider
from .multi_store import MultiEmbeddingStore, EmbeddingSlot

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MultiEmbeddingStore",
    "EmbeddingSlot",
]
