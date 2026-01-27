"""
Embedding providers for OKT-RAG.

Each provider implements the EmbeddingProvider protocol for consistent interface.
"""

from .base import EmbeddingProvider
from .openai import OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
]
