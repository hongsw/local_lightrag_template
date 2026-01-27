"""
OKT-RAG LLM Layer.

Provides model-agnostic LLM interface for pluggable model support.
"""

from .providers.base import LLMProvider, CompletionResult, Message
from .providers.openai import OpenAILLMProvider

__all__ = [
    "LLMProvider",
    "CompletionResult",
    "Message",
    "OpenAILLMProvider",
]
