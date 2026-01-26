"""
FastAPI application for the RAG API server.
"""

from .main import app, get_rag_engine

__all__ = ["app", "get_rag_engine"]
