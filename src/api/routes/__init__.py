"""
API route modules for OKT-RAG.
"""

from .query import router as query_router
from .index import router as index_router
from .sources import router as sources_router
from .embedding import router as embedding_router
from .analytics import router as analytics_router

__all__ = [
    "query_router",
    "index_router",
    "sources_router",
    "embedding_router",
    "analytics_router",
]
