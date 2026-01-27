"""
Adaptive Retrieval for OKT-RAG.

Automatically selects optimal retrieval strategy based on query type.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from ..embeddings import MultiEmbeddingStore
from ..observability import RetrievalLogger


class QueryType(str, Enum):
    """Types of queries for strategy selection."""

    FACTUAL = "factual"  # Simple fact lookup
    ANALYTICAL = "analytical"  # Analysis requiring reasoning
    COMPARATIVE = "comparative"  # Comparison between entities
    PROCEDURAL = "procedural"  # How-to questions
    KOREAN_SPECIFIC = "korean_specific"  # Korean language/culture specific
    UNKNOWN = "unknown"


@dataclass
class RetrievalStrategy:
    """Configuration for a retrieval strategy."""

    name: str
    search_type: str  # "dense", "sparse", "hybrid"
    embedding_slots: list[str]
    use_reranker: bool = False
    top_k: int = 10
    score_threshold: float = 0.5
    description: str = ""


# Default strategy matrix
STRATEGY_MATRIX = {
    QueryType.FACTUAL: RetrievalStrategy(
        name="factual_dense",
        search_type="dense",
        embedding_slots=["semantic"],
        use_reranker=False,
        top_k=5,
        score_threshold=0.6,
        description="Fast dense search for fact lookup",
    ),
    QueryType.ANALYTICAL: RetrievalStrategy(
        name="analytical_hybrid",
        search_type="hybrid",
        embedding_slots=["semantic", "semantic_fast"],
        use_reranker=True,
        top_k=10,
        score_threshold=0.5,
        description="Hybrid search with reranking for analysis",
    ),
    QueryType.COMPARATIVE: RetrievalStrategy(
        name="comparative_multi",
        search_type="hybrid",
        embedding_slots=["semantic", "semantic_fast"],
        use_reranker=True,
        top_k=15,
        score_threshold=0.4,
        description="Multi-slot search for comparisons",
    ),
    QueryType.PROCEDURAL: RetrievalStrategy(
        name="procedural_dense",
        search_type="dense",
        embedding_slots=["semantic"],
        use_reranker=True,
        top_k=8,
        score_threshold=0.55,
        description="Dense search with reranking for how-to",
    ),
    QueryType.KOREAN_SPECIFIC: RetrievalStrategy(
        name="korean_dense",
        search_type="dense",
        embedding_slots=["semantic"],  # Could add "domain_ko" when available
        use_reranker=True,
        top_k=10,
        score_threshold=0.5,
        description="Korean-optimized dense search",
    ),
    QueryType.UNKNOWN: RetrievalStrategy(
        name="default_hybrid",
        search_type="hybrid",
        embedding_slots=["semantic"],
        use_reranker=False,
        top_k=10,
        score_threshold=0.5,
        description="Default hybrid strategy",
    ),
}


@dataclass
class RetrievalResult:
    """Result from adaptive retrieval."""

    doc_ids: list[str]
    scores: list[float]
    query_type: QueryType
    strategy_used: str
    slots_used: list[str]
    metadata: dict = field(default_factory=dict)


class QueryClassifier:
    """
    Classifies queries to determine optimal retrieval strategy.

    Uses keyword patterns and optional LLM classification.
    """

    # Keyword patterns for classification
    FACTUAL_KEYWORDS = [
        "무엇", "언제", "어디", "누가", "몇", "what", "when", "where", "who", "how many",
        "정의", "의미", "뜻", "define", "meaning"
    ]

    ANALYTICAL_KEYWORDS = [
        "왜", "어떻게", "분석", "이유", "원인", "영향", "결과",
        "why", "how", "analyze", "reason", "cause", "effect", "impact"
    ]

    COMPARATIVE_KEYWORDS = [
        "비교", "차이", "다른점", "공통점", "vs", "versus",
        "compare", "difference", "similar", "between"
    ]

    PROCEDURAL_KEYWORDS = [
        "방법", "절차", "단계", "과정", "어떻게 하", "하는 법",
        "how to", "steps", "process", "procedure"
    ]

    KOREAN_INDICATORS = [
        "한국", "한글", "국내", "korea", "korean"
    ]

    def classify(self, query: str) -> QueryType:
        """
        Classify query type based on patterns.

        Args:
            query: Query text.

        Returns:
            Detected QueryType.
        """
        query_lower = query.lower()

        # Check for Korean-specific indicators
        has_korean = any(k in query_lower for k in self.KOREAN_INDICATORS)

        # Check patterns in order of specificity
        if any(k in query_lower for k in self.COMPARATIVE_KEYWORDS):
            return QueryType.COMPARATIVE

        if any(k in query_lower for k in self.PROCEDURAL_KEYWORDS):
            return QueryType.PROCEDURAL

        if any(k in query_lower for k in self.ANALYTICAL_KEYWORDS):
            return QueryType.ANALYTICAL

        if any(k in query_lower for k in self.FACTUAL_KEYWORDS):
            return QueryType.FACTUAL

        # Check Korean proportion in query
        korean_chars = sum(1 for c in query if '\uAC00' <= c <= '\uD7A3')
        if korean_chars / max(len(query), 1) > 0.3 and has_korean:
            return QueryType.KOREAN_SPECIFIC

        return QueryType.UNKNOWN

    def get_confidence(self, query: str, query_type: QueryType) -> float:
        """
        Get confidence score for classification.

        Args:
            query: Query text.
            query_type: Classified type.

        Returns:
            Confidence score (0-1).
        """
        query_lower = query.lower()

        keyword_map = {
            QueryType.FACTUAL: self.FACTUAL_KEYWORDS,
            QueryType.ANALYTICAL: self.ANALYTICAL_KEYWORDS,
            QueryType.COMPARATIVE: self.COMPARATIVE_KEYWORDS,
            QueryType.PROCEDURAL: self.PROCEDURAL_KEYWORDS,
            QueryType.KOREAN_SPECIFIC: self.KOREAN_INDICATORS,
        }

        keywords = keyword_map.get(query_type, [])
        matches = sum(1 for k in keywords if k in query_lower)

        if matches == 0:
            return 0.3  # Low confidence for unknown

        # Higher confidence with more matches
        return min(0.5 + (matches * 0.1), 0.95)


class AdaptiveRetriever:
    """
    Adaptive retrieval system that selects strategies based on query type.

    Integrates with MultiEmbeddingStore and RetrievalLogger for
    comprehensive retrieval operations.
    """

    def __init__(
        self,
        store: MultiEmbeddingStore,
        logger: Optional[RetrievalLogger] = None,
        strategy_matrix: Optional[dict[QueryType, RetrievalStrategy]] = None,
    ):
        """
        Initialize adaptive retriever.

        Args:
            store: MultiEmbeddingStore for vector operations.
            logger: Optional RetrievalLogger for observability.
            strategy_matrix: Custom strategy matrix (uses default if None).
        """
        self.store = store
        self.logger = logger
        self.classifier = QueryClassifier()
        self.strategies = strategy_matrix or STRATEGY_MATRIX

    async def classify_query(self, query: str) -> tuple[QueryType, float]:
        """
        Classify query and return type with confidence.

        Args:
            query: Query text.

        Returns:
            Tuple of (QueryType, confidence).
        """
        query_type = self.classifier.classify(query)
        confidence = self.classifier.get_confidence(query, query_type)
        return query_type, confidence

    def select_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """
        Select retrieval strategy for query type.

        Args:
            query_type: Classified query type.

        Returns:
            Appropriate RetrievalStrategy.
        """
        return self.strategies.get(query_type, self.strategies[QueryType.UNKNOWN])

    async def retrieve(
        self,
        query: str,
        override_strategy: Optional[RetrievalStrategy] = None,
        request_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Perform adaptive retrieval.

        Args:
            query: Query text.
            override_strategy: Optional strategy override.
            request_id: Optional request ID for logging.

        Returns:
            RetrievalResult with documents and metadata.
        """
        # Classify query
        query_type, confidence = await self.classify_query(query)

        # Select strategy
        strategy = override_strategy or self.select_strategy(query_type)

        # Determine available slots
        available_slots = [s.name for s in self.store.get_enabled_slots()]
        slots_to_use = [s for s in strategy.embedding_slots if s in available_slots]

        if not slots_to_use:
            slots_to_use = available_slots[:1]  # Fallback to first available

        # Perform search based on strategy
        if strategy.search_type == "hybrid" and len(slots_to_use) > 1:
            results = await self.store.hybrid_search(
                query=query,
                slot_names=slots_to_use,
                top_k=strategy.top_k,
            )
            doc_ids = [r.doc_id for r in results]
            scores = [r.combined_score for r in results]
        else:
            results = await self.store.search(
                query=query,
                slot_name=slots_to_use[0],
                top_k=strategy.top_k,
            )
            doc_ids = [r.doc_id for r in results]
            scores = [r.score for r in results]

        # Apply score threshold
        filtered_docs = []
        filtered_scores = []
        for doc_id, score in zip(doc_ids, scores):
            if score >= strategy.score_threshold:
                filtered_docs.append(doc_id)
                filtered_scores.append(score)

        # Log if enabled
        if self.logger and request_id:
            self.logger.log_search(
                request_id=request_id,
                strategy=strategy.name,
                slots_used=slots_to_use,
                top_k=strategy.top_k,
                candidates=len(filtered_docs),
                time_ms=0,  # Would need timing context
                vector_scores=scores[:10],
                final_scores=filtered_scores[:10],
            )

        return RetrievalResult(
            doc_ids=filtered_docs,
            scores=filtered_scores,
            query_type=query_type,
            strategy_used=strategy.name,
            slots_used=slots_to_use,
            metadata={
                "classification_confidence": confidence,
                "search_type": strategy.search_type,
                "use_reranker": strategy.use_reranker,
                "score_threshold": strategy.score_threshold,
            },
        )

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about available strategies."""
        return {
            query_type.value: {
                "name": strategy.name,
                "search_type": strategy.search_type,
                "embedding_slots": strategy.embedding_slots,
                "use_reranker": strategy.use_reranker,
                "top_k": strategy.top_k,
                "description": strategy.description,
            }
            for query_type, strategy in self.strategies.items()
        }
