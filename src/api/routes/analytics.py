"""
Analytics endpoints for OKT-RAG.

Provides API for retrieval analytics and observability data.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...config import get_settings
from ...observability import RetrievalLogger, RetrievalAnalytics


router = APIRouter(prefix="/analytics", tags=["Analytics"])


# Global logger instance (will be set by main.py)
_retrieval_logger: Optional[RetrievalLogger] = None


def get_retrieval_logger() -> RetrievalLogger:
    """Get the global retrieval logger instance."""
    if _retrieval_logger is None:
        settings = get_settings()
        return RetrievalLogger(
            storage_path=settings.observability.storage_path,
            buffer_size=settings.observability.buffer_size,
            enabled=settings.observability.enabled,
        )
    return _retrieval_logger


class RetrievalStatsResponse(BaseModel):
    """Response model for retrieval statistics."""

    total_queries: int
    avg_total_time_ms: float
    avg_total_cost_usd: float
    avg_embedding_time_ms: float
    avg_search_time_ms: float
    avg_response_time_ms: float
    slot_usage: dict[str, int]
    search_strategy_distribution: dict[str, int]
    avg_citation_accuracy: Optional[float]
    date_range: dict[str, Optional[str]]


class SlotPerformanceResponse(BaseModel):
    """Response model for slot performance."""

    slot_name: str
    query_count: int
    avg_time_ms: float
    avg_cost_usd: float
    total_tokens: int


class StrategyPerformanceResponse(BaseModel):
    """Response model for strategy performance."""

    strategy: str
    query_count: int
    avg_time_ms: float
    avg_candidates: float
    avg_accuracy: Optional[float]


class CostAnalysisResponse(BaseModel):
    """Response model for cost analysis."""

    total_queries: int
    total_cost_usd: float
    embedding_cost_usd: float
    llm_cost_usd: float
    avg_cost_per_query_usd: float
    cost_breakdown: dict[str, float]


@router.get("/retrieval", response_model=RetrievalStatsResponse)
async def get_retrieval_analytics(
    start_date: Optional[str] = Query(
        None,
        description="Start date (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    end_date: Optional[str] = Query(
        None,
        description="End date (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    days: Optional[int] = Query(
        7,
        description="Number of days to analyze (if start_date not provided)",
        ge=1,
        le=365,
    ),
    logger: RetrievalLogger = Depends(get_retrieval_logger),
):
    """
    Get retrieval analytics and statistics.

    Provides aggregated metrics including:
    - Query counts and timing
    - Embedding slot usage distribution
    - Search strategy distribution
    - Citation accuracy trends
    - Cost analysis
    """
    # Calculate date range if not provided
    if not start_date:
        start = datetime.utcnow() - timedelta(days=days)
        start_date = start.strftime("%Y-%m-%d")

    if not end_date:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    stats = logger.get_stats(start_date, end_date)

    return RetrievalStatsResponse(**stats)


@router.get("/slots/performance", response_model=list[SlotPerformanceResponse])
async def get_slot_performance(
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    logger: RetrievalLogger = Depends(get_retrieval_logger),
):
    """
    Get performance metrics by embedding slot.

    Compares query counts, timing, and costs across different
    embedding slots to help optimize slot configuration.
    """
    analytics = RetrievalAnalytics(logger)
    performances = analytics.get_slot_performance(start_date, end_date)

    return [
        SlotPerformanceResponse(
            slot_name=p.slot_name,
            query_count=p.query_count,
            avg_time_ms=p.avg_time_ms,
            avg_cost_usd=p.avg_cost_usd,
            total_tokens=p.total_tokens,
        )
        for p in performances
    ]


@router.get("/strategies/performance", response_model=list[StrategyPerformanceResponse])
async def get_strategy_performance(
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    logger: RetrievalLogger = Depends(get_retrieval_logger),
):
    """
    Get performance metrics by search strategy.

    Compares dense, hybrid, and adaptive strategies to
    identify optimal approaches for different query types.
    """
    analytics = RetrievalAnalytics(logger)
    performances = analytics.get_strategy_performance(start_date, end_date)

    return [
        StrategyPerformanceResponse(
            strategy=p.strategy,
            query_count=p.query_count,
            avg_time_ms=p.avg_time_ms,
            avg_candidates=p.avg_candidates,
            avg_accuracy=p.avg_accuracy,
        )
        for p in performances
    ]


@router.get("/accuracy/trend")
async def get_accuracy_trend(
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    granularity: str = Query("day", pattern=r"^(day|hour)$"),
    logger: RetrievalLogger = Depends(get_retrieval_logger),
):
    """
    Get citation accuracy trend over time.

    Tracks how citation accuracy changes over time to
    identify improvements or regressions.
    """
    analytics = RetrievalAnalytics(logger)
    trend = analytics.get_accuracy_trend(start_date, end_date, granularity)

    return {
        "granularity": granularity,
        "data": [
            {
                "timestamp": point.timestamp,
                "accuracy": point.value,
                "query_count": point.count,
            }
            for point in trend
        ],
    }


@router.get("/cost", response_model=CostAnalysisResponse)
async def get_cost_analysis(
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    logger: RetrievalLogger = Depends(get_retrieval_logger),
):
    """
    Get cost analysis breakdown.

    Analyzes costs by component (embedding vs LLM) to
    identify optimization opportunities.
    """
    analytics = RetrievalAnalytics(logger)
    cost_data = analytics.get_cost_analysis(start_date, end_date)

    return CostAnalysisResponse(**cost_data)


@router.get("/query-types")
async def get_query_type_distribution(
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    logger: RetrievalLogger = Depends(get_retrieval_logger),
):
    """
    Get distribution of query types.

    Shows how queries are classified (factual, analytical, etc.)
    to understand usage patterns.
    """
    analytics = RetrievalAnalytics(logger)
    distribution = analytics.get_query_type_distribution(start_date, end_date)

    return {
        "distribution": distribution,
        "total": sum(distribution.values()),
    }


@router.get("/experiments/{experiment_key}")
async def get_experiment_results(
    experiment_key: str,
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    logger: RetrievalLogger = Depends(get_retrieval_logger),
):
    """
    Get A/B experiment results.

    Compares metrics across experiment variants to determine
    which configuration performs better.

    Args:
        experiment_key: Metadata key used to identify experiment variant.
    """
    analytics = RetrievalAnalytics(logger)
    results = analytics.get_experiment_comparison(experiment_key, start_date, end_date)

    return {
        "experiment_key": experiment_key,
        "variants": results,
    }
