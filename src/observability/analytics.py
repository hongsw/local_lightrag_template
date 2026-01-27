"""
Analytics module for OKT-RAG.

Provides aggregated analysis and insights from retrieval logs.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any
from pathlib import Path

from .retrieval_logger import RetrievalLogger, RetrievalLog


@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""

    timestamp: str
    value: float
    count: int


@dataclass
class SlotPerformance:
    """Performance metrics for an embedding slot."""

    slot_name: str
    query_count: int
    avg_time_ms: float
    avg_cost_usd: float
    total_tokens: int


@dataclass
class StrategyPerformance:
    """Performance metrics for a search strategy."""

    strategy: str
    query_count: int
    avg_time_ms: float
    avg_candidates: float
    avg_accuracy: Optional[float]


class RetrievalAnalytics:
    """
    Analytics engine for retrieval performance analysis.

    Provides insights for:
    - Slot performance comparison
    - Search strategy effectiveness
    - Cost optimization
    - Citation accuracy trends
    """

    def __init__(self, logger: RetrievalLogger):
        """
        Initialize analytics with a retrieval logger.

        Args:
            logger: RetrievalLogger instance to analyze.
        """
        self.logger = logger

    def get_slot_performance(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[SlotPerformance]:
        """
        Get performance metrics by embedding slot.

        Returns:
            List of SlotPerformance for each slot.
        """
        logs = self.logger.get_logs(start_date, end_date, limit=10000)

        # Aggregate by slot
        slot_data: dict[str, dict] = {}

        for log in logs:
            if not log.embedding:
                continue

            slot = log.embedding.slot_name
            if slot not in slot_data:
                slot_data[slot] = {
                    "times": [],
                    "costs": [],
                    "tokens": 0,
                }

            slot_data[slot]["times"].append(log.embedding.time_ms)
            slot_data[slot]["costs"].append(log.embedding.cost_usd)
            slot_data[slot]["tokens"] += log.embedding.tokens_used

        # Build performance objects
        results = []
        for slot, data in slot_data.items():
            results.append(
                SlotPerformance(
                    slot_name=slot,
                    query_count=len(data["times"]),
                    avg_time_ms=sum(data["times"]) / len(data["times"]),
                    avg_cost_usd=sum(data["costs"]) / len(data["costs"]),
                    total_tokens=data["tokens"],
                )
            )

        return sorted(results, key=lambda x: x.query_count, reverse=True)

    def get_strategy_performance(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[StrategyPerformance]:
        """
        Get performance metrics by search strategy.

        Returns:
            List of StrategyPerformance for each strategy.
        """
        logs = self.logger.get_logs(start_date, end_date, limit=10000)

        # Aggregate by strategy
        strategy_data: dict[str, dict] = {}

        for log in logs:
            if not log.search:
                continue

            strategy = log.search.strategy
            if strategy not in strategy_data:
                strategy_data[strategy] = {
                    "times": [],
                    "candidates": [],
                    "accuracies": [],
                }

            strategy_data[strategy]["times"].append(log.search.time_ms)
            strategy_data[strategy]["candidates"].append(log.search.candidates_retrieved)

            if log.citation_accuracy is not None:
                strategy_data[strategy]["accuracies"].append(log.citation_accuracy)

        # Build performance objects
        results = []
        for strategy, data in strategy_data.items():
            avg_accuracy = None
            if data["accuracies"]:
                avg_accuracy = sum(data["accuracies"]) / len(data["accuracies"])

            results.append(
                StrategyPerformance(
                    strategy=strategy,
                    query_count=len(data["times"]),
                    avg_time_ms=sum(data["times"]) / len(data["times"]),
                    avg_candidates=sum(data["candidates"]) / len(data["candidates"]),
                    avg_accuracy=avg_accuracy,
                )
            )

        return sorted(results, key=lambda x: x.query_count, reverse=True)

    def get_accuracy_trend(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = "day",
    ) -> list[TimeSeriesPoint]:
        """
        Get citation accuracy trend over time.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            granularity: "day" or "hour".

        Returns:
            Time series of accuracy values.
        """
        logs = self.logger.get_logs(start_date, end_date, limit=10000)

        # Group by time bucket
        buckets: dict[str, list[float]] = {}

        for log in logs:
            if log.citation_accuracy is None:
                continue

            # Determine bucket key
            if granularity == "hour":
                bucket = log.timestamp[:13]  # YYYY-MM-DDTHH
            else:
                bucket = log.timestamp[:10]  # YYYY-MM-DD

            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(log.citation_accuracy)

        # Build time series
        results = []
        for timestamp, accuracies in sorted(buckets.items()):
            results.append(
                TimeSeriesPoint(
                    timestamp=timestamp,
                    value=sum(accuracies) / len(accuracies),
                    count=len(accuracies),
                )
            )

        return results

    def get_cost_analysis(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get cost analysis breakdown.

        Returns:
            Cost analysis by component.
        """
        logs = self.logger.get_logs(start_date, end_date, limit=10000)

        embedding_cost = 0.0
        llm_cost = 0.0
        total_queries = len(logs)

        for log in logs:
            if log.embedding:
                embedding_cost += log.embedding.cost_usd
            if log.response:
                llm_cost += log.response.cost_usd

        total_cost = embedding_cost + llm_cost

        return {
            "total_queries": total_queries,
            "total_cost_usd": total_cost,
            "embedding_cost_usd": embedding_cost,
            "llm_cost_usd": llm_cost,
            "avg_cost_per_query_usd": total_cost / total_queries if total_queries else 0,
            "cost_breakdown": {
                "embedding_pct": (
                    embedding_cost / total_cost * 100 if total_cost else 0
                ),
                "llm_pct": llm_cost / total_cost * 100 if total_cost else 0,
            },
        }

    def get_query_type_distribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, int]:
        """
        Get distribution of query types.

        Returns:
            Dict mapping query type to count.
        """
        logs = self.logger.get_logs(start_date, end_date, limit=10000)

        distribution: dict[str, int] = {}
        for log in logs:
            qtype = log.query_type or "unknown"
            distribution[qtype] = distribution.get(qtype, 0) + 1

        return distribution

    def get_experiment_comparison(
        self,
        experiment_key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare metrics across experiment variants.

        Args:
            experiment_key: Metadata key for experiment variant.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dict mapping variant to metrics.
        """
        logs = self.logger.get_logs(start_date, end_date, limit=10000)

        variants: dict[str, list[RetrievalLog]] = {}

        for log in logs:
            variant = log.metadata.get(experiment_key, "control")
            if variant not in variants:
                variants[variant] = []
            variants[variant].append(log)

        results = {}
        for variant, variant_logs in variants.items():
            times = [log.total_time_ms for log in variant_logs]
            costs = [log.total_cost_usd for log in variant_logs]
            accuracies = [
                log.citation_accuracy
                for log in variant_logs
                if log.citation_accuracy is not None
            ]

            results[variant] = {
                "count": len(variant_logs),
                "avg_time_ms": sum(times) / len(times) if times else 0,
                "avg_cost_usd": sum(costs) / len(costs) if costs else 0,
                "avg_accuracy": (
                    sum(accuracies) / len(accuracies) if accuracies else None
                ),
            }

        return results
