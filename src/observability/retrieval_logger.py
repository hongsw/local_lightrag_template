"""
Retrieval Logger for OKT-RAG.

Captures complete retrieval decision process for research and analysis.
"""

import json
import uuid
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class EmbeddingLog:
    """Log for embedding operation."""

    slot_name: str
    model: str
    dimension: int
    time_ms: float
    tokens_used: int
    cost_usd: float


@dataclass
class SearchLog:
    """Log for search operation."""

    strategy: str  # "dense", "sparse", "hybrid", "adaptive"
    slots_used: list[str]
    top_k_requested: int
    candidates_retrieved: int
    time_ms: float
    vector_scores: list[float] = field(default_factory=list)
    bm25_scores: Optional[list[float]] = None
    reranker_scores: Optional[list[float]] = None
    final_scores: list[float] = field(default_factory=list)


@dataclass
class ResponseLog:
    """Log for LLM response generation."""

    llm_model: str
    context_tokens: int
    response_tokens: int
    time_ms: float
    cost_usd: float


@dataclass
class RetrievalLog:
    """Complete retrieval process log."""

    # Identity
    request_id: str
    timestamp: str

    # Query analysis
    query_text: str
    query_length: int
    query_type: str = "unknown"  # "factual", "analytical", "comparative"
    detected_intent: str = ""

    # Embedding stage
    embedding: Optional[EmbeddingLog] = None

    # Search stage
    search: Optional[SearchLog] = None

    # Selected documents
    selected_doc_ids: list[str] = field(default_factory=list)
    selection_reason: str = ""  # "score_threshold", "diversity", "recency"

    # Response stage
    response: Optional[ResponseLog] = None

    # Verification (if performed)
    citation_count: int = 0
    citation_accuracy: Optional[float] = None
    removed_citations: list[int] = field(default_factory=list)

    # Performance summary
    total_time_ms: float = 0.0
    total_cost_usd: float = 0.0

    # Additional metadata
    metadata: dict = field(default_factory=dict)


class RetrievalLogger:
    """
    Logger for complete retrieval process observability.

    Captures all decision points in the retrieval pipeline for:
    - Research data collection
    - Performance analysis
    - A/B experiment tracking
    - System debugging
    """

    def __init__(
        self,
        storage_path: Path | str,
        buffer_size: int = 100,
        enabled: bool = True,
    ):
        """
        Initialize retrieval logger.

        Args:
            storage_path: Directory for log storage.
            buffer_size: Number of logs to buffer before writing.
            enabled: Whether logging is active.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.enabled = enabled

        self._buffer: list[RetrievalLog] = []
        self._active_logs: dict[str, RetrievalLog] = {}
        self._timers: dict[str, dict[str, float]] = {}

    def log_query_start(
        self,
        query: str,
        query_type: str = "unknown",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Start logging a query.

        Args:
            query: Query text.
            query_type: Type of query.
            metadata: Additional metadata.

        Returns:
            request_id for subsequent logging calls.
        """
        if not self.enabled:
            return ""

        request_id = str(uuid.uuid4())

        log = RetrievalLog(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            query_text=query,
            query_length=len(query),
            query_type=query_type,
            metadata=metadata or {},
        )

        self._active_logs[request_id] = log
        self._timers[request_id] = {"start": time.time()}

        return request_id

    def log_embedding(
        self,
        request_id: str,
        slot_name: str,
        model: str,
        dimension: int,
        time_ms: float,
        tokens_used: int,
        cost_usd: float,
    ) -> None:
        """Log embedding operation."""
        if not self.enabled or request_id not in self._active_logs:
            return

        self._active_logs[request_id].embedding = EmbeddingLog(
            slot_name=slot_name,
            model=model,
            dimension=dimension,
            time_ms=time_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )

    def log_search(
        self,
        request_id: str,
        strategy: str,
        slots_used: list[str],
        top_k: int,
        candidates: int,
        time_ms: float,
        vector_scores: Optional[list[float]] = None,
        final_scores: Optional[list[float]] = None,
    ) -> None:
        """Log search operation."""
        if not self.enabled or request_id not in self._active_logs:
            return

        self._active_logs[request_id].search = SearchLog(
            strategy=strategy,
            slots_used=slots_used,
            top_k_requested=top_k,
            candidates_retrieved=candidates,
            time_ms=time_ms,
            vector_scores=vector_scores or [],
            final_scores=final_scores or [],
        )

    def log_selection(
        self,
        request_id: str,
        doc_ids: list[str],
        reason: str,
    ) -> None:
        """Log document selection."""
        if not self.enabled or request_id not in self._active_logs:
            return

        log = self._active_logs[request_id]
        log.selected_doc_ids = doc_ids
        log.selection_reason = reason

    def log_response(
        self,
        request_id: str,
        llm_model: str,
        context_tokens: int,
        response_tokens: int,
        time_ms: float,
        cost_usd: float,
    ) -> None:
        """Log LLM response generation."""
        if not self.enabled or request_id not in self._active_logs:
            return

        self._active_logs[request_id].response = ResponseLog(
            llm_model=llm_model,
            context_tokens=context_tokens,
            response_tokens=response_tokens,
            time_ms=time_ms,
            cost_usd=cost_usd,
        )

    def log_verification(
        self,
        request_id: str,
        citation_count: int,
        accuracy: float,
        removed: list[int],
    ) -> None:
        """Log citation verification."""
        if not self.enabled or request_id not in self._active_logs:
            return

        log = self._active_logs[request_id]
        log.citation_count = citation_count
        log.citation_accuracy = accuracy
        log.removed_citations = removed

    def finalize(self, request_id: str) -> Optional[RetrievalLog]:
        """
        Finalize and buffer a log entry.

        Args:
            request_id: Request ID to finalize.

        Returns:
            The finalized RetrievalLog or None if not found.
        """
        if not self.enabled or request_id not in self._active_logs:
            return None

        log = self._active_logs.pop(request_id)
        timers = self._timers.pop(request_id, {})

        # Calculate total time
        if "start" in timers:
            log.total_time_ms = (time.time() - timers["start"]) * 1000

        # Calculate total cost
        total_cost = 0.0
        if log.embedding:
            total_cost += log.embedding.cost_usd
        if log.response:
            total_cost += log.response.cost_usd
        log.total_cost_usd = total_cost

        # Add to buffer
        self._buffer.append(log)

        # Flush if buffer is full
        if len(self._buffer) >= self.buffer_size:
            self.flush()

        return log

    @asynccontextmanager
    async def timing(self, request_id: str, stage: str):
        """
        Context manager for timing a stage.

        Usage:
            async with logger.timing(request_id, "embedding"):
                result = await embed(text)
        """
        if not self.enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            if request_id in self._timers:
                self._timers[request_id][stage] = elapsed_ms

    def flush(self) -> int:
        """
        Flush buffer to storage.

        Returns:
            Number of logs written.
        """
        if not self._buffer:
            return 0

        # Group by date for file organization
        date_logs: dict[str, list[RetrievalLog]] = {}
        for log in self._buffer:
            date = log.timestamp[:10]  # YYYY-MM-DD
            if date not in date_logs:
                date_logs[date] = []
            date_logs[date].append(log)

        count = 0
        for date, logs in date_logs.items():
            file_path = self.storage_path / f"retrieval_{date}.jsonl"

            with open(file_path, "a", encoding="utf-8") as f:
                for log in logs:
                    f.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")
                    count += 1

        self._buffer = []
        return count

    def get_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> list[RetrievalLog]:
        """
        Retrieve logs from storage.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            limit: Maximum logs to return.

        Returns:
            List of RetrievalLog objects.
        """
        logs = []

        # Get all log files
        log_files = sorted(self.storage_path.glob("retrieval_*.jsonl"))

        for file_path in log_files:
            # Extract date from filename
            date = file_path.stem.replace("retrieval_", "")

            # Filter by date range
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)

                        # Reconstruct nested dataclasses
                        if data.get("embedding"):
                            data["embedding"] = EmbeddingLog(**data["embedding"])
                        if data.get("search"):
                            data["search"] = SearchLog(**data["search"])
                        if data.get("response"):
                            data["response"] = ResponseLog(**data["response"])

                        logs.append(RetrievalLog(**data))

                        if len(logs) >= limit:
                            return logs

        return logs

    def get_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get aggregated statistics from logs.

        Returns:
            Statistics dictionary.
        """
        logs = self.get_logs(start_date, end_date, limit=10000)

        if not logs:
            return {
                "total_queries": 0,
                "avg_total_time_ms": 0.0,
                "avg_total_cost_usd": 0.0,
                "avg_embedding_time_ms": 0.0,
                "avg_search_time_ms": 0.0,
                "avg_response_time_ms": 0.0,
                "slot_usage": {},
                "search_strategy_distribution": {},
                "avg_citation_accuracy": None,
                "date_range": {"start": None, "end": None},
            }

        # Aggregate stats
        total_time = sum(log.total_time_ms for log in logs)
        total_cost = sum(log.total_cost_usd for log in logs)

        embedding_times = [
            log.embedding.time_ms for log in logs if log.embedding
        ]
        search_times = [
            log.search.time_ms for log in logs if log.search
        ]
        response_times = [
            log.response.time_ms for log in logs if log.response
        ]

        # Slot usage
        slot_usage: dict[str, int] = {}
        for log in logs:
            if log.embedding:
                slot = log.embedding.slot_name
                slot_usage[slot] = slot_usage.get(slot, 0) + 1

        # Search strategy distribution
        strategy_dist: dict[str, int] = {}
        for log in logs:
            if log.search:
                strategy = log.search.strategy
                strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1

        # Citation accuracy
        accuracies = [
            log.citation_accuracy
            for log in logs
            if log.citation_accuracy is not None
        ]

        return {
            "total_queries": len(logs),
            "avg_total_time_ms": total_time / len(logs) if logs else 0,
            "avg_total_cost_usd": total_cost / len(logs) if logs else 0,
            "avg_embedding_time_ms": (
                sum(embedding_times) / len(embedding_times) if embedding_times else 0
            ),
            "avg_search_time_ms": (
                sum(search_times) / len(search_times) if search_times else 0
            ),
            "avg_response_time_ms": (
                sum(response_times) / len(response_times) if response_times else 0
            ),
            "slot_usage": slot_usage,
            "search_strategy_distribution": strategy_dist,
            "avg_citation_accuracy": (
                sum(accuracies) / len(accuracies) if accuracies else None
            ),
            "date_range": {
                "start": logs[0].timestamp[:10] if logs else None,
                "end": logs[-1].timestamp[:10] if logs else None,
            },
        }
