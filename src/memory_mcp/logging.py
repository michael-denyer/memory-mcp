"""Logging configuration for memory MCP server.

IMPORTANT: MCP servers using STDIO transport must log to stderr only.
stdout is reserved for the MCP protocol communication.
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any

from loguru import logger

# Remove default handler
logger.remove()

# Pretty format for human readability
PRETTY_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


def json_serializer(record: dict[str, Any]) -> str:
    """Serialize log record to JSON format."""
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["extra"].get("name", record["name"]),
        "function": record["function"],
        "message": record["message"],
    }
    # Include extra fields (excluding internal ones)
    for key, value in record["extra"].items():
        if key != "name":
            # Convert non-serializable types
            if isinstance(value, datetime):
                subset[key] = value.isoformat()
            elif hasattr(value, "__dict__"):
                subset[key] = str(value)
            else:
                subset[key] = value
    return json.dumps(subset, default=str)


def json_sink(message: Any) -> None:
    """Sink that writes JSON-formatted logs to stderr."""
    record = message.record
    sys.stderr.write(json_serializer(record) + "\n")


def configure_logging(level: str = "INFO", log_format: str = "pretty") -> None:
    """Configure logging with specified level and format.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_format: 'pretty' for human-readable or 'json' for structured.
    """
    # Remove any existing handlers
    logger.remove()

    if log_format.lower() == "json":
        logger.add(json_sink, level=level.upper(), format="{message}")
    else:
        logger.add(sys.stderr, format=PRETTY_FORMAT, level=level.upper(), colorize=True)


# Default configuration
configure_logging()


def get_logger(name: str):
    """Get a logger instance bound to a module name.

    Args:
        name: Module name to bind to the logger for filtering and identification.

    Returns:
        A loguru logger instance with the module name bound.
    """
    return logger.bind(name=name)


# ========== Metrics Tracking ==========


class Metrics:
    """Simple in-memory metrics collector for observability.

    Tracks counters and gauges for key operations. Thread-safe via GIL
    for simple counter increments.

    Counters: Monotonically increasing values (e.g., total recalls)
    Gauges: Point-in-time values (e.g., hot cache size)
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._start_time = datetime.now(timezone.utc)

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric to a specific value."""
        self._gauges[name] = value

    def get_counter(self, name: str) -> int:
        """Get a counter value."""
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float | None:
        """Get a gauge value."""
        return self._gauges.get(name)

    def snapshot(self) -> dict[str, Any]:
        """Return all metrics as a dict snapshot."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return {
            "uptime_seconds": round(uptime, 1),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self._counters.clear()
        self._gauges.clear()
        self._start_time = datetime.now(timezone.utc)


# Global metrics instance
metrics = Metrics()


# Convenience functions for common metrics
def record_recall(
    query_length: int,
    results_count: int,
    gated_count: int,
    hot_hit: bool,
    threshold: float,
    query_preview: str | None = None,
    mode: str | None = None,
    elapsed_ms: float | None = None,
) -> None:
    """Record metrics for a recall operation.

    Args:
        query_length: Length of the search query.
        results_count: Number of results returned.
        gated_count: Number of results below threshold (not returned).
        hot_hit: Whether any result came from hot cache.
        threshold: Similarity threshold used for gating.
        query_preview: First 80 chars of query (for debugging).
        mode: Recall mode (precision/balanced/exploratory).
        elapsed_ms: Time taken for recall operation in milliseconds.
    """
    metrics.increment("recall.total")
    metrics.increment("recall.results_returned", results_count)
    metrics.increment("recall.results_gated", gated_count)
    if hot_hit:
        metrics.increment("recall.hot_hits")
    if results_count == 0:
        metrics.increment("recall.empty")

    log = get_logger("metrics")

    # Build log message with optional context
    preview = f'"{query_preview[:80]}"' if query_preview else f"({query_length} chars)"
    mode_str = mode or "balanced"
    elapsed_str = f" elapsed={elapsed_ms:.1f}ms" if elapsed_ms is not None else ""

    log.debug(
        "recall: query={} mode={} results={} gated={} hot={} threshold={:.2f}{}",
        preview,
        mode_str,
        results_count,
        gated_count,
        hot_hit,
        threshold,
        elapsed_str,
    )


def record_store(memory_type: str, merged: bool, contradictions: int) -> None:
    """Record metrics for a store operation.

    Args:
        memory_type: Type of memory stored (project, pattern, etc.).
        merged: Whether the memory was merged with an existing one.
        contradictions: Number of potential contradictions detected.
    """
    metrics.increment("store.total")
    metrics.increment(f"store.type.{memory_type}")
    if merged:
        metrics.increment("store.merged")
    if contradictions > 0:
        metrics.increment("store.contradictions_found", contradictions)


def record_mining(patterns_found: int, patterns_new: int, patterns_updated: int) -> None:
    """Record metrics for a mining operation.

    Args:
        patterns_found: Total patterns extracted from logs.
        patterns_new: New patterns added to the database.
        patterns_updated: Existing patterns with incremented counts.
    """
    metrics.increment("mining.runs")
    metrics.increment("mining.patterns_found", patterns_found)
    metrics.increment("mining.patterns_new", patterns_new)
    metrics.increment("mining.patterns_updated", patterns_updated)


def record_hot_cache_change(
    promoted: bool = False, demoted: bool = False, evicted: bool = False
) -> None:
    """Record metrics for hot cache changes.

    Args:
        promoted: A memory was promoted to hot cache.
        demoted: A memory was demoted from hot cache.
        evicted: A memory was evicted to make room for another.
    """
    if promoted:
        metrics.increment("hot_cache.promotions")
    if demoted:
        metrics.increment("hot_cache.demotions")
    if evicted:
        metrics.increment("hot_cache.evictions")


def update_hot_cache_stats(size: int, max_size: int, pinned: int) -> None:
    """Update hot cache gauge metrics.

    Args:
        size: Current number of items in hot cache.
        max_size: Maximum capacity of hot cache.
        pinned: Number of pinned (protected from eviction) items.
    """
    metrics.set_gauge("hot_cache.size", float(size))
    metrics.set_gauge("hot_cache.max_size", float(max_size))
    metrics.set_gauge("hot_cache.pinned", float(pinned))

    utilization = size / max_size if max_size > 0 else 0.0
    metrics.set_gauge("hot_cache.utilization", utilization)


def record_promotion_rejection(reason: str, memory_id: int | None = None) -> None:
    """Record metrics for promotion rejection.

    Tracks why memories were rejected for hot cache promotion to help
    identify patterns and tune thresholds.

    Args:
        reason: Rejection reason code (category_ineligible, threshold_not_met,
                low_helpfulness)
        memory_id: Optional memory ID for detailed logging
    """
    metrics.increment("promotion.rejections.total")
    metrics.increment(f"promotion.rejections.{reason}")

    log = get_logger("promotion")
    log.debug("promotion_rejected: reason={} memory_id={}", reason, memory_id)


def get_promotion_rejection_summary() -> dict[str, int]:
    """Get summary of promotion rejection reasons.

    Returns:
        Dict mapping rejection reasons to counts
    """
    prefix = "promotion.rejections."
    return {
        k.replace(prefix, ""): v
        for k, v in metrics._counters.items()
        if k.startswith(prefix) and k != f"{prefix}total"
    }
