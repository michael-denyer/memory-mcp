"""Metrics collection and observability for Memory MCP.

This module re-exports metrics functionality from logging.py for cleaner imports.
New code should import from this module; logging.py exports are for backwards
compatibility.

Example:
    from memory_mcp.metrics import metrics, record_recall
"""

# Re-export all metrics from logging module
from memory_mcp.logging import (
    Metrics,
    metrics,
    record_hot_cache_change,
    record_mining,
    record_recall,
    record_store,
    update_hot_cache_stats,
)

__all__ = [
    "Metrics",
    "metrics",
    "record_hot_cache_change",
    "record_mining",
    "record_recall",
    "record_store",
    "update_hot_cache_stats",
]
