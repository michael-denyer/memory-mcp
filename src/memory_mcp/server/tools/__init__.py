"""MCP tools package - import all tool modules to register them."""

# Import all tool modules to register their @mcp.tool decorators
from memory_mcp.server.tools import (
    cold_storage,
    consolidation,
    contradictions,
    hot_cache,
    maintenance,
    mining,
    predictions,
    relationships,
    retrieval,
    seeding,
    sessions,
    trust,
)

__all__ = [
    "cold_storage",
    "consolidation",
    "contradictions",
    "hot_cache",
    "maintenance",
    "mining",
    "predictions",
    "relationships",
    "retrieval",
    "seeding",
    "sessions",
    "trust",
]
