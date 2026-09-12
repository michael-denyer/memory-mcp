"""MCP tools package - import all tool modules to register them."""

# Import all tool modules to register their @mcp.tool decorators
from memory_mcp.server.tools import (
    cold_storage,
    hot_cache,
    maintenance,
    relationships,
    retrieval,
    seeding,
    sessions,
)

__all__ = [
    "cold_storage",
    "hot_cache",
    "maintenance",
    "relationships",
    "retrieval",
    "seeding",
    "sessions",
]
