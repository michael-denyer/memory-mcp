"""FastMCP server package for Memory MCP.

This package provides the MCP server with memory tools and hot cache resources.
"""

from memory_mcp.logging import get_logger

# Import all tools to register them
# Re-export app module for internal state access (like _auto_bootstrap_attempted)
from memory_mcp.server import (
    app,
    tools,  # noqa: F401
)

# Import app module to get mcp instance, resources, and helper functions
from memory_mcp.server.app import (
    format_memories_for_llm,
    get_promotion_suggestions,
    get_similarity_confidence,
    hot_cache_resource,
    mcp,
    settings,
    storage,
)

# Re-export key tools for backwards compatibility with tests
from memory_mcp.server.tools.cold_storage import recall, remember
from memory_mcp.server.tools.mining import log_output

log = get_logger("server")


def main():
    """Run the MCP server."""
    log.info("Starting Memory MCP server...")
    mcp.run()


__all__ = [
    "main",
    "mcp",
    "settings",
    "storage",
    "app",
    "get_promotion_suggestions",
    "get_similarity_confidence",
    "format_memories_for_llm",
    "hot_cache_resource",
    "recall",
    "remember",
    "log_output",
]
