"""Engram-inspired memory MCP server with a hook-injected hot cache and semantic recall."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hot-memory-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"  # Fallback for editable installs without metadata
