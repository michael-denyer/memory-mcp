"""Logging configuration for memory MCP server.

IMPORTANT: MCP servers using STDIO transport must log to stderr only.
stdout is reserved for the MCP protocol communication.
"""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add stderr handler with sensible format for MCP
LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
logger.add(sys.stderr, format=LOG_FORMAT, level="INFO", colorize=True)


def get_logger(name: str) -> "logger":
    """Get a logger instance bound to a module name."""
    return logger.bind(name=name)
