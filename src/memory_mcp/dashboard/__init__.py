"""Web dashboard for Memory MCP."""


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8765,
    reload: bool = False,
) -> None:
    """Run the dashboard server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload for development.
    """
    import uvicorn

    uvicorn.run(
        "memory_mcp.dashboard.app:app" if reload else "memory_mcp.dashboard.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


__all__ = ["run_dashboard"]
