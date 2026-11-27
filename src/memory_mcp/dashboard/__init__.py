"""Web dashboard for Memory MCP."""

from ipaddress import ip_address


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
    if host.lower() == "localhost":
        host = "127.0.0.1"
    if not ip_address(host).is_loopback:
        raise ValueError("The unauthenticated dashboard requires a loopback address")

    import uvicorn

    uvicorn.run(
        "memory_mcp.dashboard.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


__all__ = ["run_dashboard"]
