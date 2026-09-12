"""Maintenance tools: db_maintenance."""

from memory_mcp.responses import (
    MaintenanceResponse,
)
from memory_mcp.server.app import log, mcp, storage


@mcp.tool
def db_maintenance() -> MaintenanceResponse:
    """Run database maintenance (vacuum, analyze, auto-demote stale).

    Compacts the database to reclaim unused space, updates
    query planner statistics, and demotes stale hot memories
    (if auto_demote is enabled).
    """
    log.info("db_maintenance() called")
    result = storage.maintenance()
    log.info(
        "Maintenance complete: {} bytes reclaimed, {} memories, {} auto-demoted",
        result["bytes_reclaimed"],
        result["memory_count"],
        result["auto_demoted_count"],
    )
    return MaintenanceResponse(**result)
