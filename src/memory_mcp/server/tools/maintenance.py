"""Maintenance tools: db_maintenance, run_cleanup, validate_embeddings, db_info, etc."""

import os
from typing import Annotated

from pydantic import Field

from memory_mcp.responses import (
    AuditEntryResponse,
    AuditHistoryResponse,
    MaintenanceResponse,
    VectorRebuildResponse,
    error_response,
)
from memory_mcp.server.app import log, mcp, settings, storage
from memory_mcp.storage import AuditOperation


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


@mcp.tool
def run_cleanup() -> dict:
    """Run comprehensive cleanup of stale data.

    Performs all cleanup operations in one call:
    - Demotes stale hot memories (not accessed in demotion_days)
    - Expires old pending mining patterns (30+ days without activity)
    - Deletes old output logs (based on log_retention_days)
    - Deletes stale memories by type-specific retention policies

    Use this periodically to keep the database lean. For just database
    compaction, use db_maintenance() instead.
    """
    log.info("run_cleanup() called")
    result = storage.run_full_cleanup()
    log.info(
        "Cleanup complete: {} hot demoted, {} patterns expired, "
        "{} logs deleted, {} memories deleted",
        result["hot_cache_demoted"],
        result["patterns_expired"],
        result["logs_deleted"],
        result["memories_deleted"],
    )
    return {"success": True, **result}


@mcp.tool
def validate_embeddings() -> dict:
    """Check if the embedding model has changed since database was created.

    If the model or dimension has changed, existing embeddings may be
    incompatible and memories may need re-embedding.

    Returns validation status and details about any mismatches.
    """
    from memory_mcp.embeddings import get_embedding_engine

    engine = get_embedding_engine()
    result = storage.validate_embedding_model(
        current_model=settings.embedding_model,
        current_dim=engine.dimension,
    )

    if not result["valid"]:
        return {
            "success": False,
            "warning": "Embedding model has changed! Existing embeddings may be invalid.",
            **result,
        }

    return {"success": True, **result}


@mcp.tool
def db_info() -> dict:
    """Get database information including schema version and size."""
    db_size = os.path.getsize(storage.db_path) if storage.db_path.exists() else 0
    stats = storage.get_stats()

    return {
        "db_path": str(storage.db_path),
        "db_size_bytes": db_size,
        "db_size_mb": round(db_size / (1024 * 1024), 2),
        "schema_version": storage.get_schema_version(),
        **stats,
    }


@mcp.tool
def embedding_info() -> dict:
    """Get embedding provider and cache information."""
    from memory_mcp.embeddings import get_embedding_engine

    engine = get_embedding_engine()
    cache_stats = engine.cache_stats()

    return {
        "provider": cache_stats["provider"],
        "dimension": engine.dimension,
        "cache_size": cache_stats["size"],
        "cache_max_size": cache_stats["max_size"],
    }


@mcp.tool
def audit_history(
    limit: int = 50,
    operation: str | None = None,
) -> AuditHistoryResponse:
    """Get recent audit log entries for destructive operations.

    Shows history of operations like delete_memory, demote, maintenance,
    unlink_memories, etc. Useful for understanding what changed and when.

    Args:
        limit: Maximum entries to return (default 50, max 500).
        operation: Filter by operation type (e.g., "delete_memory", "demote_memory").
                   Available types: delete_memory, demote_memory, demote_stale,
                   delete_pattern, expire_patterns, cleanup_memories, maintenance,
                   unlink_memories.
    """
    op_enum = None
    if operation:
        try:
            op_enum = AuditOperation(operation)
        except ValueError:
            valid_ops = [op.value for op in AuditOperation]
            log.warning("Invalid operation '{}', valid options: {}", operation, valid_ops)

    entries = storage.audit_history(limit=limit, operation=op_enum)

    return AuditHistoryResponse(
        entries=[
            AuditEntryResponse(
                id=e.id,
                operation=e.operation,
                target_type=e.target_type,
                target_id=e.target_id,
                details=e.details,
                timestamp=e.timestamp,
            )
            for e in entries
        ],
        count=len(entries),
    )


@mcp.tool
def db_rebuild_vectors(
    batch_size: Annotated[
        int, Field(default=100, description="Memories to embed per batch (default 100)")
    ] = 100,
) -> VectorRebuildResponse | dict:
    """Rebuild all memory vectors with the current embedding model.

    Use this when:
    - Switching to a different embedding model
    - Fixing dimension mismatch errors
    - Recovering from corrupted vector data

    This operation:
    1. Clears all existing vectors (memories are preserved)
    2. Re-embeds every memory with the current model
    3. Updates stored model info

    Warning: This can take time for large databases. Progress is logged.
    """
    try:
        result = storage.rebuild_vectors(batch_size=batch_size)

        return VectorRebuildResponse(
            success=True,
            vectors_cleared=result["vectors_cleared"],
            memories_total=result["memories_total"],
            memories_embedded=result["memories_embedded"],
            memories_failed=result["memories_failed"],
            new_dimension=result["new_dimension"],
            new_model=result["new_model"],
            message=(
                f"Rebuilt {result['memories_embedded']}/{result['memories_total']} "
                f"memory vectors with {result['new_model']} (dim={result['new_dimension']})"
            ),
        )
    except Exception as e:
        log.error("Vector rebuild failed: {}", e)
        return error_response(f"Vector rebuild failed: {e}")
