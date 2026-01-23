"""Mining tools: log_output, mining_status, review/approve/reject candidates, run_mining."""

from typing import Annotated

from pydantic import Field

from memory_mcp.helpers import invalid_memory_type_error, parse_memory_type
from memory_mcp.logging import record_mining
from memory_mcp.responses import error_response, success_response
from memory_mcp.server.app import (
    get_auto_project_id,
    get_current_session_id,
    log,
    mcp,
    settings,
    storage,
)
from memory_mcp.storage import MemorySource


@mcp.tool
def log_output(
    content: Annotated[str, Field(description="Output content to log for pattern mining")],
    session_id: Annotated[
        str | None, Field(description="Session ID for provenance tracking")
    ] = None,
) -> dict:
    """Log an output for pattern mining. Called automatically or manually."""
    if not settings.mining_enabled:
        return error_response("Mining is disabled")

    # Validate content not empty
    if not content or not content.strip():
        return error_response("Content cannot be empty")

    # Validate content length
    if len(content) > settings.max_content_length:
        return error_response(
            f"Content too long ({len(content)} chars). Max: {settings.max_content_length}"
        )

    # Use auto-session if not explicitly provided (zero-config session tracking)
    effective_session_id = session_id or get_current_session_id()

    # Get project_id for project-scoped mining
    project_id = get_auto_project_id()

    log_id = storage.log_output(content, session_id=effective_session_id, project_id=project_id)
    return success_response("Output logged", log_id=log_id, project_id=project_id)


@mcp.tool
def mining_status() -> dict:
    """Show pattern mining statistics."""
    candidates = storage.get_promotion_candidates()
    outputs = storage.get_recent_outputs(hours=24)

    return {
        "enabled": settings.mining_enabled,
        "promotion_threshold": settings.promotion_threshold,
        "candidates_ready": len(candidates),
        "outputs_last_24h": len(outputs),
        "candidates": [
            {
                "id": c.id,
                "pattern": c.pattern[:100] + "..." if len(c.pattern) > 100 else c.pattern,
                "type": c.pattern_type,
                "occurrences": c.occurrence_count,
            }
            for c in candidates[:10]
        ],
    }


@mcp.tool
def review_candidates() -> list[dict]:
    """Review mined patterns that are ready for promotion."""
    candidates = storage.get_promotion_candidates()
    return [
        {
            "id": c.id,
            "pattern": c.pattern,
            "type": c.pattern_type,
            "occurrences": c.occurrence_count,
            "first_seen": c.first_seen.isoformat(),
            "last_seen": c.last_seen.isoformat(),
        }
        for c in candidates
    ]


@mcp.tool
def approve_candidate(
    pattern_id: Annotated[int, Field(description="ID of mined pattern to approve")],
    memory_type: Annotated[str, Field(description="Type to assign")] = "pattern",
    tags: Annotated[list[str] | None, Field(description="Tags to assign")] = None,
) -> dict:
    """Approve a mined pattern, storing it as a memory and optionally promoting to hot cache."""
    candidates = storage.get_promotion_candidates()
    candidate = next((c for c in candidates if c.id == pattern_id), None)

    if not candidate:
        return error_response(f"Candidate #{pattern_id} not found")

    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return invalid_memory_type_error()

    # Use project_id for project-aware memory
    project_id = get_auto_project_id()

    memory_id, is_new = storage.store_memory(
        content=candidate.pattern,
        memory_type=mem_type,
        source=MemorySource.MINED,
        tags=tags or [],
        project_id=project_id,
    )

    storage.promote_to_hot(memory_id)
    storage.delete_mined_pattern(pattern_id)

    if is_new:
        return success_response(
            f"Pattern approved as memory #{memory_id} and promoted to hot cache",
            memory_id=memory_id,
        )
    else:
        return success_response(
            f"Pattern matched existing memory #{memory_id} (tags merged, promoted to hot cache)",
            memory_id=memory_id,
            was_duplicate=True,
        )


@mcp.tool
def reject_candidate(
    pattern_id: Annotated[int, Field(description="ID of mined pattern to reject")],
) -> dict:
    """Reject a mined pattern, removing it from candidates."""
    if storage.delete_mined_pattern(pattern_id):
        return success_response(f"Pattern #{pattern_id} rejected")
    return error_response(f"Pattern #{pattern_id} not found")


@mcp.tool
def bulk_reject_candidates(
    pattern_ids: Annotated[
        list[int] | None,
        Field(description="List of pattern IDs to reject"),
    ] = None,
    pattern_type_prefix: Annotated[
        str | None,
        Field(
            description="Reject patterns whose type starts with this prefix "
            "(e.g., 'entity_' matches entity_misc, entity_person, entity_org)"
        ),
    ] = None,
) -> dict:
    """Bulk reject multiple mining candidates at once.

    Provide either pattern_ids (list of specific IDs) OR pattern_type_prefix
    (e.g., 'entity_' to reject all entity extractions).
    """
    if not pattern_ids and not pattern_type_prefix:
        return error_response("Must provide either pattern_ids or pattern_type_prefix")

    try:
        deleted = storage.delete_mined_patterns_bulk(
            pattern_ids=pattern_ids,
            pattern_type_prefix=pattern_type_prefix,
        )
        return success_response(
            f"Bulk rejected {deleted} patterns",
            deleted_count=deleted,
            pattern_ids=pattern_ids,
            pattern_type_prefix=pattern_type_prefix,
        )
    except ValueError as e:
        return error_response(str(e))


@mcp.tool
def run_mining(
    hours: Annotated[int, Field(description="Hours of logs to process")] = 24,
) -> dict:
    """Run pattern mining on recent output logs.

    Extracts patterns (imports, facts, commands, code) from logged outputs
    and updates the mined_patterns table with occurrence counts.

    Mining is project-scoped: only logs from the current project are processed.
    """
    log.info("run_mining() called: hours={}", hours)
    if not settings.mining_enabled:
        return error_response("Mining is disabled")

    from memory_mcp.mining import run_mining as _run_mining

    # Get project_id for project-scoped mining
    project_id = get_auto_project_id()

    result = _run_mining(storage, hours=hours, project_id=project_id)
    log.info(
        "Mining complete: {} outputs processed, {} patterns found",
        result["outputs_processed"],
        result["patterns_found"],
    )

    # Record mining metrics
    record_mining(
        patterns_found=result["patterns_found"],
        patterns_new=result["new_memories"],
        patterns_updated=result["updated_patterns"],
    )

    return {"success": True, **result}
