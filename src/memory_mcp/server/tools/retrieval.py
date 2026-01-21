"""Retrieval quality tracking tools: mark_memory_used, retrieval_quality_stats."""

from memory_mcp.responses import success_response
from memory_mcp.server.app import mcp, storage


@mcp.tool
def mark_memory_used(
    memory_id: int,
    feedback: str | None = None,
) -> dict:
    """Mark a memory as actually used/helpful after recall.

    Call this when a recalled memory was useful in your response.
    Helps improve ranking by tracking which memories are valuable.

    Args:
        memory_id: ID of the memory that was useful
        feedback: Optional feedback (e.g., "helpful", "partially_helpful")

    Returns:
        Success response with update count
    """
    updated = storage.mark_retrieval_used(memory_id, feedback=feedback)
    if updated > 0:
        return success_response(
            f"Marked memory {memory_id} as used",
            updated_count=updated,
        )
    return success_response(
        "No retrieval event found to update (tracking may be disabled)",
        updated_count=0,
    )


@mcp.tool
def retrieval_quality_stats(
    memory_id: int | None = None,
    days: int = 30,
) -> dict:
    """Get retrieval quality statistics.

    Shows which memories are frequently retrieved and actually used.
    Helps identify high-value and low-utility memories.

    Args:
        memory_id: Get stats for specific memory (None for global)
        days: How many days back to analyze (default 30)

    Returns:
        Statistics on retrieval and usage patterns
    """
    stats = storage.get_retrieval_stats(memory_id=memory_id, days=days)
    return success_response("Retrieval quality stats", **stats)
