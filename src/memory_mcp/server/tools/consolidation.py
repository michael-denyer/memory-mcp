"""Memory consolidation tools: preview_consolidation, run_consolidation."""

from memory_mcp.helpers import invalid_memory_type_error, parse_memory_type
from memory_mcp.responses import success_response
from memory_mcp.server.app import mcp, storage


@mcp.tool
def preview_consolidation(
    memory_type: str | None = None,
) -> dict:
    """Preview memory consolidation without making changes.

    Shows clusters of similar memories that could be merged.
    Use this before running actual consolidation to review changes.

    Args:
        memory_type: Filter by type (project/pattern/reference/conversation)

    Returns:
        Preview of clusters and potential space savings
    """
    mem_type = None
    if memory_type:
        mem_type = parse_memory_type(memory_type)
        if mem_type is None:
            return invalid_memory_type_error()

    result = storage.preview_consolidation(memory_type=mem_type)
    return success_response(
        f"Found {result['cluster_count']} clusters "
        f"({result['memories_to_delete']} memories can be merged)",
        **result,
    )


@mcp.tool
def run_consolidation(
    memory_type: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Consolidate similar memories by merging near-duplicates.

    Finds clusters of semantically similar memories and merges them,
    keeping the most accessed/valuable one as representative.

    Args:
        memory_type: Filter by type (project/pattern/reference/conversation)
        dry_run: If True (default), only preview without making changes

    Returns:
        Consolidation results (clusters processed, memories deleted)
    """
    mem_type = None
    if memory_type:
        mem_type = parse_memory_type(memory_type)
        if mem_type is None:
            return invalid_memory_type_error()

    result = storage.run_consolidation(memory_type=mem_type, dry_run=dry_run)

    if dry_run:
        return success_response(
            f"DRY RUN: Would process {result.get('cluster_count', 0)} clusters",
            **result,
        )

    return success_response(
        f"Consolidated {result['clusters_processed']} clusters, "
        f"deleted {result['memories_deleted']} duplicate memories",
        **result,
    )
