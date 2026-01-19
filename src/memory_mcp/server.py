"""FastMCP server with memory tools and hot cache resource."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from memory_mcp.config import get_settings
from memory_mcp.logging import get_logger
from memory_mcp.storage import Memory, MemorySource, MemoryType, RecallResult, Storage

log = get_logger("server")

# Initialize
settings = get_settings()
storage = Storage(settings)
mcp = FastMCP("memory-mcp")

log.info("Memory MCP server initialized")


# ========== Pydantic Models for Tool Responses ==========


class MemoryResponse(BaseModel):
    """Response for a single memory."""

    id: int
    content: str
    memory_type: str
    source: str
    is_hot: bool
    tags: list[str]
    access_count: int
    similarity: float | None = None
    created_at: str


class RecallResponse(BaseModel):
    """Response for recall operation."""

    memories: list[MemoryResponse]
    confidence: str
    gated_count: int
    hint: str


class StatsResponse(BaseModel):
    """Response for stats operation."""

    total_memories: int
    hot_cache_count: int
    by_type: dict[str, int]
    by_source: dict[str, int]


class HotCacheResponse(BaseModel):
    """Response for hot cache status."""

    items: list[MemoryResponse]
    max_items: int
    current_count: int


def memory_to_response(m: Memory) -> MemoryResponse:
    """Convert Memory to response model."""
    return MemoryResponse(
        id=m.id,
        content=m.content,
        memory_type=m.memory_type.value,
        source=m.source.value,
        is_hot=m.is_hot,
        tags=m.tags,
        access_count=m.access_count,
        similarity=m.similarity,
        created_at=m.created_at.isoformat(),
    )


def success_response(message: str, **extra) -> dict:
    """Create a success response dict."""
    return {"success": True, "message": message, **extra}


def error_response(error: str) -> dict:
    """Create an error response dict."""
    return {"success": False, "error": error}


# ========== Cold Storage Tools ==========


def parse_memory_type(memory_type: str) -> MemoryType | None:
    """Parse memory type string, returning None if invalid."""
    try:
        return MemoryType(memory_type)
    except ValueError:
        return None


def invalid_memory_type_error() -> dict:
    """Return error for invalid memory type."""
    return error_response(f"Invalid memory_type. Use: {[t.value for t in MemoryType]}")


@mcp.tool
def remember(
    content: Annotated[str, Field(description="The content to remember")],
    memory_type: Annotated[
        str,
        Field(
            description="Type: 'project' (project facts), 'pattern' (code patterns), 'reference' (docs), 'conversation' (discussion facts)"
        ),
    ] = "project",
    tags: Annotated[
        list[str] | None, Field(description="Tags for categorization")
    ] = None,
) -> dict:
    """Store a new memory. Returns the memory ID."""
    log.debug("remember() called: type={} tags={}", memory_type, tags)

    # Validate content length
    if len(content) > settings.max_content_length:
        return error_response(
            f"Content too long ({len(content)} chars). Max: {settings.max_content_length}"
        )

    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return invalid_memory_type_error()

    memory_id = storage.store_memory(
        content=content,
        memory_type=mem_type,
        source=MemorySource.MANUAL,
        tags=tags or [],
    )

    return success_response(f"Stored as memory #{memory_id}", memory_id=memory_id)


@mcp.tool
def recall(
    query: Annotated[str, Field(description="Search query for semantic similarity")],
    limit: Annotated[int, Field(description="Maximum results to return (1-100)")] = 5,
    threshold: Annotated[
        float, Field(description="Minimum similarity (0.0-1.0) to include results")
    ] = None,
) -> RecallResponse:
    """Semantic search with confidence gating.

    Returns memories above threshold with confidence level:
    - 'high': Top result > 0.85 similarity - use confidently
    - 'medium': Results above threshold - use but verify
    - 'low': No results passed threshold - reason from scratch
    """
    # Validate and clamp inputs
    limit = max(1, min(settings.max_recall_limit, limit)) if limit else settings.default_recall_limit
    if threshold is not None:
        threshold = max(0.0, min(1.0, threshold))

    log.debug("recall() called: query='{}' limit={} threshold={}", query[:50], limit, threshold)
    result = storage.recall(query=query, limit=limit, threshold=threshold)

    hints = {
        "high": "High confidence match - use this information directly",
        "medium": "Medium confidence - verify this applies to current context",
        "low": "No confident matches found - reason from scratch or try different query",
    }

    return RecallResponse(
        memories=[memory_to_response(m) for m in result.memories],
        confidence=result.confidence,
        gated_count=result.gated_count,
        hint=hints[result.confidence],
    )


@mcp.tool
def recall_by_tag(
    tag: Annotated[str, Field(description="Tag to filter by")],
    limit: Annotated[int, Field(description="Maximum results")] = 10,
) -> list[MemoryResponse]:
    """Get memories with a specific tag."""
    memories = storage.recall_by_tag(tag=tag, limit=limit)
    return [memory_to_response(m) for m in memories]


@mcp.tool
def forget(
    memory_id: Annotated[int, Field(description="ID of memory to delete")],
) -> dict:
    """Delete a memory permanently."""
    if storage.delete_memory(memory_id):
        return success_response(f"Deleted memory #{memory_id}")
    return error_response(f"Memory #{memory_id} not found")


@mcp.tool
def list_memories(
    limit: Annotated[int, Field(description="Maximum results")] = 20,
    offset: Annotated[int, Field(description="Skip first N results")] = 0,
    memory_type: Annotated[str | None, Field(description="Filter by type")] = None,
) -> list[MemoryResponse] | dict:
    """List stored memories with pagination."""
    mem_type = None
    if memory_type:
        mem_type = parse_memory_type(memory_type)
        if mem_type is None:
            return invalid_memory_type_error()

    memories = storage.list_memories(limit=limit, offset=offset, memory_type=mem_type)
    return [memory_to_response(m) for m in memories]


@mcp.tool
def memory_stats() -> StatsResponse:
    """Get memory statistics."""
    stats = storage.get_stats()
    return StatsResponse(**stats)


# ========== Hot Cache Tools ==========


@mcp.tool
def hot_cache_status() -> HotCacheResponse:
    """Show current hot cache contents and stats."""
    hot_memories = storage.get_hot_memories()
    return HotCacheResponse(
        items=[memory_to_response(m) for m in hot_memories],
        max_items=settings.hot_cache_max_items,
        current_count=len(hot_memories),
    )


@mcp.tool
def promote(
    memory_id: Annotated[int, Field(description="ID of memory to promote to hot cache")],
) -> dict:
    """Manually promote a memory to hot cache for zero-latency access."""
    if storage.promote_to_hot(memory_id):
        return success_response(f"Memory #{memory_id} promoted to hot cache")
    return error_response(f"Failed to promote memory #{memory_id}")


@mcp.tool
def demote(
    memory_id: Annotated[int, Field(description="ID of memory to remove from hot cache")],
) -> dict:
    """Remove a memory from hot cache (keeps in cold storage)."""
    if storage.demote_from_hot(memory_id):
        return success_response(f"Memory #{memory_id} demoted from hot cache")
    return error_response(f"Failed to demote memory #{memory_id}")


# ========== Hot Cache Resource (Auto-Injection) ==========


@mcp.resource("memory://hot-cache")
def hot_cache_resource() -> str:
    """Auto-injectable system context with high-confidence patterns.

    Configure Claude Code to include this resource in system prompts
    for zero-latency access to frequently-used knowledge.
    """
    hot_memories = storage.get_hot_memories()

    if not hot_memories:
        return "[MEMORY: Hot cache empty - no frequently-accessed patterns yet]"

    lines = ["[MEMORY: Hot Cache - High-confidence patterns]"]
    for m in hot_memories:
        tags_str = f" [{', '.join(m.tags)}]" if m.tags else ""
        lines.append(f"- {m.content}{tags_str}")

    return "\n".join(lines)


# ========== Mining Tools ==========


@mcp.tool
def log_output(
    content: Annotated[str, Field(description="Output content to log for pattern mining")],
) -> dict:
    """Log an output for pattern mining. Called automatically or manually."""
    if not settings.mining_enabled:
        return error_response("Mining is disabled")

    # Validate content length
    if len(content) > settings.max_content_length:
        return error_response(
            f"Content too long ({len(content)} chars). Max: {settings.max_content_length}"
        )

    log_id = storage.log_output(content)
    return success_response("Output logged", log_id=log_id)


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

    memory_id = storage.store_memory(
        content=candidate.pattern,
        memory_type=mem_type,
        source=MemorySource.MINED,
        tags=tags or [],
    )

    storage.promote_to_hot(memory_id)
    storage.delete_mined_pattern(pattern_id)

    return success_response(
        f"Pattern approved as memory #{memory_id} and promoted to hot cache",
        memory_id=memory_id,
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
def run_mining(
    hours: Annotated[int, Field(description="Hours of logs to process")] = 24,
) -> dict:
    """Run pattern mining on recent output logs.

    Extracts patterns (imports, facts, commands, code) from logged outputs
    and updates the mined_patterns table with occurrence counts.
    """
    log.info("run_mining() called: hours={}", hours)
    if not settings.mining_enabled:
        return error_response("Mining is disabled")

    from memory_mcp.mining import run_mining as _run_mining

    result = _run_mining(storage, hours=hours)
    log.info("Mining complete: {} outputs processed, {} patterns found",
             result["outputs_processed"], result["patterns_found"])
    return {"success": True, **result}


# ========== Entry Point ==========


def main():
    """Run the MCP server."""
    log.info("Starting Memory MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
