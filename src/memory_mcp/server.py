"""FastMCP server with memory tools and hot cache resource."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from memory_mcp.config import get_settings
from memory_mcp.logging import get_logger
from memory_mcp.storage import Memory, MemorySource, MemoryType, RecallMode, Storage
from memory_mcp.text_parsing import parse_content_into_chunks

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
    is_pinned: bool
    promotion_source: str | None
    tags: list[str]
    access_count: int
    # Trust and provenance
    trust_score: float
    source_log_id: int | None = None
    extracted_at: str | None = None
    # Computed scores
    similarity: float | None = None
    hot_score: float | None = None
    # Recall scoring (populated during recall)
    recency_score: float | None = None
    trust_score_decayed: float | None = None
    composite_score: float | None = None
    created_at: str


class RecallResponse(BaseModel):
    """Response for recall operation."""

    memories: list[MemoryResponse]
    confidence: str
    gated_count: int
    mode: str
    guidance: str
    # Scoring explanation
    ranking_factors: str


class StatsResponse(BaseModel):
    """Response for stats operation."""

    total_memories: int
    hot_cache_count: int
    by_type: dict[str, int]
    by_source: dict[str, int]


class HotCacheMetricsResponse(BaseModel):
    """Metrics for hot cache observability."""

    hits: int
    misses: int
    evictions: int
    promotions: int


class HotCacheResponse(BaseModel):
    """Response for hot cache status."""

    items: list[MemoryResponse]
    max_items: int
    current_count: int
    pinned_count: int
    avg_hot_score: float
    metrics: HotCacheMetricsResponse


def memory_to_response(m: Memory) -> MemoryResponse:
    """Convert Memory to response model."""
    return MemoryResponse(
        id=m.id,
        content=m.content,
        memory_type=m.memory_type.value,
        source=m.source.value,
        is_hot=m.is_hot,
        is_pinned=m.is_pinned,
        promotion_source=m.promotion_source.value if m.promotion_source else None,
        tags=m.tags,
        access_count=m.access_count,
        trust_score=m.trust_score,
        source_log_id=m.source_log_id,
        extracted_at=m.extracted_at.isoformat() if m.extracted_at else None,
        similarity=m.similarity,
        hot_score=m.hot_score,
        recency_score=m.recency_score,
        trust_score_decayed=m.trust_score_decayed,
        composite_score=m.composite_score,
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


def build_ranking_factors(mode: RecallMode | None, prefix: str = "") -> str:
    """Build ranking factors explanation string for recall responses."""
    mode_config = storage.get_recall_mode_config(mode or RecallMode.BALANCED)
    sim_w = int(mode_config.similarity_weight * 100)
    rec_w = int(mode_config.recency_weight * 100)
    acc_w = int(mode_config.access_weight * 100)
    mode_name = mode.value if mode else "balanced"
    base = (
        f"Mode: {mode_name} | "
        f"Ranked by: similarity ({sim_w}%) + recency ({rec_w}%) + access ({acc_w}%)"
    )
    return f"{prefix} | {base}" if prefix else base


@mcp.tool
def remember(
    content: Annotated[str, Field(description="The content to remember")],
    memory_type: Annotated[
        str,
        Field(
            description=(
                "Type: 'project' (project facts), 'pattern' (code patterns), "
                "'reference' (docs), 'conversation' (discussion facts)"
            )
        ),
    ] = "project",
    tags: Annotated[list[str] | None, Field(description="Tags for categorization")] = None,
) -> dict:
    """Store a new memory. Returns the memory ID."""
    log.debug("remember() called: type={} tags={}", memory_type, tags)

    # Validate content length
    if len(content) > settings.max_content_length:
        return error_response(
            f"Content too long ({len(content)} chars). Max: {settings.max_content_length}"
        )

    # Validate tags
    tag_list = tags or []
    if len(tag_list) > settings.max_tags:
        return error_response(f"Too many tags ({len(tag_list)}). Max: {settings.max_tags}")

    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return invalid_memory_type_error()

    memory_id, is_new = storage.store_memory(
        content=content,
        memory_type=mem_type,
        source=MemorySource.MANUAL,
        tags=tag_list,
    )

    if is_new:
        return success_response(f"Stored as memory #{memory_id}", memory_id=memory_id)
    else:
        return success_response(
            f"Memory #{memory_id} already exists (access count incremented, tags merged)",
            memory_id=memory_id,
            was_duplicate=True,
        )


def parse_recall_mode(mode: str | None) -> RecallMode | None:
    """Parse recall mode string, returning None if invalid."""
    if mode is None:
        return None
    try:
        return RecallMode(mode)
    except ValueError:
        return None


@mcp.tool
def recall(
    query: Annotated[str, Field(description="Search query for semantic similarity")],
    mode: Annotated[
        str | None,
        Field(
            description=(
                "Recall mode: 'precision' (high threshold, few results), "
                "'balanced' (default), 'exploratory' (low threshold, more results)"
            )
        ),
    ] = None,
    limit: Annotated[int | None, Field(description="Max results (overrides mode default)")] = None,
    threshold: Annotated[
        float | None, Field(description="Min similarity (overrides mode default)")
    ] = None,
    memory_type: Annotated[
        str | None, Field(description="Filter by type: project, pattern, reference")
    ] = None,
) -> RecallResponse:
    """Semantic search with confidence gating and composite ranking.

    Modes:
    - 'precision': High threshold (0.8), few results (3), prioritizes similarity
    - 'balanced': Default settings, good for general use
    - 'exploratory': Low threshold (0.5), more results (10), diverse ranking

    Returns memories with confidence level and hallucination-prevention guidance.
    """
    # Parse mode
    recall_mode = parse_recall_mode(mode)
    if mode is not None and recall_mode is None:
        valid = [m.value for m in RecallMode]
        return RecallResponse(
            memories=[],
            confidence="low",
            gated_count=0,
            mode="error",
            guidance=f"Invalid mode '{mode}'. Use: {valid}",
            ranking_factors="N/A",
        )

    # Parse memory type filter
    memory_types = None
    if memory_type:
        mem_type = parse_memory_type(memory_type)
        if mem_type is None:
            return RecallResponse(
                memories=[],
                confidence="low",
                gated_count=0,
                mode="error",
                guidance=f"Invalid memory_type. Use: {[t.value for t in MemoryType]}",
                ranking_factors="N/A",
            )
        memory_types = [mem_type]

    # Validate and clamp explicit overrides
    if limit is not None:
        limit = max(1, min(settings.max_recall_limit, limit))
    if threshold is not None:
        threshold = max(0.0, min(1.0, threshold))

    log.debug(
        "recall() called: query='{}' mode={} limit={} threshold={}",
        query[:50],
        mode,
        limit,
        threshold,
    )

    result = storage.recall(
        query=query,
        limit=limit,
        threshold=threshold,
        mode=recall_mode,
        memory_types=memory_types,
    )

    return RecallResponse(
        memories=[memory_to_response(m) for m in result.memories],
        confidence=result.confidence,
        gated_count=result.gated_count,
        mode=result.mode.value if result.mode else "balanced",
        guidance=result.guidance or "",
        ranking_factors=build_ranking_factors(result.mode),
    )


@mcp.tool
def recall_with_fallback(
    query: Annotated[str, Field(description="Search query for semantic similarity")],
    mode: Annotated[
        str | None,
        Field(description="Recall mode: 'precision', 'balanced', 'exploratory'"),
    ] = None,
    min_results: Annotated[
        int, Field(description="Minimum results before trying next fallback")
    ] = 1,
) -> RecallResponse:
    """Recall with automatic fallback through memory types.

    Tries searching in order: patterns -> project facts -> all types.
    Stops when min_results are found with medium+ confidence.

    Use this when you're unsure which memory type contains the answer.
    """
    recall_mode = parse_recall_mode(mode)

    log.debug(
        "recall_with_fallback() called: query='{}' mode={} min={}",
        query[:50],
        mode,
        min_results,
    )

    result = storage.recall_with_fallback(
        query=query,
        mode=recall_mode,
        min_results=min_results,
    )

    return RecallResponse(
        memories=[memory_to_response(m) for m in result.memories],
        confidence=result.confidence,
        gated_count=result.gated_count,
        mode=result.mode.value if result.mode else "balanced",
        guidance=result.guidance or "",
        ranking_factors=build_ranking_factors(result.mode, prefix="Fallback search"),
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
    """Show current hot cache contents, stats, and observability metrics.

    Returns items sorted by hot_score (highest first), along with:
    - metrics.hits: Times hot cache resource was read with content
    - metrics.misses: Times hot cache resource was empty
    - metrics.evictions: Items removed to make space for new ones
    - metrics.promotions: Items added to hot cache
    - avg_hot_score: Average hot score of items (for LRU ranking)
    """
    stats = storage.get_hot_cache_stats()
    hot_memories = storage.get_hot_memories()
    metrics = storage.get_hot_cache_metrics()

    return HotCacheResponse(
        items=[memory_to_response(m) for m in hot_memories],
        max_items=stats["max_items"],
        current_count=stats["current_count"],
        pinned_count=stats["pinned_count"],
        avg_hot_score=stats["avg_hot_score"],
        metrics=HotCacheMetricsResponse(
            hits=metrics.hits,
            misses=metrics.misses,
            evictions=metrics.evictions,
            promotions=metrics.promotions,
        ),
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


@mcp.tool
def pin(
    memory_id: Annotated[int, Field(description="ID of hot memory to pin")],
) -> dict:
    """Pin a hot cache memory to prevent auto-eviction.

    Pinned memories stay in hot cache even when space is needed for new items.
    Only works on memories already in hot cache.
    """
    if storage.pin_memory(memory_id):
        return success_response(f"Memory #{memory_id} pinned (won't be auto-evicted)")
    return error_response(f"Failed to pin memory #{memory_id} (not in hot cache or not found)")


@mcp.tool
def unpin(
    memory_id: Annotated[int, Field(description="ID of memory to unpin")],
) -> dict:
    """Unpin a memory, making it eligible for auto-eviction from hot cache."""
    if storage.unpin_memory(memory_id):
        return success_response(f"Memory #{memory_id} unpinned")
    return error_response(f"Failed to unpin memory #{memory_id}")


# ========== Hot Cache Resource (Auto-Injection) ==========


@mcp.resource("memory://hot-cache")
def hot_cache_resource() -> str:
    """Auto-injectable system context with high-confidence patterns.

    Configure Claude Code to include this resource in system prompts
    for zero-latency access to frequently-used knowledge.

    Records hit/miss metrics for observability (see hot_cache_status).
    """
    hot_memories = storage.get_hot_memories()

    if not hot_memories:
        storage.record_hot_cache_miss()
        return "[MEMORY: Hot cache empty - no frequently-accessed patterns yet]"

    storage.record_hot_cache_hit()
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

    memory_id, is_new = storage.store_memory(
        content=candidate.pattern,
        memory_type=mem_type,
        source=MemorySource.MINED,
        tags=tags or [],
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
    log.info(
        "Mining complete: {} outputs processed, {} patterns found",
        result["outputs_processed"],
        result["patterns_found"],
    )
    return {"success": True, **result}


# ========== Seeding Tools ==========


class SeedResult(BaseModel):
    """Result from seeding operation."""

    memories_created: int
    memories_skipped: int
    errors: list[str]


@mcp.tool
def seed_from_text(
    content: Annotated[str, Field(description="Text content to parse and seed memories from")],
    memory_type: Annotated[
        str, Field(description="Memory type for all extracted items")
    ] = "project",
    promote_to_hot: Annotated[bool, Field(description="Promote all to hot cache")] = False,
) -> SeedResult:
    """Seed memories from text content.

    Parses the content into individual memories (one per paragraph or list item)
    and stores them. Useful for initial setup or bulk import.

    Content is split on:
    - Double newlines (paragraphs)
    - Lines starting with '- ' or '* ' (list items)
    - Lines starting with numbers like '1. ' (numbered lists)
    """
    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return SeedResult(memories_created=0, memories_skipped=0, errors=["Invalid memory_type"])

    chunks = parse_content_into_chunks(content)
    created, skipped, errors = 0, 0, []

    for chunk in chunks:
        if len(chunk) > settings.max_content_length:
            errors.append(f"Chunk too long ({len(chunk)} chars), skipped")
            continue

        memory_id, is_new = storage.store_memory(
            content=chunk,
            memory_type=mem_type,
            source=MemorySource.MANUAL,
        )
        if is_new:
            created += 1
            if promote_to_hot:
                storage.promote_to_hot(memory_id)
        else:
            skipped += 1

    log.info("seed_from_text: created={} skipped={} errors={}", created, skipped, len(errors))
    return SeedResult(memories_created=created, memories_skipped=skipped, errors=errors)


@mcp.tool
def seed_from_file(
    file_path: Annotated[str, Field(description="Path to file to import")],
    memory_type: Annotated[str, Field(description="Memory type for content")] = "project",
    promote_to_hot: Annotated[bool, Field(description="Promote to hot cache")] = False,
) -> SeedResult:
    """Seed memories from a file.

    Reads the file and extracts memories based on content structure.
    Supports markdown files (splits on headers and lists) and plain text.

    Common use: Import from project CLAUDE.md or documentation files.
    """
    from pathlib import Path

    path = Path(file_path).expanduser()
    if not path.exists():
        return SeedResult(
            memories_created=0, memories_skipped=0, errors=[f"File not found: {file_path}"]
        )

    if not path.is_file():
        return SeedResult(
            memories_created=0, memories_skipped=0, errors=[f"Not a file: {file_path}"]
        )

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return SeedResult(memories_created=0, memories_skipped=0, errors=[f"Read error: {e}"])

    return seed_from_text(content=content, memory_type=memory_type, promote_to_hot=promote_to_hot)


# ========== Maintenance Tools ==========


class MaintenanceResponse(BaseModel):
    """Response for maintenance operation."""

    size_before_bytes: int
    size_after_bytes: int
    bytes_reclaimed: int
    memory_count: int
    vector_count: int
    schema_version: int


@mcp.tool
def db_maintenance() -> MaintenanceResponse:
    """Run database maintenance (vacuum and analyze).

    Compacts the database to reclaim unused space and updates
    query planner statistics for better performance.
    """
    log.info("db_maintenance() called")
    result = storage.maintenance()
    log.info(
        "Maintenance complete: {} bytes reclaimed, {} memories",
        result["bytes_reclaimed"],
        result["memory_count"],
    )
    return MaintenanceResponse(**result)


@mcp.tool
def db_info() -> dict:
    """Get database information including schema version and size."""
    import os

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


# ========== Entry Point ==========


def main():
    """Run the MCP server."""
    log.info("Starting Memory MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
