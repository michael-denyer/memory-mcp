"""FastMCP server with memory tools and hot cache resource."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from memory_mcp.config import find_bootstrap_files, get_settings
from memory_mcp.helpers import (
    build_ranking_factors as _build_ranking_factors,
)
from memory_mcp.helpers import (
    format_memories_for_llm as _format_memories_for_llm,
)
from memory_mcp.helpers import (
    get_promotion_suggestions as _get_promotion_suggestions,
)
from memory_mcp.helpers import (
    get_similarity_confidence as _get_similarity_confidence,
)
from memory_mcp.helpers import (
    invalid_memory_type_error,
    parse_memory_type,
)
from memory_mcp.logging import (
    configure_logging,
    get_logger,
    metrics,
    record_hot_cache_change,
    record_mining,
    record_recall,
    record_store,
    update_hot_cache_stats,
)
from memory_mcp.models import Memory
from memory_mcp.responses import (
    AccessPatternResponse,
    AuditEntryResponse,
    AuditHistoryResponse,
    BootstrapResponse,
    ContradictionPairResponse,
    ContradictionResponse,
    CrossSessionPatternResponse,
    HotCacheEffectivenessResponse,
    HotCacheMetricsResponse,
    HotCacheResponse,
    MaintenanceResponse,
    MemoryResponse,
    PredictionResponse,
    RecallResponse,
    RelatedMemoryResponse,
    RelationshipResponse,
    RelationshipStatsResponse,
    SeedResult,
    SessionResponse,
    StatsResponse,
    TrustHistoryEntry,
    TrustHistoryResponse,
    TrustResponse,
    VectorRebuildResponse,
    error_response,
    memory_to_response,
    relation_to_response,
    session_to_response,
    success_response,
)
from memory_mcp.storage import (
    AuditOperation,
    MemorySource,
    MemoryType,
    RecallMode,
    RelationType,
    Storage,
    TrustReason,
)
from memory_mcp.text_parsing import parse_content_into_chunks

log = get_logger("server")

# Initialize
settings = get_settings()
configure_logging(level=settings.log_level, log_format=settings.log_format)
storage = Storage(settings)
mcp = FastMCP("memory-mcp")

log.info("Memory MCP server initialized")


# ========== Helper Function Wrappers ==========
# These wrap pure functions from helpers.py with module-level settings/storage


def build_ranking_factors(mode: RecallMode | None, prefix: str = "") -> str:
    """Build ranking factors string using module storage."""
    mode_config = storage.get_recall_mode_config(mode or RecallMode.BALANCED)
    mode_name = mode.value if mode else "balanced"
    return _build_ranking_factors(
        mode_name,
        mode_config.similarity_weight,
        mode_config.recency_weight,
        mode_config.access_weight,
        prefix,
    )


def get_promotion_suggestions(memories: list[Memory], max_suggestions: int = 2) -> list[dict]:
    """Get promotion suggestions using module settings."""
    return _get_promotion_suggestions(memories, settings.promotion_threshold, max_suggestions)


def format_memories_for_llm(memories: list[Memory]):
    """Format memories for LLM using module settings."""
    return _format_memories_for_llm(
        memories,
        settings.high_confidence_threshold,
        settings.default_confidence_threshold,
    )


def get_similarity_confidence(similarity: float | None) -> str:
    """Map similarity score to confidence label using module settings."""
    return _get_similarity_confidence(
        similarity,
        settings.high_confidence_threshold,
        settings.default_confidence_threshold,
    )


# ========== Cold Storage Tools ==========


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
    session_id: Annotated[
        str | None, Field(description="Session ID for conversation provenance tracking")
    ] = None,
) -> dict:
    """Store a new memory. Returns the memory ID."""
    log.debug("remember() called: type={} tags={} session={}", memory_type, tags, session_id)

    # Validate content not empty
    if not content or not content.strip():
        return error_response("Content cannot be empty")

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
        session_id=session_id,
    )

    # Record metrics (merged=False since we can't detect semantic merges at this level)
    record_store(memory_type=mem_type.value, merged=not is_new, contradictions=0)

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
    include_related: Annotated[
        bool,
        Field(description="Include related memories from knowledge graph for top results"),
    ] = False,
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

    # Record metrics
    hot_hit = any(m.is_hot for m in result.memories)
    effective_threshold = (
        threshold if threshold is not None else settings.default_confidence_threshold
    )
    record_recall(
        query_length=len(query),
        results_count=len(result.memories),
        gated_count=result.gated_count,
        hot_hit=hot_hit,
        threshold=effective_threshold,
    )

    # Record retrieval events for quality tracking (RAG-inspired)
    if result.memories:
        memory_ids = [m.id for m in result.memories]
        similarities = [m.similarity or 0.0 for m in result.memories]
        storage.record_retrieval_event(query, memory_ids, similarities)

    # Generate promotion suggestions for frequently-accessed cold memories
    suggestions = get_promotion_suggestions(result.memories) if result.memories else None

    # Fetch related memories if requested (for top 3 results)
    related_list: list[RelatedMemoryResponse] | None = None
    if include_related and result.memories:
        related_list = []
        seen_ids: set[int] = {m.id for m in result.memories}  # Avoid duplicates
        for memory in result.memories[:3]:  # Only top 3 to limit response size
            related = storage.get_related(memory.id)
            for rel_memory, relation in related:
                if rel_memory.id not in seen_ids:
                    seen_ids.add(rel_memory.id)
                    related_list.append(
                        RelatedMemoryResponse(
                            memory=memory_to_response(rel_memory),
                            relationship=relation_to_response(relation),
                        )
                    )

    # Format memories for LLM-friendly consumption
    formatted_context, context_summary = format_memories_for_llm(result.memories)

    return RecallResponse(
        memories=[memory_to_response(m) for m in result.memories],
        confidence=result.confidence,
        gated_count=result.gated_count,
        mode=result.mode.value if result.mode else "balanced",
        guidance=result.guidance or "",
        ranking_factors=build_ranking_factors(result.mode),
        formatted_context=formatted_context if formatted_context else None,
        context_summary=context_summary,
        promotion_suggestions=suggestions if suggestions else None,
        related_memories=related_list if related_list else None,
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

    # Generate promotion suggestions for frequently-accessed cold memories
    suggestions = get_promotion_suggestions(result.memories) if result.memories else None

    # Format memories for LLM-friendly consumption
    formatted_context, context_summary = format_memories_for_llm(result.memories)

    return RecallResponse(
        memories=[memory_to_response(m) for m in result.memories],
        confidence=result.confidence,
        gated_count=result.gated_count,
        mode=result.mode.value if result.mode else "balanced",
        guidance=result.guidance or "",
        ranking_factors=build_ranking_factors(result.mode, prefix="Fallback search"),
        formatted_context=formatted_context if formatted_context else None,
        context_summary=context_summary,
        promotion_suggestions=suggestions if suggestions else None,
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
    - effectiveness: Value metrics (hit rate, tool calls saved, most/least used)
    - avg_hot_score: Average hot score of items (for LRU ranking)
    """
    stats = storage.get_hot_cache_stats()
    hot_memories = storage.get_hot_memories()
    cache_metrics = storage.get_hot_cache_metrics()

    # Compute effectiveness metrics
    total_accesses = sum(m.access_count for m in hot_memories)
    total_reads = cache_metrics.hits + cache_metrics.misses
    hit_rate = (cache_metrics.hits / total_reads * 100) if total_reads > 0 else 0.0

    # Most and least accessed items (for feedback)
    most_accessed = max(hot_memories, key=lambda m: m.access_count) if hot_memories else None
    # Least accessed non-pinned item (candidate for demotion)
    unpinned = [m for m in hot_memories if not m.is_pinned]
    least_accessed = min(unpinned, key=lambda m: m.access_count) if unpinned else None

    return HotCacheResponse(
        items=[memory_to_response(m) for m in hot_memories],
        max_items=stats["max_items"],
        current_count=stats["current_count"],
        pinned_count=stats["pinned_count"],
        avg_hot_score=stats["avg_hot_score"],
        metrics=HotCacheMetricsResponse(
            hits=cache_metrics.hits,
            misses=cache_metrics.misses,
            evictions=cache_metrics.evictions,
            promotions=cache_metrics.promotions,
        ),
        effectiveness=HotCacheEffectivenessResponse(
            total_accesses=total_accesses,
            estimated_tool_calls_saved=cache_metrics.hits,  # Each hit = 1 recall tool call saved
            hit_rate_percent=round(hit_rate, 1),
            most_accessed_id=most_accessed.id if most_accessed else None,
            least_accessed_id=least_accessed.id if least_accessed else None,
        ),
    )


@mcp.tool
def metrics_status() -> dict:
    """Get observability metrics for monitoring and debugging.

    Returns counters and gauges for key operations:
    - recall: queries, results returned/gated, hot hits, empty results
    - store: total stores, by type, merges, contradictions
    - mining: runs, patterns found/new/updated
    - hot_cache: promotions, demotions, evictions, utilization

    Useful for debugging performance issues, monitoring usage patterns,
    and understanding system behavior.
    """
    # Update hot cache gauges before returning
    stats = storage.get_hot_cache_stats()
    update_hot_cache_stats(
        size=stats["current_count"],
        max_size=stats["max_items"],
        pinned=stats["pinned_count"],
    )
    return {"success": True, **metrics.snapshot()}


@mcp.tool
def promote(
    memory_id: Annotated[int, Field(description="ID of memory to promote to hot cache")],
) -> dict:
    """Manually promote a memory to hot cache for zero-latency access."""
    if storage.promote_to_hot(memory_id):
        record_hot_cache_change(promoted=True)
        return success_response(f"Memory #{memory_id} promoted to hot cache")
    return error_response(f"Failed to promote memory #{memory_id}")


@mcp.tool
def demote(
    memory_id: Annotated[int, Field(description="ID of memory to remove from hot cache")],
) -> dict:
    """Remove a memory from hot cache (keeps in cold storage)."""
    if storage.demote_from_hot(memory_id):
        record_hot_cache_change(demoted=True)
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

# Track whether we've attempted auto-bootstrap this session (per working directory)
_auto_bootstrap_attempted: set[str] = set()


def _try_auto_bootstrap() -> bool:
    """Attempt to auto-bootstrap from current directory if hot cache is empty.

    Returns True if bootstrap was attempted and created memories.
    Only runs once per working directory per session.
    """
    import os
    from pathlib import Path

    cwd = os.getcwd()

    # Only attempt once per directory per session
    if cwd in _auto_bootstrap_attempted:
        return False
    _auto_bootstrap_attempted.add(cwd)

    file_paths = find_bootstrap_files(Path(cwd))
    if not file_paths:
        log.debug("Auto-bootstrap: no documentation files found in {}", cwd)
        return False

    log.info("Auto-bootstrap: found {} documentation files in {}", len(file_paths), cwd)

    result = storage.bootstrap_from_files(
        file_paths=file_paths,
        memory_type=MemoryType.PROJECT,
        promote_to_hot=True,
        tags=["auto-bootstrap"],
    )

    if result["memories_created"] > 0:
        log.info(
            "Auto-bootstrap: created {} memories from {} files",
            result["memories_created"],
            result["files_processed"],
        )
        return True

    return False


@mcp.resource("memory://hot-cache")
def hot_cache_resource() -> str:
    """Auto-injectable system context with high-confidence patterns.

    Configure Claude Code to include this resource in system prompts
    for zero-latency access to frequently-used knowledge.

    If the hot cache is empty and documentation files exist in the current
    directory, auto-bootstraps from README.md, CLAUDE.md, etc.

    Records hit/miss metrics for observability (see hot_cache_status).
    """
    hot_memories = storage.get_hot_memories()

    # Auto-bootstrap if hot cache is empty
    if not hot_memories:
        if _try_auto_bootstrap():
            # Re-fetch after bootstrap
            hot_memories = storage.get_hot_memories()

    if not hot_memories:
        storage.record_hot_cache_miss()
        return "[MEMORY: Hot cache empty - no frequently-accessed patterns yet]"

    storage.record_hot_cache_hit()
    lines = ["[MEMORY: Hot Cache - High-confidence patterns]"]
    max_chars = settings.hot_cache_display_max_chars

    for m in hot_memories:
        # Truncate long content for context efficiency
        content = m.content
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        # Limit tags shown (first 3)
        tags_str = f" [{', '.join(m.tags[:3])}]" if m.tags else ""
        lines.append(f"- {content}{tags_str}")

    return "\n".join(lines)


# ========== Mining Tools ==========


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

    log_id = storage.log_output(content, session_id=session_id)
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

    # Record mining metrics
    record_mining(
        patterns_found=result["patterns_found"],
        patterns_new=result["new_patterns"],
        patterns_updated=result["updated_patterns"],
    )

    return {"success": True, **result}


# ========== Seeding Tools ==========


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


def _empty_bootstrap_response(
    message: str,
    errors: list[str] | None = None,
    success: bool = True,
) -> BootstrapResponse:
    """Create a BootstrapResponse for early-exit cases (no files processed)."""
    return BootstrapResponse(
        success=success,
        message=message,
        errors=errors or [],
    )


@mcp.tool
def bootstrap_project(
    root_path: Annotated[
        str,
        Field(description="Project root directory (default: current directory)"),
    ] = ".",
    file_patterns: Annotated[
        list[str] | None,
        Field(
            description=(
                "Specific files to seed. If not provided, auto-detects: "
                "CLAUDE.md, README.md, CONTRIBUTING.md, ARCHITECTURE.md"
            )
        ),
    ] = None,
    promote_to_hot: Annotated[
        bool,
        Field(description="Promote all bootstrapped memories to hot cache"),
    ] = True,
    memory_type: Annotated[
        str,
        Field(description="Memory type for all content"),
    ] = "project",
    tags: Annotated[
        list[str] | None,
        Field(description="Tags to apply to all memories"),
    ] = None,
) -> BootstrapResponse:
    """Bootstrap hot cache from project documentation files.

    Scans for common project documentation files (README.md, CLAUDE.md, etc.),
    parses them into memories, and optionally promotes to hot cache.

    This is ideal for quickly populating the hot cache when starting work
    on a new codebase.

    Edge cases handled gracefully:
    - Empty repo: Returns success with files_found=0 and helpful message
    - No markdown files: Returns success with message
    - File read errors: Logged in errors list, continues with other files
    - Empty files: Skipped silently
    - Binary files: Skipped with warning
    - All content already exists: Returns memories_skipped count
    """
    from pathlib import Path

    root = Path(root_path).expanduser().resolve()

    if not root.exists():
        return _empty_bootstrap_response(
            "Root path does not exist.",
            errors=[f"Root path not found: {root_path}"],
        )

    if not root.is_dir():
        return _empty_bootstrap_response(
            "Root path is not a directory.",
            errors=[f"Not a directory: {root_path}"],
        )

    # Determine files to process
    if file_patterns:
        file_paths = [root / f for f in file_patterns]
    else:
        file_paths = find_bootstrap_files(root)

    if not file_paths:
        return _empty_bootstrap_response(
            "No documentation files found. Create README.md or CLAUDE.md to bootstrap."
        )

    # Validate memory type
    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return _empty_bootstrap_response(
            "Invalid memory type specified.",
            errors=[f"Invalid memory_type. Use: {[t.value for t in MemoryType]}"],
            success=False,
        )

    result = storage.bootstrap_from_files(
        file_paths=file_paths,
        memory_type=mem_type,
        promote_to_hot=promote_to_hot,
        tags=tags,
    )

    return BootstrapResponse(**result)


# ========== Trust Management Tools ==========


STRENGTHENING_REASONS = ["used_correctly", "explicitly_confirmed", "cross_validated"]
WEAKENING_REASONS = [
    "outdated",
    "partially_incorrect",
    "factually_wrong",
    "superseded",
    "low_utility",
]


def _parse_trust_reason(
    reason: str | None, expected_positive: bool
) -> tuple[TrustReason | None, str | None]:
    """Parse and validate a trust reason string.

    Args:
        reason: The reason string to parse
        expected_positive: True for strengthening reasons, False for weakening

    Returns:
        Tuple of (TrustReason or None, error message or None)
    """
    if not reason:
        return None, None

    from memory_mcp.storage import TRUST_REASON_DEFAULTS

    try:
        trust_reason = TrustReason(reason)
        default_delta = TRUST_REASON_DEFAULTS.get(trust_reason, 0)

        if expected_positive and default_delta < 0:
            return None, f"Reason '{reason}' is a weakening reason. Use invalidate_memory instead."
        if not expected_positive and default_delta > 0:
            return (
                None,
                f"Reason '{reason}' is a strengthening reason. Use validate_memory instead.",
            )

        return trust_reason, None
    except ValueError:
        valid = STRENGTHENING_REASONS if expected_positive else WEAKENING_REASONS
        action = "strengthening" if expected_positive else "weakening"
        return None, f"Invalid reason '{reason}'. Valid {action} reasons: {valid}"


@mcp.tool
def validate_memory(
    memory_id: Annotated[int, Field(description="ID of memory to validate")],
    reason: Annotated[
        str | None,
        Field(
            description=(
                "Reason for validation: 'used_correctly' (applied and worked), "
                "'explicitly_confirmed' (user verified), "
                "'cross_validated' (multiple sources agree). "
                "If not specified, uses default boost."
            )
        ),
    ] = None,
    boost: Annotated[
        float | None,
        Field(
            description=(
                "Custom trust boost (overrides reason default). If None, uses reason's default."
            )
        ),
    ] = None,
    note: Annotated[
        str | None,
        Field(description="Optional note explaining the validation context"),
    ] = None,
) -> TrustResponse | dict:
    """Mark a memory as validated/confirmed useful.

    Increases the memory's trust score and records the reason in the trust history.
    Use this when you verify that a recalled memory is still accurate and helpful.

    Reasons and their default boosts:
    - used_correctly: +0.05 (memory was applied successfully)
    - explicitly_confirmed: +0.15 (user explicitly confirmed accuracy)
    - cross_validated: +0.20 (corroborated by multiple sources)

    Trust score is capped at 1.0.
    """
    memory = storage.get_memory(memory_id)
    if not memory:
        return error_response(f"Memory #{memory_id} not found")

    old_trust = memory.trust_score

    trust_reason, error = _parse_trust_reason(reason, expected_positive=True)
    if error:
        return error_response(error)

    if trust_reason:
        new_trust = storage.adjust_trust(memory_id, reason=trust_reason, delta=boost, note=note)
    else:
        new_trust = storage.strengthen_trust(memory_id, boost=boost or 0.1)

    if new_trust is None:
        return error_response(f"Failed to validate memory #{memory_id}")

    reason_msg = f" (reason: {reason})" if reason else ""
    return TrustResponse(
        memory_id=memory_id,
        old_trust=old_trust,
        new_trust=new_trust,
        message=f"Trust increased: {old_trust:.2f} -> {new_trust:.2f}{reason_msg}",
    )


@mcp.tool
def invalidate_memory(
    memory_id: Annotated[int, Field(description="ID of memory found to be incorrect/outdated")],
    reason: Annotated[
        str | None,
        Field(
            description=(
                "Reason for invalidation: 'outdated' (info is stale), "
                "'partially_incorrect' (some details wrong), "
                "'factually_wrong' (fundamentally incorrect), "
                "'superseded' (replaced by newer info), 'low_utility' (not useful). "
                "If not specified, uses default penalty."
            )
        ),
    ] = None,
    penalty: Annotated[
        float | None,
        Field(
            description=(
                "Custom trust penalty (overrides reason default). If None, uses reason's default."
            )
        ),
    ] = None,
    note: Annotated[
        str | None,
        Field(description="Optional note explaining the invalidation context"),
    ] = None,
) -> TrustResponse | dict:
    """Mark a memory as incorrect or outdated.

    Decreases the memory's trust score and records the reason in the trust history.
    Use this when you discover that a recalled memory contains inaccurate or outdated information.

    Reasons and their default penalties:
    - outdated: -0.10 (information is stale but was once correct)
    - partially_incorrect: -0.15 (some details are wrong)
    - factually_wrong: -0.30 (fundamentally incorrect)
    - superseded: -0.05 (replaced by newer information)
    - low_utility: -0.05 (not useful in practice)

    Trust score is floored at 0.0. Memories with very low trust will
    rank lower in recall results due to trust-weighted scoring.
    """
    memory = storage.get_memory(memory_id)
    if not memory:
        return error_response(f"Memory #{memory_id} not found")

    old_trust = memory.trust_score

    trust_reason, error = _parse_trust_reason(reason, expected_positive=False)
    if error:
        return error_response(error)

    if trust_reason:
        new_trust = storage.adjust_trust(memory_id, reason=trust_reason, delta=penalty, note=note)
    else:
        new_trust = storage.weaken_trust(memory_id, penalty=penalty or 0.1)

    if new_trust is None:
        return error_response(f"Failed to invalidate memory #{memory_id}")

    reason_msg = f" (reason: {reason})" if reason else ""
    return TrustResponse(
        memory_id=memory_id,
        old_trust=old_trust,
        new_trust=new_trust,
        message=f"Trust decreased: {old_trust:.2f} -> {new_trust:.2f}{reason_msg}",
    )


@mcp.tool
def get_trust_history(
    memory_id: Annotated[int, Field(description="ID of memory to get trust history for")],
    limit: Annotated[
        int, Field(description="Maximum number of history entries to return (default 20)")
    ] = 20,
) -> TrustHistoryResponse | dict:
    """Get the trust adjustment history for a memory.

    Shows all trust changes with reasons, timestamps, and context.
    Useful for understanding why a memory's trust score evolved.
    """
    memory = storage.get_memory(memory_id)
    if not memory:
        return error_response(f"Memory #{memory_id} not found")

    events = storage.get_trust_history(memory_id, limit=limit)

    entries = [
        TrustHistoryEntry(
            id=e.id,
            memory_id=e.memory_id,
            reason=e.reason.value,
            old_trust=e.old_trust,
            new_trust=e.new_trust,
            delta=e.delta,
            similarity=e.similarity,
            note=e.note,
            created_at=e.created_at.isoformat(),
        )
        for e in events
    ]

    return TrustHistoryResponse(
        memory_id=memory_id,
        entries=entries,
        current_trust=memory.trust_score,
        total_changes=len(entries),
    )


# ========== Maintenance Tools ==========


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


# ========== Database Maintenance Tools ==========


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


# ========== Memory Relationships (Knowledge Graph) Tools ==========


def parse_relation_type(relation_type: str) -> RelationType | None:
    """Parse relation type string, returning None if invalid."""
    try:
        return RelationType(relation_type)
    except ValueError:
        return None


def invalid_relation_type_error() -> dict:
    """Return error for invalid relation type."""
    return error_response(f"Invalid relation_type. Use: {[t.value for t in RelationType]}")


@mcp.tool
def link_memories(
    from_memory_id: Annotated[int, Field(description="Source memory ID")],
    to_memory_id: Annotated[int, Field(description="Target memory ID")],
    relation_type: Annotated[
        str,
        Field(
            description=(
                "Relationship type: 'relates_to' (general), 'depends_on' (prerequisite), "
                "'supersedes' (replaces), 'refines' (more specific), "
                "'contradicts' (conflict), 'elaborates' (more detail)"
            )
        ),
    ],
) -> RelationshipResponse | dict:
    """Create a typed relationship between two memories.

    Use this to build a knowledge graph connecting related concepts.
    Relationships are directional: from_memory -[relation_type]-> to_memory.

    Examples:
    - "Python 3.12 features" -[supersedes]-> "Python 3.11 features"
    - "Auth implementation" -[depends_on]-> "Database schema"
    - "API endpoint details" -[elaborates]-> "API overview"
    """
    rel_type = parse_relation_type(relation_type)
    if rel_type is None:
        return invalid_relation_type_error()

    relation = storage.link_memories(from_memory_id, to_memory_id, rel_type)
    if relation is None:
        return error_response(
            f"Failed to link memories #{from_memory_id} -> #{to_memory_id}. "
            "Check that both memories exist and aren't already linked with this type."
        )

    return relation_to_response(relation)


@mcp.tool
def unlink_memories(
    from_memory_id: Annotated[int, Field(description="Source memory ID")],
    to_memory_id: Annotated[int, Field(description="Target memory ID")],
    relation_type: Annotated[
        str | None,
        Field(description="Specific relation type to remove, or None to remove all"),
    ] = None,
) -> dict:
    """Remove relationship(s) between two memories.

    If relation_type is specified, only removes that specific relationship.
    If not specified, removes all relationships between the two memories.
    """
    rel_type = None
    if relation_type:
        rel_type = parse_relation_type(relation_type)
        if rel_type is None:
            return invalid_relation_type_error()

    count = storage.unlink_memories(from_memory_id, to_memory_id, rel_type)
    if count == 0:
        return error_response(
            f"No relationships found between #{from_memory_id} and #{to_memory_id}"
        )

    return success_response(
        f"Removed {count} relationship(s) between #{from_memory_id} and #{to_memory_id}",
        removed_count=count,
    )


@mcp.tool
def get_related_memories(
    memory_id: Annotated[int, Field(description="Memory ID to find relationships for")],
    relation_type: Annotated[
        str | None,
        Field(description="Filter by relation type"),
    ] = None,
    direction: Annotated[
        str,
        Field(
            description=(
                "Direction: 'outgoing' (from this memory), "
                "'incoming' (to this memory), or 'both' (default)"
            )
        ),
    ] = "both",
) -> list[RelatedMemoryResponse] | dict:
    """Get memories related to a given memory.

    Returns related memories along with their relationship information.
    Use this to explore the knowledge graph around a specific concept.
    """
    if direction not in ("outgoing", "incoming", "both"):
        return error_response("Invalid direction. Use: 'outgoing', 'incoming', or 'both'")

    rel_type = None
    if relation_type:
        rel_type = parse_relation_type(relation_type)
        if rel_type is None:
            return invalid_relation_type_error()

    related = storage.get_related(memory_id, rel_type, direction)

    return [
        RelatedMemoryResponse(
            memory=memory_to_response(memory),
            relationship=relation_to_response(relation),
        )
        for memory, relation in related
    ]


@mcp.tool
def relationship_stats() -> RelationshipStatsResponse:
    """Get statistics about memory relationships in the knowledge graph."""
    stats = storage.get_relationship_stats()
    return RelationshipStatsResponse(**stats)


# ========== Contradiction Detection Tools ==========


@mcp.tool
def find_contradictions(
    memory_id: Annotated[int, Field(description="Memory ID to check for contradictions")],
    similarity_threshold: Annotated[
        float,
        Field(description="Minimum similarity to consider (0.75 = same topic)"),
    ] = 0.75,
    limit: Annotated[int, Field(description="Maximum contradictions to return")] = 5,
) -> list[ContradictionResponse]:
    """Find memories that may contradict a given memory.

    Searches for semantically similar memories that could contain
    conflicting information. High similarity means same topic area,
    which is where contradictions are likely.

    Use this after storing a new memory or when validating existing ones.
    """
    contradictions = storage.find_contradictions(
        memory_id=memory_id,
        similarity_threshold=similarity_threshold,
        limit=limit,
    )
    return [
        ContradictionResponse(
            memory_a=memory_to_response(c.memory_a),
            memory_b=memory_to_response(c.memory_b),
            similarity=c.similarity,
            already_linked=c.already_linked,
        )
        for c in contradictions
    ]


@mcp.tool
def get_contradictions() -> list[ContradictionPairResponse]:
    """Get all memory pairs marked as contradictions.

    Returns pairs that have been flagged as containing conflicting
    information. Use resolve_contradiction to handle these.
    """
    contradictions = storage.get_all_contradictions()
    return [
        ContradictionPairResponse(
            memory_a=memory_to_response(m1),
            memory_b=memory_to_response(m2),
            relationship=relation_to_response(rel),
        )
        for m1, m2, rel in contradictions
    ]


@mcp.tool
def mark_contradiction(
    memory_id_a: Annotated[int, Field(description="First memory ID")],
    memory_id_b: Annotated[int, Field(description="Second memory ID")],
) -> RelationshipResponse | dict:
    """Mark two memories as contradicting each other.

    Creates a CONTRADICTS relationship. Use this when you discover
    that two memories contain conflicting information about the same topic.
    """
    relation = storage.mark_contradiction(memory_id_a, memory_id_b)
    if relation is None:
        return error_response(
            f"Failed to mark contradiction between #{memory_id_a} and #{memory_id_b}. "
            "Check that both memories exist and aren't already marked as contradicting."
        )
    return relation_to_response(relation)


@mcp.tool
def resolve_contradiction(
    memory_id_a: Annotated[int, Field(description="First memory in contradiction")],
    memory_id_b: Annotated[int, Field(description="Second memory in contradiction")],
    keep_id: Annotated[int, Field(description="ID of memory to keep (must be one of the two)")],
    resolution: Annotated[
        str,
        Field(
            description=(
                "How to handle the discarded memory: "
                "'supersedes' (kept memory replaces other), "
                "'delete' (remove the other memory), "
                "'weaken' (reduce trust in other memory)"
            )
        ),
    ] = "supersedes",
) -> dict:
    """Resolve a contradiction by keeping one memory and handling the other.

    After resolving:
    - 'supersedes': Creates SUPERSEDES relationship, weakens trust in discarded
    - 'delete': Removes the discarded memory entirely
    - 'weaken': Reduces trust in discarded memory but keeps it
    """
    if resolution not in ("supersedes", "delete", "weaken"):
        return error_response(
            f"Invalid resolution '{resolution}'. Use: supersedes, delete, or weaken"
        )

    success = storage.resolve_contradiction(memory_id_a, memory_id_b, keep_id, resolution)
    if not success:
        return error_response(
            "Failed to resolve contradiction. Ensure keep_id is one of the two memories."
        )

    other_id = memory_id_b if keep_id == memory_id_a else memory_id_a
    return success_response(f"Resolved contradiction: kept #{keep_id}, {resolution} #{other_id}")


# ========== Session (Conversation Provenance) Tools ==========


@mcp.tool
def get_sessions(
    limit: Annotated[int, Field(description="Maximum sessions to return")] = 20,
    project_path: Annotated[
        str | None, Field(description="Filter to sessions from this project path")
    ] = None,
) -> list[SessionResponse]:
    """Get recent conversation sessions.

    Sessions track which conversations memories originated from.
    Use this to see conversation history and navigate to specific sessions.
    """
    sessions = storage.get_sessions(limit=limit, project_path=project_path)
    return [session_to_response(s) for s in sessions]


@mcp.tool
def get_session(
    session_id: Annotated[str, Field(description="Session ID to retrieve")],
) -> SessionResponse | dict:
    """Get details for a specific session."""
    session = storage.get_session(session_id)
    if session is None:
        return error_response(f"Session not found: {session_id}")
    return session_to_response(session)


@mcp.tool
def get_session_memories(
    session_id: Annotated[str, Field(description="Session ID to get memories from")],
    limit: Annotated[int, Field(description="Maximum memories to return")] = 100,
) -> list[MemoryResponse] | dict:
    """Get all memories from a specific conversation session.

    Use this to explore what was learned during a particular conversation.
    """
    session = storage.get_session(session_id)
    if session is None:
        return error_response(f"Session not found: {session_id}")

    memories = storage.get_session_memories(session_id, limit=limit)
    return [memory_to_response(m) for m in memories]


@mcp.tool
def cross_session_patterns(
    min_sessions: Annotated[
        int, Field(description="Minimum sessions a pattern must appear in")
    ] = 2,
) -> list[CrossSessionPatternResponse]:
    """Find content patterns appearing across multiple conversation sessions.

    Useful for identifying frequently-discussed topics that might warrant
    promotion to hot cache. Patterns appearing in many sessions are likely
    important project knowledge.

    Returns patterns sorted by session count and total accesses.
    """
    patterns = storage.get_cross_session_patterns(min_sessions=min_sessions)
    return [CrossSessionPatternResponse(**p) for p in patterns]


@mcp.tool
def set_session_topic(
    session_id: Annotated[str, Field(description="Session ID to update")],
    topic: Annotated[str, Field(description="Topic description for the session")],
) -> dict:
    """Set or update the topic for a conversation session.

    Topics help identify what conversations were about when reviewing
    session history. Can be auto-detected or manually set.
    """
    if storage.update_session_topic(session_id, topic):
        return success_response(f"Updated topic for session {session_id}", topic=topic)
    return error_response(f"Session not found: {session_id}")


# ========== Predictive Hot Cache Tools ==========


@mcp.tool
def access_patterns(
    memory_id: Annotated[int | None, Field(description="Memory ID to get patterns for")] = None,
    min_count: Annotated[int, Field(description="Minimum access count to include")] = 2,
    limit: Annotated[int, Field(description="Maximum patterns to return")] = 20,
) -> list[AccessPatternResponse]:
    """Get learned access patterns for predictive caching.

    When memory_id is provided, shows patterns from that specific memory.
    Otherwise, shows all learned patterns across all memories.

    Requires MEMORY_MCP_PREDICTIVE_CACHE_ENABLED=true.
    """
    if memory_id is not None:
        patterns = storage.get_access_patterns(memory_id, limit=limit)
    else:
        patterns = storage.get_all_access_patterns(min_count=min_count, limit=limit)

    return [
        AccessPatternResponse(
            from_memory_id=p.from_memory_id,
            to_memory_id=p.to_memory_id,
            count=p.count,
            probability=p.probability,
            last_seen=p.last_seen.isoformat(),
        )
        for p in patterns
    ]


@mcp.tool
def predict_next(
    memory_id: Annotated[int, Field(description="Memory ID to predict from")],
    threshold: Annotated[float | None, Field(description="Minimum probability threshold")] = None,
    limit: Annotated[int | None, Field(description="Maximum predictions")] = None,
) -> list[PredictionResponse]:
    """Predict which memories might be needed next based on access patterns.

    Uses learned Markov chain of access sequences to predict what
    memories typically follow after accessing the given memory.

    Requires MEMORY_MCP_PREDICTIVE_CACHE_ENABLED=true.
    """
    predictions = storage.predict_next_memories(
        memory_id=memory_id,
        threshold=threshold,
        limit=limit,
    )
    return [
        PredictionResponse(
            memory=memory_to_response(p.memory),
            probability=p.probability,
            source_memory_id=p.source_memory_id,
        )
        for p in predictions
    ]


@mcp.tool
def warm_cache(
    memory_id: Annotated[int, Field(description="Memory ID to predict from")],
) -> dict:
    """Pre-warm hot cache with predicted next memories.

    Promotes predicted memories to hot cache for zero-latency access.
    Only promotes memories that aren't already in hot cache.

    Requires MEMORY_MCP_PREDICTIVE_CACHE_ENABLED=true.
    """
    if not settings.predictive_cache_enabled:
        return error_response(
            "Predictive cache is disabled. Set MEMORY_MCP_PREDICTIVE_CACHE_ENABLED=true to enable."
        )

    promoted_ids = storage.warm_predicted_cache(memory_id)
    if promoted_ids:
        return success_response(
            f"Pre-warmed {len(promoted_ids)} memories",
            promoted_ids=promoted_ids,
        )
    return success_response("No memories needed warming (already hot or no predictions)")


@mcp.tool
def predictive_cache_status() -> dict:
    """Get status of the predictive hot cache system.

    Shows whether predictive caching is enabled, configuration,
    and learned pattern statistics.
    """
    patterns = storage.get_all_access_patterns(min_count=1, limit=1000)
    unique_sources = len({p.from_memory_id for p in patterns})
    total_transitions = sum(p.count for p in patterns)

    return {
        "enabled": settings.predictive_cache_enabled,
        "config": {
            "prediction_threshold": settings.prediction_threshold,
            "max_predictions": settings.max_predictions,
            "sequence_decay_days": settings.sequence_decay_days,
        },
        "stats": {
            "total_patterns": len(patterns),
            "unique_source_memories": unique_sources,
            "total_transitions": total_transitions,
        },
    }


# ========== Retrieval Quality Tracking (RAG-inspired) ==========


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


# ========== Memory Consolidation (MemoryBank-inspired) ==========


def _parse_memory_type_for_consolidation(
    memory_type: str | None,
) -> tuple[MemoryType | None, dict | None]:
    """Parse memory type string for consolidation tools.

    Returns:
        Tuple of (parsed MemoryType or None, error response dict or None)
    """
    if not memory_type:
        return None, None
    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return None, error_response(f"Invalid memory_type. Use: {[t.value for t in MemoryType]}")
    return mem_type, None


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
    mem_type, err = _parse_memory_type_for_consolidation(memory_type)
    if err:
        return err

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
    mem_type, err = _parse_memory_type_for_consolidation(memory_type)
    if err:
        return err

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


# ========== Entry Point ==========


def main():
    """Run the MCP server."""
    log.info("Starting Memory MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
