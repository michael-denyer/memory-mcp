"""Cold storage tools: remember, recall, forget, list_memories, memory_stats."""

from typing import Annotated

from pydantic import Field

from memory_mcp.helpers import invalid_memory_type_error, parse_memory_type
from memory_mcp.logging import record_recall, record_store
from memory_mcp.models import RecallResult
from memory_mcp.responses import (
    MemoryResponse,
    RecallResponse,
    RelatedMemoryResponse,
    StatsResponse,
    error_response,
    memory_to_response,
    relation_to_response,
    success_response,
)
from memory_mcp.server.app import (
    build_ranking_factors,
    format_memories_for_llm,
    get_auto_project_id,
    get_current_session_id,
    get_promotion_suggestions,
    log,
    mcp,
    settings,
    storage,
)
from memory_mcp.storage import MemorySource, MemoryType, RecallMode


def parse_recall_mode(mode: str | None) -> RecallMode | None:
    """Parse recall mode string, returning None if invalid."""
    if mode is None:
        return None
    try:
        return RecallMode(mode)
    except ValueError:
        return None


def build_recall_response(
    result: RecallResult,
    ranking_prefix: str = "",
    related_memories: list[RelatedMemoryResponse] | None = None,
) -> RecallResponse:
    """Build a RecallResponse from a RecallResult with common formatting.

    Args:
        result: The RecallResult from storage.recall()
        ranking_prefix: Optional prefix for ranking factors string
        related_memories: Optional list of related memories to include

    Returns:
        Formatted RecallResponse ready for the client.
    """
    suggestions = get_promotion_suggestions(result.memories) if result.memories else None
    formatted_context, context_summary = format_memories_for_llm(result.memories)

    return RecallResponse(
        memories=[memory_to_response(m) for m in result.memories],
        confidence=result.confidence,
        gated_count=result.gated_count,
        mode=result.mode.value if result.mode else "balanced",
        guidance=result.guidance or "",
        ranking_factors=build_ranking_factors(result.mode, prefix=ranking_prefix),
        formatted_context=formatted_context if formatted_context else None,
        context_summary=context_summary,
        promotion_suggestions=suggestions if suggestions else None,
        related_memories=related_memories if related_memories else None,
    )


@mcp.tool
def remember(
    content: Annotated[str, Field(description="The content to remember")],
    memory_type: Annotated[
        str,
        Field(
            description=(
                "Type: 'project' (project facts), 'pattern' (code patterns), "
                "'reference' (docs), 'conversation' (discussion facts), "
                "'episodic' (session-bound short-term context)"
            )
        ),
    ] = "project",
    tags: Annotated[list[str] | None, Field(description="Tags for categorization")] = None,
    category: Annotated[
        str | None,
        Field(
            description=(
                "Subcategory within type (e.g., 'decision', 'architecture', 'import', "
                "'command', 'api', 'config'). Helps organize memories."
            )
        ),
    ] = None,
    session_id: Annotated[
        str | None, Field(description="Session ID for conversation provenance tracking")
    ] = None,
) -> dict:
    """Store a new memory. Returns the memory ID."""
    log.debug(
        "remember() called: type={} category={} tags={} session={}",
        memory_type,
        category,
        tags,
        session_id,
    )

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

    # Auto-detect current project for project-aware memory
    project_id = get_auto_project_id()

    # Use auto-session if not explicitly provided (zero-config session tracking)
    log.debug(
        "remember() session handling: input session_id={!r} (type={})",
        session_id,
        type(session_id).__name__,
    )
    effective_session_id = session_id or get_current_session_id()
    log.debug("remember() effective_session_id={}", effective_session_id)

    memory_id, is_new = storage.store_memory(
        content=content,
        memory_type=mem_type,
        source=MemorySource.MANUAL,
        tags=tag_list,
        session_id=effective_session_id,
        project_id=project_id,
        category=category,
    )

    # Record metrics (merged=False since we can't detect semantic merges at this level)
    record_store(memory_type=mem_type.value, merged=not is_new, contradictions=0)

    if is_new:
        msg = f"Stored as memory #{memory_id}"
        if project_id:
            msg += f" (project: {project_id})"
        return success_response(msg, memory_id=memory_id, project_id=project_id)
    else:
        return success_response(
            f"Memory #{memory_id} already exists (access count incremented, tags merged)",
            memory_id=memory_id,
            was_duplicate=True,
        )


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
    expand_relations: Annotated[
        bool | None,
        Field(
            description=(
                "Expand results via knowledge graph (Engram-style associative recall). "
                "Related memories are added with decayed scores. None uses config default."
            )
        ),
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

    # Get current project for project-aware recall (if enabled)
    project_id = get_auto_project_id() if settings.project_filter_recall else None

    log.info(
        "recall() called: query='{}' mode={} limit={} threshold={} project={}",
        query[:50],
        mode,
        limit,
        threshold,
        project_id,
    )

    import time

    start_time = time.perf_counter()
    result = storage.recall(
        query=query,
        limit=limit,
        threshold=threshold,
        mode=recall_mode,
        memory_types=memory_types,
        expand_relations=expand_relations,
        project_id=project_id,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Record metrics with enhanced context
    hot_hit = any(m.is_hot for m in result.memories)
    effective_threshold = (
        threshold if threshold is not None else settings.default_confidence_threshold
    )
    effective_mode = result.mode.value if result.mode else "balanced"
    record_recall(
        query_length=len(query),
        results_count=len(result.memories),
        gated_count=result.gated_count,
        hot_hit=hot_hit,
        threshold=effective_threshold,
        query_preview=query,
        mode=effective_mode,
        elapsed_ms=elapsed_ms,
    )

    # Record retrieval events for quality tracking (RAG-inspired)
    if result.memories:
        memory_ids = [m.id for m in result.memories]
        similarities = [m.similarity or 0.0 for m in result.memories]
        storage.record_retrieval_event(query, memory_ids, similarities)

        # Log injection for feedback loop analysis
        session_id = get_current_session_id()
        storage.log_injections_batch(
            memory_ids=memory_ids,
            resource="recall",
            session_id=session_id,
            project_id=project_id,
        )

        # Auto-mark as used if enabled (assumes recalled = used)
        if settings.retrieval_auto_mark_used:
            for memory_id in memory_ids:
                storage.mark_retrieval_used(memory_id, feedback="auto")

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

    return build_recall_response(result, related_memories=related_list)


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

    return build_recall_response(result, ranking_prefix="Fallback search")


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
    # Validate pagination parameters
    if offset < 0:
        return error_response("offset must be >= 0")
    if limit < 1:
        return error_response("limit must be >= 1")

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
