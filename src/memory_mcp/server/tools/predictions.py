"""Predictive hot cache tools: access_patterns, predict_next, warm_cache, status."""

from typing import Annotated

from pydantic import Field

from memory_mcp.responses import (
    AccessPatternResponse,
    PredictionResponse,
    error_response,
    memory_to_response,
    success_response,
)
from memory_mcp.server.app import mcp, settings, storage


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

    Promotes predicted memories to hot cache for instant recall.
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
