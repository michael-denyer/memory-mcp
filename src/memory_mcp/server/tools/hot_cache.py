"""Hot cache tools: hot_cache_status, metrics_status, promote, demote, pin, unpin."""

from typing import Annotated

from pydantic import Field

from memory_mcp.logging import (
    metrics,
    record_hot_cache_change,
    update_hot_cache_stats,
)
from memory_mcp.responses import (
    HotCacheEffectivenessResponse,
    HotCacheMetricsResponse,
    HotCacheResponse,
    error_response,
    memory_to_response,
    success_response,
)
from memory_mcp.server.app import mcp, storage


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
    hot_memories = storage.get_promoted_memories()
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
    """Manually promote a memory to hot cache for instant recall."""
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
