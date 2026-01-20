"""Helper functions for MCP server tool implementations.

This module contains utility functions used by server.py tools for
validation, formatting, and response building.
"""

from datetime import datetime, timezone

from memory_mcp.models import Memory, MemoryType
from memory_mcp.responses import FormattedMemory, error_response


def parse_memory_type(memory_type: str) -> MemoryType | None:
    """Parse memory type string, returning None if invalid."""
    try:
        return MemoryType(memory_type)
    except ValueError:
        return None


def invalid_memory_type_error() -> dict:
    """Return error for invalid memory type."""
    return error_response(f"Invalid memory_type. Use: {[t.value for t in MemoryType]}")


def format_age(created_at: datetime) -> str:
    """Format memory age as human-readable string."""
    now = datetime.now(timezone.utc)
    # Handle naive datetime (assume UTC) vs aware datetime
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = now - created_at

    if delta.days >= 365:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''}"
    elif delta.days >= 30:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''}"
    elif delta.days >= 7:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    elif delta.days >= 1:
        return f"{delta.days} day{'s' if delta.days > 1 else ''}"
    elif delta.seconds >= 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''}"
    else:
        return "just now"


def get_similarity_confidence(
    similarity: float | None,
    high_threshold: float,
    default_threshold: float,
) -> str:
    """Map similarity score to confidence label.

    Args:
        similarity: Similarity score (0-1) or None
        high_threshold: Threshold for 'high' confidence
        default_threshold: Threshold for 'medium' confidence

    Returns:
        'high', 'medium', 'low', or 'unknown'
    """
    if similarity is None:
        return "unknown"
    if similarity >= high_threshold:
        return "high"
    if similarity >= default_threshold:
        return "medium"
    return "low"


def summarize_content(content: str, max_length: int = 150) -> str:
    """Create concise summary of memory content.

    - Strips code blocks to first line
    - Truncates long content with ellipsis
    - Preserves key information
    """
    lines = content.strip().split("\n")

    # If it's a code block, take the first meaningful line
    if content.startswith("```") or content.startswith("["):
        # Find first non-empty, non-fence line
        for line in lines:
            if line and not line.startswith("```") and not line.startswith("["):
                summary = line.strip()
                break
        else:
            summary = lines[0] if lines else content
    else:
        # Take first line for prose
        summary = lines[0].strip() if lines else content

    # Truncate if needed
    if len(summary) > max_length:
        summary = summary[: max_length - 3] + "..."

    return summary


def format_memories_for_llm(
    memories: list[Memory],
    high_threshold: float,
    default_threshold: float,
) -> tuple[list[FormattedMemory], str]:
    """Transform memories into LLM-friendly format.

    Args:
        memories: List of Memory objects to format
        high_threshold: Similarity threshold for 'high' confidence
        default_threshold: Similarity threshold for 'medium' confidence

    Returns:
        Tuple of (formatted memories list, context summary string)
    """
    if not memories:
        return [], "No matching memories found"

    formatted = []
    type_counts: dict[str, int] = {}

    for m in memories:
        # Count by type for summary
        mem_type = m.memory_type.value
        type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        formatted.append(
            FormattedMemory(
                summary=summarize_content(m.content),
                memory_type=mem_type,
                tags=m.tags[:5],  # Limit tags shown
                age=format_age(m.created_at),
                confidence=get_similarity_confidence(
                    m.similarity, high_threshold, default_threshold
                ),
                source_hint="hot cache" if m.is_hot else "cold storage",
            )
        )

    # Build context summary
    type_parts = [f"{count} {typ}" for typ, count in type_counts.items()]
    summary = f"Found {len(memories)} memories: {', '.join(type_parts)}"

    return formatted, summary


def get_promotion_suggestions(
    memories: list[Memory],
    promotion_threshold: int,
    max_suggestions: int = 2,
) -> list[dict]:
    """Generate promotion suggestions for frequently-accessed cold memories.

    Suggests promoting memories that:
    - Are NOT already in hot cache
    - Have high access count (>= promotion_threshold)
    - Were useful in this recall (high similarity)

    Args:
        memories: List of memories from recall
        promotion_threshold: Minimum access count to suggest promotion
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of dicts with memory_id, access_count, and reason.
    """
    suggestions = []

    for m in memories:
        if m.is_hot:
            continue  # Already hot, skip

        if m.access_count >= promotion_threshold:
            suggestions.append(
                {
                    "memory_id": m.id,
                    "access_count": m.access_count,
                    "reason": f"Accessed {m.access_count}x - consider promoting to hot cache",
                }
            )

        if len(suggestions) >= max_suggestions:
            break

    return suggestions


def build_ranking_factors(
    mode_name: str,
    similarity_weight: float,
    recency_weight: float,
    access_weight: float,
    prefix: str = "",
) -> str:
    """Build ranking factors explanation string for recall responses.

    Args:
        mode_name: Name of the recall mode (e.g., 'balanced')
        similarity_weight: Weight for similarity factor (0-1)
        recency_weight: Weight for recency factor (0-1)
        access_weight: Weight for access factor (0-1)
        prefix: Optional prefix to prepend

    Returns:
        Human-readable ranking factors string
    """
    sim_w = int(similarity_weight * 100)
    rec_w = int(recency_weight * 100)
    acc_w = int(access_weight * 100)
    base = (
        f"Mode: {mode_name} | "
        f"Ranked by: similarity ({sim_w}%) + recency ({rec_w}%) + access ({acc_w}%)"
    )
    return f"{prefix} | {base}" if prefix else base
