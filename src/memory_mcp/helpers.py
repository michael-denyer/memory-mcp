"""Helper functions for MCP server tool implementations.

This module contains utility functions used by server.py tools for
validation, formatting, and response building.
"""

import re
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


# ========== Importance Scoring (MemGPT-inspired) ==========

# Pattern definitions for importance scoring
_CODE_INDICATORS = [
    r"```",  # Code blocks
    r"def\s+\w+\s*\(",  # Python functions
    r"class\s+\w+",  # Classes
    r"import\s+\w+",  # Imports
    r"function\s+\w+",  # JS functions
    r"\w+\s*=\s*['\"]",  # Assignments
    r"npm\s+\w+|pip\s+\w+|uv\s+\w+",  # Package commands
    r"git\s+\w+",  # Git commands
]

_ENTITY_PATTERNS = [
    r"/[\w/.-]+",  # File paths
    r"https?://\S+",  # URLs
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+",  # CamelCase names
    r"\b[A-Z]{2,}\b",  # ACRONYMS
    r"\b\d+\.\d+\.\d+\b",  # Version numbers
    r"@\w+",  # Mentions/decorators
]


def _compute_length_score(length: int) -> float:
    """Compute length-based importance score (optimal around 200-500 chars)."""
    if length < 20:
        return 0.2  # Too short, low value
    if length < 100:
        return 0.5
    if length < 500:
        return 1.0  # Sweet spot
    if length < 2000:
        return 0.8  # Still good but might be verbose
    return 0.6  # Very long, might need summarization


def _count_pattern_matches(content: str, patterns: list[str]) -> int:
    """Count how many patterns match in the content."""
    return sum(1 for pat in patterns if re.search(pat, content))


def compute_importance_score(
    content: str,
    length_weight: float = 0.3,
    code_weight: float = 0.4,
    entity_weight: float = 0.3,
) -> float:
    """Compute importance score for content at admission time.

    Uses simple heuristics to estimate content value:
    - Length factor: Longer content often has more information
    - Code factor: Code blocks/patterns are high-value
    - Entity factor: Named entities, paths, URLs indicate specificity

    Args:
        content: The memory content to score
        length_weight: Weight for length component (0-1)
        code_weight: Weight for code detection component (0-1)
        entity_weight: Weight for entity density component (0-1)

    Returns:
        Importance score between 0.0 and 1.0
    """
    length_score = _compute_length_score(len(content))
    code_matches = _count_pattern_matches(content, _CODE_INDICATORS)
    code_score = min(1.0, code_matches * 0.25)
    entity_matches = _count_pattern_matches(content, _ENTITY_PATTERNS)
    entity_score = min(1.0, entity_matches * 0.2)

    total = length_score * length_weight + code_score * code_weight + entity_score * entity_weight

    return round(min(1.0, max(0.0, total)), 3)


def get_importance_breakdown(
    content: str,
    length_weight: float = 0.3,
    code_weight: float = 0.4,
    entity_weight: float = 0.3,
) -> dict:
    """Get detailed breakdown of importance score components.

    Useful for debugging and transparency.
    """
    length = len(content)
    length_score = _compute_length_score(length)
    code_matches = _count_pattern_matches(content, _CODE_INDICATORS)
    code_score = min(1.0, code_matches * 0.25)
    entity_matches = _count_pattern_matches(content, _ENTITY_PATTERNS)
    entity_score = min(1.0, entity_matches * 0.2)

    total = length_score * length_weight + code_score * code_weight + entity_score * entity_weight

    return {
        "score": round(min(1.0, max(0.0, total)), 3),
        "length": {"chars": length, "score": length_score, "weight": length_weight},
        "code": {"matches": code_matches, "score": code_score, "weight": code_weight},
        "entities": {
            "matches": entity_matches,
            "score": entity_score,
            "weight": entity_weight,
        },
    }


# ========== Salience Scoring (Engram-inspired) ==========


def _compute_recency_score(
    last_accessed_at: datetime | None, halflife_days: float
) -> tuple[float, float | None]:
    """Compute recency score with exponential decay.

    Args:
        last_accessed_at: When last accessed (None if never)
        halflife_days: Half-life for recency decay

    Returns:
        Tuple of (recency_score, days_since_access or None)
    """
    if not last_accessed_at:
        return 0.0, None

    now = datetime.now(timezone.utc)
    if last_accessed_at.tzinfo is None:
        last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)
    days_since = (now - last_accessed_at).total_seconds() / 86400
    score = 2 ** (-days_since / halflife_days)
    return score, days_since


def compute_salience_score(
    importance_score: float,
    trust_score: float,
    access_count: int,
    last_accessed_at: datetime | None,
    importance_weight: float = 0.25,
    trust_weight: float = 0.25,
    access_weight: float = 0.25,
    recency_weight: float = 0.25,
    recency_halflife_days: float = 14.0,
    max_access_count: int = 20,
) -> float:
    """Compute unified salience score for promotion/eviction decisions.

    Combines multiple signals into a single metric (Engram-inspired):
    - Importance: Content-based value (code, entities, length)
    - Trust: Confidence in accuracy (decays over time, boosted by use)
    - Access: Usage frequency (normalized)
    - Recency: How recently accessed (exponential decay)

    Args:
        importance_score: Admission-time importance (0-1)
        trust_score: Current trust score (0-1)
        access_count: Number of times accessed
        last_accessed_at: When last accessed (None if never)
        importance_weight: Weight for importance component
        trust_weight: Weight for trust component
        access_weight: Weight for access component
        recency_weight: Weight for recency component
        recency_halflife_days: Half-life for recency decay
        max_access_count: Access count that maps to 1.0 (for normalization)

    Returns:
        Salience score between 0.0 and 1.0
    """
    access_normalized = min(1.0, access_count / max_access_count)
    recency_score, _ = _compute_recency_score(last_accessed_at, recency_halflife_days)

    salience = (
        importance_score * importance_weight
        + trust_score * trust_weight
        + access_normalized * access_weight
        + recency_score * recency_weight
    )

    return round(min(1.0, max(0.0, salience)), 3)


def get_salience_breakdown(
    importance_score: float,
    trust_score: float,
    access_count: int,
    last_accessed_at: datetime | None,
    importance_weight: float = 0.25,
    trust_weight: float = 0.25,
    access_weight: float = 0.25,
    recency_weight: float = 0.25,
    recency_halflife_days: float = 14.0,
    max_access_count: int = 20,
) -> dict:
    """Get detailed breakdown of salience score components.

    Useful for debugging and transparency.
    """
    access_normalized = min(1.0, access_count / max_access_count)
    recency_score, days_since_access = _compute_recency_score(
        last_accessed_at, recency_halflife_days
    )

    salience = compute_salience_score(
        importance_score,
        trust_score,
        access_count,
        last_accessed_at,
        importance_weight,
        trust_weight,
        access_weight,
        recency_weight,
        recency_halflife_days,
        max_access_count,
    )

    return {
        "salience": salience,
        "importance": {
            "score": importance_score,
            "weight": importance_weight,
            "component": round(importance_score * importance_weight, 3),
        },
        "trust": {
            "score": trust_score,
            "weight": trust_weight,
            "component": round(trust_score * trust_weight, 3),
        },
        "access": {
            "count": access_count,
            "normalized": round(access_normalized, 3),
            "weight": access_weight,
            "component": round(access_normalized * access_weight, 3),
        },
        "recency": {
            "days_since": round(days_since_access, 1) if days_since_access else None,
            "score": round(recency_score, 3),
            "weight": recency_weight,
            "component": round(recency_score * recency_weight, 3),
        },
    }
