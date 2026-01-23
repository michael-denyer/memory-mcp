"""Helper functions for MCP server tool implementations.

This module contains utility functions used by server.py tools for
validation, formatting, and response building.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.models import DisplayCluster, Memory, MemoryType
from memory_mcp.responses import FormattedMemory, error_response

if TYPE_CHECKING:
    import numpy as np

log = get_logger("helpers")


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


def get_bayesian_helpfulness(
    used_count: int,
    retrieved_count: int,
    alpha: float = 1.0,
    beta: float = 3.0,
) -> float:
    """Compute Bayesian helpfulness rate using Beta-Binomial posterior.

    Uses the formula: (used + α) / (retrieved + α + β)

    This provides smooth handling of cold start:
    - 0 used, 0 retrieved → 0.25 (benefit of doubt)
    - 0 used, 5 retrieved → 0.125 (evidence of low utility)
    - 2 used, 5 retrieved → 0.33 (decent signal)
    - 5 used, 10 retrieved → 0.43 (strong signal)

    Args:
        used_count: Number of times memory was marked as used/helpful
        retrieved_count: Number of times memory was returned in recall
        alpha: Prior successes (default 1.0, optimistic)
        beta: Prior failures (default 3.0, assumes 25% base rate)

    Returns:
        Estimated helpfulness rate between 0 and 1
    """
    result = (used_count + alpha) / (retrieved_count + alpha + beta)

    log.debug(
        "get_bayesian_helpfulness: rate={:.3f} (used={}, retrieved={}, alpha={}, beta={})",
        result,
        used_count,
        retrieved_count,
        alpha,
        beta,
    )

    return result


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
    trust_weight: float = 0.0,
    helpfulness_weight: float = 0.0,
    hybrid_enabled: bool = False,
    prefix: str = "",
) -> str:
    """Build ranking factors explanation string for recall responses.

    Args:
        mode_name: Name of the recall mode (e.g., 'balanced')
        similarity_weight: Weight for similarity factor (0-1)
        recency_weight: Weight for recency factor (0-1)
        access_weight: Weight for access factor (0-1)
        trust_weight: Weight for trust factor (0-1)
        helpfulness_weight: Weight for helpfulness factor (0-1)
        hybrid_enabled: Whether keyword boost is active
        prefix: Optional prefix to prepend

    Returns:
        Human-readable ranking factors string
    """
    factors = []

    # Only include factors with non-zero weight
    if similarity_weight > 0:
        factors.append(f"similarity ({int(similarity_weight * 100)}%)")
    if recency_weight > 0:
        factors.append(f"recency ({int(recency_weight * 100)}%)")
    if access_weight > 0:
        factors.append(f"access ({int(access_weight * 100)}%)")
    if trust_weight > 0:
        factors.append(f"trust ({int(trust_weight * 100)}%)")
    if helpfulness_weight > 0:
        factors.append(f"helpfulness ({int(helpfulness_weight * 100)}%)")

    base = f"Mode: {mode_name} | Ranked by: " + " + ".join(factors)

    if hybrid_enabled:
        base += " [keyword boost active]"

    return f"{prefix} | {base}" if prefix else base


# ========== Category Inference ==========

# Pattern definitions for auto-detecting memory categories
# Focus on HIGH-VALUE categories that capture non-discoverable context
# Excludes: import, command, config, api (these are already in the codebase)
#
# Temporal scope labels:
#   [durable]    - Long-lived knowledge, rarely changes (architecture, conventions)
#   [stable]     - Stable but may evolve over time (decisions, preferences)
#   [transient]  - Short-lived, session or task-specific (context, todo, bug)
_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "antipattern": [
        # [durable] Negative guidance - what NOT to do (surface early in plans)
        r"\bdon't\s+(use|do|try|call|create|add|put)\b",
        r"\bnever\s+(use|do|try|call|create|add|put)\b",
        r"\bavoid\s+(using|doing|calling)\b",
        r"\b(bad|wrong|incorrect)\s+(way|approach|pattern)\b",
        r"\b(anti-?pattern|code smell)\b",
        r"\bdon't\b.*\binstead\b",
        r"\b(deprecated|obsolete|legacy)\b",
        r"\bshould\s+(not|never)\b",
    ],
    "landmine": [
        # [durable] Critical warnings about things that can break silently
        r"\b(warning|careful|watch out|beware|danger)\b",
        r"\bdon't\s+(forget|miss|skip|ignore)\b",
        r"\b(gotcha|pitfall|trap|landmine|caveat)\b",
        r"\b(easy to|easily)\s+(miss|forget|overlook|break)\b",
        r"\b(silently|quietly)\s+(fail|break|ignore)\b",
        r"\b(subtle|hidden|tricky)\s+(bug|issue|problem)\b",
    ],
    "decision": [
        # [stable] Why we chose X over Y - rationale persists even if choice changes
        r"\b(decided|chose|chosen|picked|selected|opted)\b",
        r"\b(decision|choice):\s",
        r"\b(because|reason|rationale|why)\b.*\b(chose|use|picked|decided)\b",
        r"\b(over|instead of|rather than)\b",
        r"\bwe (went with|will use|are using)\b",
        r"\btrade-?off",
    ],
    "convention": [
        # [durable] Project rules, standards, and workflow patterns
        r"\b(convention|standard|rule|guideline)\b",
        r"\b(always|never)\s+(use|do|name|put|run)\b",
        r"\b(naming|coding|style)\s+(convention|pattern|standard)\b",
        r"\bwe (always|never)\b",
        r"\bthe (rule|norm|standard|process|workflow) is\b",
        r"\b(consistent|consistency)\b",
        r"\b(first|then|after|before)\s+(you|we|i)\s+(should|need|must)\b",
    ],
    "preference": [
        # [stable] User/team preferences and recommendations
        r"\b(prefer|like|want|favor)\s+(to|using|that)\b",
        r"\b(recommend|suggest|advise|propose)\b",
        r"\bshould (use|consider|try|avoid)\b",
        r"\bbest practice\b",
        r"\b(better|worse) (to|than)\b",
        r"\b(ideal|optimal)ly?\b",
        r"\bmy (style|approach|preference)\b",
    ],
    "lesson": [
        # [stable] Learnings and discoveries - knowledge that persists
        r"\b(learned|realized|discovered|found out)\b",
        r"\b(turns out|it appears|apparently)\b",
        r"\b(insight|takeaway|lesson)\b",
        r"\b(didn't know|now know|now understand)\b",
        r"\bin hindsight\b",
    ],
    "constraint": [
        # [stable] Limitations and requirements - may change with versions/updates
        r"\b(must|cannot|can't|won't|unable)\b",
        r"\b(limitation|restriction|requirement)\b",
        r"\b(blocked by|depends on|requires)\b",
        r"\b(incompatible|unsupported)\b",
        r"\b(only works|doesn't work)\b",
        r"\b(mandatory|required|necessary)\b",
    ],
    "architecture": [
        # [durable] System design and structure - core knowledge
        r"\b(architecture|design|pattern|structure)\b",
        r"\b(component|module|layer|service)\b.*\b(responsible|handles|manages)\b",
        r"\bdata flow\b",
        r"\b(API|endpoint|route)s?\b.*\b(design|structure)\b",
        r"\b(schema|model)s?\b",
    ],
    "context": [
        # [transient] Background for current task/session
        r"\b(background|context|history)\b",
        r"\b(for reference|fyi|note that)\b",
        r"\b(previously|earlier|before this)\b",
        r"\b(the situation|the state|currently)\b",
        r"\b(at the time|back when)\b",
        r"\b(because|since|therefore|thus|hence)\b",
        r"\bthe reason\b",
    ],
    "bug": [
        # [transient] Issues and workarounds - resolved once fixed
        r"\b(bug|issue|error|problem|fix|fixed)\b",
        r"\b(workaround|hack)\b",
        r"\b(broke|broken|fails?|failed)\b",
    ],
    "todo": [
        # [transient] Future work - completed or abandoned
        r"\b(TODO|FIXME|HACK|XXX)\b",
        r"\bneed to\b",
        r"\bshould (add|fix|update|implement)\b",
    ],
    "workflow": [
        # [stable] Deployment and operational processes
        r"\b(deploy|deployment|release|rollout)\b",
        r"\b(pipeline|workflow|process|procedure)\b",
        r"\b(step\s*\d|first|then|next|finally)\b.*\b(run|execute|do)\b",
        r"\b(ssh|curl|wget)\s+",
        r"\b(staging|production|prod|uat|dev)\b.*\b(server|environment)\b",
        r"\b(script|verify|check)\b.*\b(before|after)\b",
    ],
    "snippet": [
        # [transient] Code snippets with language markers - lower value, in codebase
        r"^\[(java|python|bash|json|sql|yaml|xml|go|rust|typescript|javascript)\]",
        r"^```(java|python|bash|json|sql|yaml|xml|go|rust|typescript|javascript)",
    ],
    "command": [
        # [transient] Short CLI commands - easily discoverable, never promote
        r"^(git|npm|yarn|pnpm|uv|pip|cargo|go|make|docker|kubectl)\s+\w+$",
        r"^(cd|ls|pwd|mkdir|rm|cp|mv|cat|grep|find|chmod|chown)\s+",
        r"^\./[\w.-]+\.sh\b",
        r"^[\w-]+\s+(--\w+|\-\w)\s*$",  # command with flags only
    ],
    "reference": [
        # [stable] External resources and documentation pointers
        r"https?://\S+",
        r"\b(documentation|docs|readme|guide|tutorial)\b",
        r"\b(see|refer to|check|read)\b.*\b(file|doc|page|link)\b",
        r"\b(version|v)\s*\d+\.\d+",
    ],
    "observation": [
        # [transient] Factual findings and discoveries
        r"\bfound\s+\d+\b",
        r"\b(noticed|observed|saw|see)\s+(that|the)\b",
        r"\b(there (is|are)|it (is|has))\b",
        r"[✓✗!]\s+",
        r"\b(confirmed|verified|checked)\b",
    ],
}

# Map categories to temporal scope for retention/decay decisions
CATEGORY_TEMPORAL_SCOPE: dict[str, str] = {
    "antipattern": "durable",  # "Don't do X" persists - surface early in plans
    "landmine": "durable",  # Critical warnings persist
    "decision": "stable",  # Rationale persists even if choice changes
    "convention": "durable",  # Project rules rarely change
    "preference": "stable",  # May evolve but core preferences persist
    "lesson": "stable",  # Knowledge that persists
    "constraint": "stable",  # May change with versions
    "architecture": "durable",  # Core system knowledge
    "context": "transient",  # Session/task-specific
    "bug": "transient",  # Resolved once fixed
    "todo": "transient",  # Completed or abandoned
    "workflow": "stable",  # Deployment processes persist
    "snippet": "transient",  # Code snippets - low value, in codebase
    "reference": "stable",  # External resources persist
    "observation": "transient",  # Factual findings - session-specific
    "command": "transient",  # CLI commands - easily discoverable
}


def get_temporal_scope(category: str | None) -> str:
    """Get temporal scope for a category.

    Args:
        category: Category string or None

    Returns:
        'durable', 'stable', or 'transient' (default for unknown/None)
    """
    if category is None:
        return "stable"  # Default for uncategorized
    return CATEGORY_TEMPORAL_SCOPE.get(category, "stable")


# High-value categories that should be promoted more eagerly
# These contain critical warnings, constraints, or "don't do X" guidance
_HIGH_VALUE_CATEGORIES = {"antipattern", "landmine", "constraint"}

# Promotion threshold multipliers by category
# Higher multiplier = harder to promote (requires higher salience)
# Category promotion multipliers: lower = easier to promote (lower salience needed)
# Note: command and snippet are blocked entirely via _CATEGORY_INELIGIBLE
_CATEGORY_PROMOTION_MULTIPLIERS = {
    "antipattern": 0.6,  # Lower threshold for warnings
    "landmine": 0.6,  # Lower threshold for gotchas
    "constraint": 0.6,  # Lower threshold for rules
}

# Categories blocked from auto-promotion entirely
# command: Easily discoverable via shell history
# snippet: Transient and rarely worth hot cache space
_CATEGORY_INELIGIBLE = {"command", "snippet"}


def should_promote_category(category: str | None) -> bool:
    """Check if a category is eligible for hot cache promotion.

    Low-value categories (command, snippet) are blocked from auto-promotion:
    - Commands are easily discoverable via shell history
    - Snippets are transient and rarely worth keeping in hot cache

    Args:
        category: Memory category or None

    Returns:
        True if category can be auto-promoted, False if blocked
    """
    if category is None:
        return True
    return category not in _CATEGORY_INELIGIBLE


def get_promotion_salience_threshold(category: str | None, default_threshold: float) -> float:
    """Get salience threshold for auto-promotion based on category.

    Uses multipliers to adjust thresholds by category:
    - High-value categories (antipattern, landmine, constraint): Lower threshold
    - Low-value categories (command, snippet): Higher threshold
    - Others: Default threshold

    Args:
        category: Memory category or None
        default_threshold: Default salience threshold from settings

    Returns:
        Salience threshold to use for promotion decision
    """
    if category is None:
        return default_threshold

    multiplier = _CATEGORY_PROMOTION_MULTIPLIERS.get(category, 1.0)
    adjusted = default_threshold * multiplier

    # Clamp to reasonable bounds
    return max(0.1, min(0.95, adjusted))


def should_auto_pin(category: str | None, trust_score: float = 1.0) -> bool:
    """Check if a memory should be auto-pinned when promoted.

    Memories with constraint/guardrail content are pinned to prevent
    auto-eviction. This ensures critical project rules stay in hot cache.

    Criteria for auto-pin:
    - Category is constraint, antipattern, or landmine
    - Trust score is high (>= 0.8) indicating validated content

    Args:
        category: Memory category
        trust_score: Current trust score (0-1)

    Returns:
        True if memory should be pinned on promotion
    """
    if category not in _HIGH_VALUE_CATEGORIES:
        return False
    # Only auto-pin if trust is high (validated content)
    return trust_score >= 0.8


def get_demotion_multiplier(category: str | None) -> float:
    """Get demotion time multiplier based on category's temporal scope.

    Durable memories resist demotion longer, transient memories demote faster.

    Args:
        category: Memory category or None

    Returns:
        Multiplier for demotion_days setting (e.g., 2.0 = twice as long)
    """
    scope = get_temporal_scope(category)
    multipliers = {
        "durable": 2.0,  # 14 days -> 28 days
        "stable": 1.0,  # 14 days -> 14 days
        "transient": 0.5,  # 14 days -> 7 days
    }
    return multipliers.get(scope, 1.0)


# Category-specific TTL (time-to-staleness) for helpfulness decay
# Higher values = longer retention, lower = faster decay
_CATEGORY_TTL_DAYS: dict[str, float] = {
    "constraint": 90.0,  # Long-lived: "must use X"
    "architecture": 60.0,  # Core system knowledge
    "antipattern": 60.0,  # "Don't do X" persists
    "landmine": 60.0,  # Critical warnings persist
    "convention": 45.0,  # Project rules
    "workflow": 45.0,  # Deployment processes
    "decision": 30.0,  # May change with context
    "preference": 30.0,  # Evolves over time
    "lesson": 30.0,  # Learnings persist
    "pattern": 30.0,  # Code patterns
    "reference": 30.0,  # External links
    "todo": 14.0,  # Short-lived task items
    "bug": 14.0,  # Resolved once fixed
    "observation": 14.0,  # Session findings
    "context": 7.0,  # Very transient
    "snippet": 7.0,  # Code snippets are ephemeral
    "command": 7.0,  # CLI commands - easily re-discoverable
}


def get_category_ttl(category: str | None) -> float:
    """Get the time-to-staleness (TTL) in days for a category.

    Used for helpfulness recency decay. Memories in transient categories
    have shorter TTLs and their helpfulness decays faster.

    Args:
        category: Memory category or None

    Returns:
        TTL in days (default 30.0 for unknown/None categories)
    """
    if category is None:
        return 30.0  # Default for uncategorized
    return _CATEGORY_TTL_DAYS.get(category, 30.0)


def compute_helpfulness_with_decay(
    utility_score: float,
    last_used_at: datetime | None,
    category: str | None = None,
    decay_base: float = 0.95,
) -> float:
    """Apply category-aware recency decay to helpfulness score.

    Uses the formula: utility_score * decay_base^(days_since_use / category_ttl)

    This ensures:
    - Recently used memories maintain high helpfulness
    - Transient categories (todo, bug, context) decay faster
    - Durable categories (constraint, architecture) decay slower
    - Never-used memories (last_used_at = None) get slight penalty (0.8x)

    Args:
        utility_score: Bayesian helpfulness from usage tracking
        last_used_at: When the memory was last marked as used
        category: Memory category for TTL lookup
        decay_base: Base decay rate (default 0.95)

    Returns:
        Helpfulness score with decay applied (0-1)
    """
    if last_used_at is None:
        # Never used - apply slight penalty but don't zero out
        # This gives cold-start memories a chance
        result = utility_score * 0.8
        log.debug(
            "compute_helpfulness_with_decay: result={:.3f} (utility={:.3f}, penalty=0.8)",
            result,
            utility_score,
        )
        return result

    ttl_days = get_category_ttl(category)

    now = datetime.now(timezone.utc)
    if last_used_at.tzinfo is None:
        last_used_at = last_used_at.replace(tzinfo=timezone.utc)

    days_since_use = (now - last_used_at).total_seconds() / 86400
    decay = decay_base ** (days_since_use / ttl_days)
    result = utility_score * decay

    log.debug(
        "compute_helpfulness_with_decay: result={:.3f} (utility={:.3f}, decay={:.3f}, "
        "days_since={:.1f}, ttl={}, category={})",
        result,
        utility_score,
        decay,
        days_since_use,
        ttl_days,
        category,
    )

    return result


def infer_category(content: str) -> str | None:
    """Infer category from content using pattern matching.

    Scans content for language patterns indicating category type.
    Returns the category with most matches, or None if no strong signal.

    Args:
        content: Memory content to analyze

    Returns:
        Category string (e.g., 'decision', 'architecture') or None
    """
    scores: dict[str, int] = {}

    for category, patterns in _CATEGORY_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                score += 1
        if score > 0:
            scores[category] = score

    if not scores:
        log.debug(
            "infer_category: no patterns matched, content_preview='{}'",
            content[:60].replace("\n", " "),
        )
        return None

    # Return category with highest score, with tie-breaker for more specific categories
    # Priority: high-value tacit knowledge first, then explicit knowledge
    priority = [
        "antipattern",  # "Don't do X" - surface early in plans
        "landmine",  # Critical: things that can break silently
        "decision",  # Why we chose X over Y
        "convention",  # Project rules and standards
        "preference",  # User/team preferences and recommendations
        "lesson",  # Learnings and discoveries
        "constraint",  # Limitations and requirements
        "architecture",  # System design
        "workflow",  # Deployment and processes
        "context",  # Background and reasoning
        "reference",  # External resources
        "observation",  # Factual findings
        "todo",  # Future work
        "bug",  # Issues
        "snippet",  # Code snippets (lowest - in codebase)
        "command",  # CLI commands (lowest - easily discoverable)
    ]
    max_score = max(scores.values())
    candidates = [cat for cat, score in scores.items() if score == max_score]

    # Pick highest priority among tied candidates
    result = next(
        (cat for cat in priority if cat in candidates), candidates[0] if candidates else None
    )

    log.debug(
        "infer_category: category='{}' score={} candidates={} content_preview='{}'",
        result,
        max_score,
        candidates,
        content[:60].replace("\n", " "),
    )
    return result


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
    result = round(min(1.0, max(0.0, total)), 3)

    log.debug(
        "compute_importance: {:.3f} (len={:.2f}*{}, code={:.2f}*{}, ent={:.2f}*{}) chars={}",
        result,
        length_score,
        length_weight,
        code_score,
        code_weight,
        entity_score,
        entity_weight,
        len(content),
    )

    return result


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
    importance_weight: float = 0.15,
    trust_weight: float = 0.15,
    access_weight: float = 0.40,
    recency_weight: float = 0.30,
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
    result = round(min(1.0, max(0.0, salience)), 3)

    log.debug(
        "compute_salience_score: salience={:.3f} (importance={:.2f}*{}, trust={:.2f}*{}, "
        "access={:.2f}*{}, recency={:.2f}*{})",
        result,
        importance_score,
        importance_weight,
        trust_score,
        trust_weight,
        access_normalized,
        access_weight,
        recency_score,
        recency_weight,
    )

    return result


def get_salience_breakdown(
    importance_score: float,
    trust_score: float,
    access_count: int,
    last_accessed_at: datetime | None,
    importance_weight: float = 0.15,
    trust_weight: float = 0.15,
    access_weight: float = 0.40,
    recency_weight: float = 0.30,
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


# ========== Semantic Clustering for Display (RePo-inspired) ==========


def _generate_cluster_label(memories: list[Memory], max_words: int = 4) -> str:
    """Generate a human-readable label from cluster contents.

    Strategy:
    1. Find common tags across members
    2. Use most common tag as label
    3. Fall back to memory type if no tags

    Args:
        memories: Memories in the cluster
        max_words: Maximum words in the label

    Returns:
        Human-readable label (e.g., "Python Development", "Testing")
    """
    # Collect all tags and count frequency
    tag_counts: Counter[str] = Counter()
    for m in memories:
        for tag in m.tags:
            tag_counts[tag] += 1

    if tag_counts:
        # Use most common tag, title-cased
        most_common = tag_counts.most_common(1)[0][0]
        # Title-case and limit words
        words = most_common.replace("-", " ").replace("_", " ").split()
        label = " ".join(w.capitalize() for w in words[:max_words])
        return label

    # Fall back to memory type
    type_counts: Counter[str] = Counter(m.memory_type.value for m in memories)
    most_common_type = type_counts.most_common(1)[0][0]
    return most_common_type.capitalize()


def _compute_pairwise_similarity(emb_a: "np.ndarray", emb_b: "np.ndarray") -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        emb_a: First embedding vector (assumed normalized or will normalize)
        emb_b: Second embedding vector

    Returns:
        Cosine similarity (0-1)
    """
    import numpy as np

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def cluster_memories_for_display(
    memories: list[Memory],
    embeddings: dict[int, "np.ndarray"],
    threshold: float = 0.70,
    min_cluster_size: int = 2,
    max_clusters: int = 5,
) -> tuple[list[DisplayCluster], list[Memory]]:
    """Cluster memories semantically for display.

    Uses greedy clustering to group similar memories together,
    reducing cognitive load (RePo research-inspired).

    Algorithm:
    1. Start with first memory as cluster seed
    2. For each subsequent memory, check similarity to existing cluster centers
    3. If similarity >= threshold, add to that cluster
    4. Otherwise, start new cluster (up to max_clusters)
    5. Remaining items go to unclustered list

    Args:
        memories: Memories to cluster (should be pre-sorted by score)
        embeddings: Dict mapping memory_id to embedding vector
        threshold: Similarity threshold for grouping (default 0.70)
        min_cluster_size: Minimum items to form a named cluster
        max_clusters: Maximum clusters before remaining go to 'Other'

    Returns:
        Tuple of (clusters, unclustered) where:
        - clusters: List of DisplayCluster objects with label, members, similarity
        - unclustered: List of memories not in any cluster
    """
    if not memories or not embeddings:
        return [], list(memories)

    # Build cluster assignments
    # Each cluster is: [seed_memory_id, [member_memories], [similarities]]
    clusters_raw: list[tuple[int, list[Memory], list[float]]] = []
    unclustered: list[Memory] = []

    for memory in memories:
        if memory.id not in embeddings:
            unclustered.append(memory)
            continue

        mem_embedding = embeddings[memory.id]
        best_cluster_idx = -1
        best_similarity = 0.0

        # Find best matching cluster
        for i, (seed_id, members, _) in enumerate(clusters_raw):
            if seed_id not in embeddings:
                continue
            seed_embedding = embeddings[seed_id]
            sim = _compute_pairwise_similarity(mem_embedding, seed_embedding)
            if sim >= threshold and sim > best_similarity:
                best_cluster_idx = i
                best_similarity = sim

        if best_cluster_idx >= 0:
            # Add to existing cluster
            clusters_raw[best_cluster_idx][1].append(memory)
            clusters_raw[best_cluster_idx][2].append(best_similarity)
        elif len(clusters_raw) < max_clusters:
            # Start new cluster
            clusters_raw.append((memory.id, [memory], [1.0]))
        else:
            # Max clusters reached, add to unclustered
            unclustered.append(memory)

    # Convert to DisplayCluster objects
    clusters: list[DisplayCluster] = []
    for seed_id, members, similarities in clusters_raw:
        if len(members) < min_cluster_size:
            # Too small, move to unclustered
            unclustered.extend(members)
        else:
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            label = _generate_cluster_label(members)
            clusters.append(
                DisplayCluster(
                    label=label,
                    members=members,  # Already sorted by original order (score)
                    avg_similarity=round(avg_sim, 3),
                )
            )

    return clusters, unclustered


# ========== Concise Memory Formatting ==========

# Category display prefixes (uppercase for visibility)
_CATEGORY_PREFIXES: dict[str, str] = {
    "antipattern": "ANTIPATTERN",
    "landmine": "WARNING",
    "decision": "DECISION",
    "convention": "CONVENTION",
    "preference": "PREFERENCE",
    "lesson": "LESSON",
    "constraint": "CONSTRAINT",
    "architecture": "ARCHITECTURE",
    "context": "CONTEXT",
    "bug": "BUG",
    "todo": "TODO",
    "workflow": "WORKFLOW",
    "snippet": "SNIPPET",
    "reference": "REFERENCE",
    "observation": "OBSERVATION",
    "command": "COMMAND",
}


def _get_category_prefix(memory: Memory) -> str:
    """Get display prefix for memory category.

    Args:
        memory: Memory to get prefix for

    Returns:
        Category prefix (e.g., "CONSTRAINT") or empty string
    """
    if memory.category:
        return _CATEGORY_PREFIXES.get(memory.category, memory.category.upper())
    return ""


def _compute_decayed_trust(memory: Memory) -> float:
    """Compute time-decayed trust for a memory with usage-aware adjustment.

    Uses exponential decay based on staleness to penalize memories
    that haven't been accessed recently. Frequently-used memories
    decay slower, reflecting their proven reliability.

    Args:
        memory: Memory to compute decayed trust for

    Returns:
        Decayed trust score (0.0 to 1.0)
    """
    import math
    from datetime import datetime

    # Type-specific decay half-lives (days)
    # Mirrors settings defaults but simplified for helper context
    type_halflife_days = {
        "project": 90,
        "pattern": 60,
        "reference": 180,
        "conversation": 14,
        "episodic": 7,
    }
    default_halflife = 90

    halflife = type_halflife_days.get(memory.memory_type, default_halflife)
    reference_time = memory.last_accessed_at if memory.last_accessed_at else memory.created_at
    days_since_activity = (datetime.now() - reference_time).total_seconds() / 86400

    # Base exponential decay
    base_decay = 2 ** (-days_since_activity / halflife)

    # Usage multiplier: frequently-used memories decay slower
    # Multiplier ranges from 1.0 (unused) to 1.5 (heavily used)
    access_count = memory.access_count or 0
    usage_factor = min(1.5, 1.0 + 0.1 * math.log(max(1, access_count + 1)))

    # Apply usage factor only if there's actual decay happening
    adjusted_decay = base_decay * usage_factor if base_decay < 1.0 else base_decay

    return (memory.trust_score or 1.0) * min(1.0, adjusted_decay)


def _get_confidence_label(memory: Memory) -> str:
    """Get confidence label for memory.

    Uses decayed trust score to account for staleness.
    Stale memories with high raw trust will show lower confidence
    to signal that they may need re-verification.

    Args:
        memory: Memory to get confidence for

    Returns:
        'high', 'medium', or 'low'
    """
    decayed_trust = _compute_decayed_trust(memory)
    if decayed_trust >= 0.8:
        return "high"
    elif decayed_trust >= 0.5:
        return "medium"
    return "low"


def _get_source_label(memory: Memory) -> str | None:
    """Get source label for memory.

    Args:
        memory: Memory to get source for

    Returns:
        Source label or None
    """
    if memory.source.value == "mined":
        return "mined"
    # Check tags for bootstrap source
    if memory.tags:
        for tag in memory.tags:
            if tag == "auto-bootstrap":
                return "bootstrap"
            if tag.endswith(".md"):
                return tag
    return None


def _format_verified_date(memory: Memory) -> str | None:
    """Format last verified date (last_accessed_at) as YYYY-MM-DD.

    Args:
        memory: Memory to format date for

    Returns:
        Formatted date or None
    """
    if memory.last_accessed_at:
        return memory.last_accessed_at.strftime("%Y-%m-%d")
    return None


def _get_staleness_indicator(memory: Memory) -> str | None:
    """Get staleness indicator for memory.

    Shows a warning when memory hasn't been verified recently,
    signaling to the LLM that the information may need re-verification.

    Args:
        memory: Memory to check staleness for

    Returns:
        Staleness warning string or None if fresh
    """
    from datetime import datetime

    # If never verified (no last_used_at), warn about it
    if not memory.last_used_at:
        return "never verified"

    days_since_use = (datetime.now() - memory.last_used_at).days

    # Stale thresholds by memory type
    # Transient types stale faster, durable types stale slower
    from memory_mcp.storage import MemoryType

    type_stale_thresholds = {
        MemoryType.EPISODIC: 3,
        MemoryType.CONVERSATION: 7,
        MemoryType.PATTERN: 14,
        MemoryType.PROJECT: 21,
        MemoryType.REFERENCE: 30,
    }
    default_threshold = 14

    threshold = type_stale_thresholds.get(memory.memory_type, default_threshold)

    if days_since_use >= threshold:
        return f"stale {days_since_use}d"

    return None


def format_memory_concise(
    memory: Memory,
    max_chars: int = 200,
    include_metadata: bool = True,
) -> str:
    """Format a memory in concise format for hot cache injection.

    Format:
    - CATEGORY: content [id:N, confidence: X, source: Y, verified: Z]

    Args:
        memory: Memory to format
        max_chars: Maximum characters for content
        include_metadata: Whether to include metadata brackets

    Returns:
        Formatted memory string
    """
    # Get category prefix
    prefix = _get_category_prefix(memory)

    # Truncate content
    content = memory.content
    if len(content) > max_chars:
        content = content[:max_chars] + "..."

    # Build metadata parts
    metadata_parts = [f"id:{memory.id}"]

    if include_metadata:
        # Confidence (uses decayed trust)
        confidence = _get_confidence_label(memory)
        metadata_parts.append(f"confidence: {confidence}")

        # Staleness warning (optional - only if stale)
        staleness = _get_staleness_indicator(memory)
        if staleness:
            metadata_parts.append(staleness)

        # Source (optional)
        source = _get_source_label(memory)
        if source:
            metadata_parts.append(f"source: {source}")

        # Verified date (optional)
        verified = _format_verified_date(memory)
        if verified:
            metadata_parts.append(f"verified: {verified}")

    metadata = ", ".join(metadata_parts)

    # Build final string
    if prefix:
        return f"- {prefix}: {content} [{metadata}]"
    return f"- {content} [{metadata}]"


def format_hot_cache_concise(
    memories: list[Memory],
    project_id: str | None = None,
    max_chars: int = 200,
) -> str:
    """Format hot cache memories in concise format.

    Format:
    [MEMORY: Hot Cache (repo: X, N items)]
    - CONSTRAINT: Use pnpm, never npm. [id:42, confidence: high, verified: 2026-01-10]
    - CONVENTION: Server changes go in src/... [id:43, confidence: high]

    Args:
        memories: List of hot cache memories
        project_id: Current project ID (e.g., "owner/repo")
        max_chars: Maximum characters per memory content

    Returns:
        Formatted hot cache string
    """
    if not memories:
        return "[MEMORY: Hot cache empty - no frequently-accessed patterns yet]"

    # Build header
    repo_part = f"repo: {project_id}, " if project_id else ""
    header = f"[MEMORY: Hot Cache ({repo_part}{len(memories)} items)]"

    lines = [header]

    for memory in memories:
        lines.append(format_memory_concise(memory, max_chars=max_chars))

    # Add feedback hint
    lines.append("")
    lines.append("(If helpful, call mark_memory_used(id) to improve ranking)")

    return "\n".join(lines)


# ========== Query Intent Detection ==========

# Intent patterns map keyword patterns to preferred categories
# Categories listed in priority order (first is most preferred)
INTENT_PATTERNS: dict[str, tuple[list[str], list[str]]] = {
    # (keywords, preferred_categories)
    "debugging": (
        ["bug", "fix", "error", "issue", "broken", "fail", "crash", "exception"],
        ["bug", "antipattern", "landmine", "pattern"],
    ),
    "howto": (
        ["how to", "how do i", "how can i", "how should i", "what is the way"],
        ["reference", "pattern", "command", "snippet"],
    ),
    "architecture": (
        ["architecture", "design", "structure", "pattern", "layout", "organize"],
        ["architecture", "project", "decision"],
    ),
    "decision": (
        ["why", "decision", "chose", "choice", "rationale", "reason", "tradeoff"],
        ["decision", "constraint", "lesson"],
    ),
    "convention": (
        ["rule", "convention", "conventions", "standard", "guideline", "best practice", "style"],
        ["convention", "constraint", "preference"],
    ),
    "setup": (
        ["setup", "install", "configure", "config", "env", "environment"],
        ["reference", "command", "project"],
    ),
    "api": (
        ["api", "endpoint", "route", "request", "response", "rest", "graphql"],
        ["reference", "pattern", "api"],
    ),
    "todo": (
        ["todo", "task", "remaining", "pending", "next", "backlog"],
        ["todo", "bug"],
    ),
}


def infer_query_intent(query: str) -> dict[str, float]:
    """Infer intent from query keywords to boost matching memory categories.

    Returns a dict of category → boost score (0.0 to 1.0).
    Categories matching query intent get a boost in recall ranking.

    This is a cheap heuristic (regex-based) that adds no latency.
    It's NOT a filter - just a ranking signal.

    Args:
        query: Search query string

    Returns:
        Dict mapping category names to boost scores (0.0-1.0).
        Empty dict if no intent detected.

    Example:
        >>> infer_query_intent("how to fix authentication bug")
        {'bug': 0.5, 'antipattern': 0.4, 'landmine': 0.3, 'pattern': 0.2,
         'reference': 0.5, 'pattern': 0.4, 'command': 0.3, 'snippet': 0.2}
    """
    import re

    query_lower = query.lower()
    category_boosts: dict[str, float] = {}

    for intent_name, (keywords, categories) in INTENT_PATTERNS.items():
        # Check if any keyword matches
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                # Keyword matched - add boosts for preferred categories
                # First category gets highest boost, decreasing for later ones
                for i, category in enumerate(categories):
                    # Boost decays: 0.5, 0.4, 0.3, 0.2 for positions 0, 1, 2, 3
                    boost = max(0.2, 0.5 - (i * 0.1))
                    # Take max if category already has a boost (from another intent)
                    category_boosts[category] = max(category_boosts.get(category, 0), boost)
                break  # Only match first keyword per intent

    return category_boosts


def compute_intent_boost(
    category: str | None,
    intent_boosts: dict[str, float],
    max_boost: float = 0.15,
) -> float:
    """Compute intent-based ranking boost for a memory.

    Args:
        category: Memory's category (e.g., "bug", "convention")
        intent_boosts: Intent boosts from infer_query_intent()
        max_boost: Maximum boost to apply (default 0.15 = 15% boost)

    Returns:
        Boost value to add to composite score (0.0 to max_boost)
    """
    if not category or not intent_boosts:
        return 0.0

    # Get boost for this category (if any)
    raw_boost = intent_boosts.get(category, 0.0)

    # Scale to max_boost
    return raw_boost * max_boost / 0.5  # 0.5 is max raw boost
