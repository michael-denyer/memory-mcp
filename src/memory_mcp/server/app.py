"""FastMCP app setup, shared state, helper functions, and resources."""

import os
import uuid

from fastmcp import FastMCP

from memory_mcp.config import (
    check_stop_hook_configured,
    find_bootstrap_files,
    get_hook_install_instructions,
    get_settings,
)
from memory_mcp.helpers import (
    build_ranking_factors as _build_ranking_factors,
)
from memory_mcp.helpers import (
    cluster_memories_for_display as _cluster_memories_for_display,
)
from memory_mcp.helpers import (
    format_hot_cache_concise as _format_hot_cache_concise,
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
from memory_mcp.logging import (
    configure_logging,
    get_logger,
)
from memory_mcp.models import DisplayCluster, Memory
from memory_mcp.project import detect_project, get_current_project_id
from memory_mcp.storage import MemoryType, RecallMode, Storage

log = get_logger("server")

# ========== Shared State ==========
settings = get_settings()
configure_logging(level=settings.log_level, log_format=settings.log_format)
storage = Storage(settings)
mcp = FastMCP("memory-mcp")

log.info("Memory MCP server initialized")

# Warn if Stop hook not configured (pattern mining won't work without it)
if settings.warn_missing_hook and settings.mining_enabled:
    if not check_stop_hook_configured():
        log.warning("Stop hook not configured - pattern mining will not auto-log outputs")
        log.warning(get_hook_install_instructions())


# ========== Helper Function Wrappers ==========


def build_ranking_factors(mode: RecallMode | None, prefix: str = "") -> str:
    """Build ranking factors string using module storage."""
    mode_config = storage.get_recall_mode_config(mode or RecallMode.BALANCED)
    mode_name = mode.value if mode else "balanced"
    return _build_ranking_factors(
        mode_name,
        mode_config.similarity_weight,
        mode_config.recency_weight,
        mode_config.access_weight,
        trust_weight=settings.recall_trust_weight,
        helpfulness_weight=settings.recall_helpfulness_weight,
        hybrid_enabled=settings.hybrid_search_enabled,
        prefix=prefix,
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


def get_auto_project_id() -> str | None:
    """Get the current project ID if project awareness is enabled."""
    if not settings.project_awareness_enabled:
        return None
    return get_current_project_id()


# ========== Session State ==========

_current_session_id: str | None = None


def get_current_session_id() -> str:
    """Get or create the current server session ID.

    Creates a session on first call, reusing it for all subsequent calls.
    This enables zero-config session tracking - memories are automatically
    associated with the current session without requiring explicit session_id.
    """
    global _current_session_id
    if _current_session_id is None:
        _current_session_id = str(uuid.uuid4())
        project_path = os.getcwd()
        storage.create_or_get_session(_current_session_id, project_path=project_path)
        log.info("Auto-created session: {} for project: {}", _current_session_id, project_path)
    return _current_session_id


# ========== Hot Cache Formatting ==========

# Track whether we've attempted auto-bootstrap this session (per working directory)
_auto_bootstrap_attempted: set[str] = set()


def _try_auto_bootstrap() -> bool:
    """Attempt to auto-bootstrap from current directory if hot cache is empty.

    Returns True if bootstrap was attempted and created memories.
    Only runs once per working directory per session.
    Disabled by default (auto_bootstrap=False) since markdown files are often
    redundant with what's already in the codebase/context.
    """
    from pathlib import Path

    # Check if auto-bootstrap is enabled (disabled by default)
    if not settings.auto_bootstrap:
        return False

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


def _format_memory_list(memories: list[Memory], header: str, include_ids: bool = False) -> str:
    """Format a list of memories as a resource string.

    Args:
        memories: List of Memory objects to format
        header: Header line for the resource (e.g., "[MEMORY: Hot Cache]")
        include_ids: If True, include memory IDs for feedback tracking

    Returns:
        Formatted string with header and memory items
    """
    max_chars = settings.hot_cache_display_max_chars
    lines = [header]

    for m in memories:
        content = m.content[:max_chars] + "..." if len(m.content) > max_chars else m.content
        tags_str = f" [{', '.join(m.tags[:3])}]" if m.tags else ""
        id_prefix = f"[id:{m.id}] " if include_ids else ""
        lines.append(f"- {id_prefix}{content}{tags_str}")

    return "\n".join(lines)


def _format_clustered_memory_list(
    clusters: list[DisplayCluster],
    unclustered: list[Memory],
    header: str,
    include_ids: bool = False,
) -> str:
    """Format clustered memories as a resource string.

    Semantic clustering groups related memories together to reduce
    cognitive load (RePo research-inspired).

    Args:
        clusters: List of DisplayCluster objects
        unclustered: Memories that didn't fit in any cluster
        header: Header line for the resource
        include_ids: If True, include memory IDs for feedback tracking

    Returns:
        Formatted string with headers for each cluster
    """
    max_chars = settings.hot_cache_display_max_chars
    lines = [header]

    for cluster in clusters:
        lines.append(f"\n## {cluster.label} ({cluster.size} items)")
        for m in cluster.members:
            content = m.content[:max_chars] + "..." if len(m.content) > max_chars else m.content
            tags_str = f" [{', '.join(m.tags[:3])}]" if m.tags else ""
            id_prefix = f"[id:{m.id}] " if include_ids else ""
            lines.append(f"  - {id_prefix}{content}{tags_str}")

    if unclustered:
        lines.append("\n## Other")
        for m in unclustered:
            content = m.content[:max_chars] + "..." if len(m.content) > max_chars else m.content
            tags_str = f" [{', '.join(m.tags[:3])}]" if m.tags else ""
            id_prefix = f"[id:{m.id}] " if include_ids else ""
            lines.append(f"  - {id_prefix}{content}{tags_str}")

    return "\n".join(lines)


# ========== MCP Resources ==========


@mcp.resource("memory://promoted-memories")
def promoted_memories_resource() -> str:
    """All promoted memories (~20 items) - the backing store for hot cache.

    Disabled by default. The hot-cache resource (session-aware ~10 items) is
    the recommended injection. Enable this with MEMORY_MCP_PROMOTED_RESOURCE_ENABLED=true.

    Contains all frequently-used memories that have been auto-promoted based on
    access patterns or manually promoted. The hot-cache resource draws from this.

    If empty and documentation files exist, auto-bootstraps from README.md, CLAUDE.md, etc.

    Project-aware: If project awareness is enabled, filters to current project
    plus global memories.
    """
    if not settings.promoted_resource_enabled:
        return ""

    # Get current project for project-aware hot cache
    project_id = get_auto_project_id()
    hot_memories = storage.get_promoted_memories(project_id=project_id)

    if not hot_memories and _try_auto_bootstrap():
        hot_memories = storage.get_promoted_memories(project_id=project_id)

    if not hot_memories:
        storage.record_hot_cache_miss()
        return "[MEMORY: Hot cache empty - no frequently-accessed patterns yet]"

    storage.record_hot_cache_hit()

    # Log injections for feedback loop tracking
    session_id = get_current_session_id()
    memory_ids = [m.id for m in hot_memories]
    storage.log_injections_batch(
        memory_ids=memory_ids,
        resource="hot-cache",
        session_id=session_id,
        project_id=project_id,
    )

    # Use concise format with category prefixes
    return _format_hot_cache_concise(
        memories=hot_memories,
        project_id=project_id,
        max_chars=settings.hot_cache_display_max_chars,
    )


@mcp.resource("memory://hot-cache")
def hot_cache_resource() -> str:
    """Session-aware active memory context - instant recall (0ms).

    The primary context injection. Provides a compact set (~10 items) of
    contextually relevant memories:
    1. Recently recalled memories (that were actually used)
    2. Predicted next memories (from access pattern learning)
    3. Top items from promoted memories (to fill remaining slots)

    This is what "hot cache" means - instantly available, no tool call needed.
    Semantic clustering groups related items together.
    Logs injections for feedback loop analysis (7-day retention).
    """
    if not settings.hot_cache_enabled:
        return "[MEMORY: Hot cache disabled]"

    hot_memories = storage.get_hot_cache()

    if not hot_memories:
        return "[MEMORY: Hot cache empty - no recent activity]"

    # Log injections for feedback loop tracking
    project_id = get_auto_project_id()
    session_id = get_current_session_id()
    memory_ids = [m.id for m in hot_memories]
    storage.log_injections_batch(
        memory_ids=memory_ids,
        resource="hot-cache",
        session_id=session_id,
        project_id=project_id,
    )

    header = "[MEMORY: Hot Cache - Active context]"

    # Apply semantic clustering if enabled and enough memories
    if settings.clustering_display_enabled and len(hot_memories) >= 4:
        memory_ids = [m.id for m in hot_memories]
        embeddings = storage.get_embeddings_for_memories(memory_ids)

        if embeddings:
            clusters, unclustered = _cluster_memories_for_display(
                memories=hot_memories,
                embeddings=embeddings,
                threshold=settings.clustering_display_threshold,
                min_cluster_size=settings.clustering_min_size,
                max_clusters=settings.clustering_max_clusters,
            )

            if clusters:
                return _format_clustered_memory_list(clusters, unclustered, header)

    return _format_memory_list(hot_memories, header)


@mcp.resource("memory://project-context")
def project_context_resource() -> str:
    """Project-specific memory context for the current git repository.

    Returns:
    - Current project ID (from git remote URL)
    - Project-specific hot cache memories
    - Recent project activity

    Use this to understand what memories are associated with the current project.
    """
    project = detect_project()

    if not project:
        return "[MEMORY: No git project detected - memories will be global]"

    # Get project-specific hot memories
    hot_memories = storage.get_promoted_memories(project_id=project.id)

    lines = [
        f"[MEMORY: Project Context - {project.name}]",
        f"Project ID: {project.id}",
        f"Path: {project.path}",
    ]

    if project.remote_url:
        lines.append(f"Remote: {project.remote_url}")

    lines.append(f"Hot memories: {len(hot_memories)}")

    if hot_memories:
        lines.append("")
        lines.append("Recent hot memories:")
        max_chars = settings.hot_cache_display_max_chars
        for m in hot_memories[:5]:  # Show top 5
            content = m.content[:max_chars] + "..." if len(m.content) > max_chars else m.content
            lines.append(f"  - {content}")

    return "\n".join(lines)
