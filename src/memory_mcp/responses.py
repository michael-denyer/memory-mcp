"""Pydantic response models for MCP tools.

This module contains all Pydantic BaseModel classes used as return types
for MCP tool functions. Extracted from server.py for cleaner organization.
"""

from pydantic import BaseModel, Field

from memory_mcp.models import Memory, MemoryRelation, Session

# ========== Core Response Models ==========


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
    session_id: str | None = None  # Conversation session this came from
    project_id: str | None = None  # Project this memory belongs to (e.g., "github/owner/repo")
    # Computed scores
    similarity: float | None = None
    hot_score: float | None = None
    salience_score: float | None = None  # Unified metric for promotion (Engram-inspired)
    # Recall scoring (populated during recall)
    recency_score: float | None = None
    trust_score_decayed: float | None = None
    composite_score: float | None = None
    created_at: str


class RelationshipResponse(BaseModel):
    """Response for a single memory relationship."""

    id: int
    from_memory_id: int
    to_memory_id: int
    relation_type: str
    created_at: str


class RelatedMemoryResponse(BaseModel):
    """A related memory with its relationship info."""

    memory: MemoryResponse
    relationship: RelationshipResponse


class FormattedMemory(BaseModel):
    """LLM-friendly formatted memory with annotations."""

    summary: str  # Concise one-line summary
    memory_type: str  # project, pattern, reference, conversation
    tags: list[str]
    age: str  # Human-readable age: "2 hours", "3 days", "2 weeks"
    confidence: str  # high, medium, low based on similarity
    source_hint: str  # "hot cache" or "cold storage"


class RecallResponse(BaseModel):
    """Response for recall operation."""

    memories: list[MemoryResponse]
    confidence: str
    gated_count: int
    mode: str
    guidance: str
    # Scoring explanation
    ranking_factors: str
    # LLM-friendly context (use this for quick understanding)
    formatted_context: list[FormattedMemory] | None = None
    context_summary: str | None = None  # One-line summary of what was found
    # Promotion feedback
    promotion_suggestions: list[dict] | None = None
    # Related memories (when include_related=True)
    related_memories: list[RelatedMemoryResponse] | None = None


class StatsResponse(BaseModel):
    """Response for stats operation."""

    total_memories: int
    hot_cache_count: int
    by_type: dict[str, int]
    by_source: dict[str, int]


# ========== Hot Cache Response Models ==========


class HotCacheMetricsResponse(BaseModel):
    """Metrics for hot cache observability."""

    hits: int
    misses: int
    evictions: int
    promotions: int


class HotCacheEffectivenessResponse(BaseModel):
    """Effectiveness metrics showing hot cache value."""

    total_accesses: int  # Sum of access_count for hot items
    estimated_tool_calls_saved: int  # Rough estimate based on hits
    hit_rate_percent: float  # hits / (hits + misses) * 100
    most_accessed_id: int | None  # Most frequently used hot item
    least_accessed_id: int | None  # Candidate for demotion


class HotCacheResponse(BaseModel):
    """Response for hot cache status."""

    items: list[MemoryResponse]
    max_items: int
    current_count: int
    pinned_count: int
    avg_hot_score: float
    metrics: HotCacheMetricsResponse
    effectiveness: HotCacheEffectivenessResponse


# ========== Relationship Response Models ==========


class RelationshipStatsResponse(BaseModel):
    """Statistics about memory relationships."""

    total_relationships: int
    by_type: dict[str, int]
    linked_memories: int


# ========== Session Response Models ==========


class SessionResponse(BaseModel):
    """Response for a conversation session."""

    id: str
    started_at: str
    last_activity_at: str
    topic: str | None
    project_path: str | None
    memory_count: int
    log_count: int


class CrossSessionPatternResponse(BaseModel):
    """Pattern appearing across multiple sessions."""

    content: str
    memory_type: str
    session_count: int
    total_accesses: int
    sessions: list[str]


# ========== Seeding Response Models ==========


class SeedResult(BaseModel):
    """Result from seeding operation."""

    memories_created: int
    memories_skipped: int
    errors: list[str]


class BootstrapResponse(BaseModel):
    """Response for bootstrap operation."""

    success: bool
    files_found: int = 0
    files_processed: int = 0
    memories_created: int = 0
    memories_skipped: int = 0
    hot_cache_promoted: int = 0
    errors: list[str] = Field(default_factory=list)
    message: str = ""


# ========== Trust Response Models ==========


class TrustResponse(BaseModel):
    """Response for trust operations."""

    memory_id: int
    old_trust: float
    new_trust: float
    message: str


class TrustHistoryEntry(BaseModel):
    """A single trust history entry."""

    id: int
    memory_id: int
    reason: str
    old_trust: float
    new_trust: float
    delta: float
    similarity: float | None
    note: str | None
    created_at: str


class TrustHistoryResponse(BaseModel):
    """Response for trust history query."""

    memory_id: int
    entries: list[TrustHistoryEntry]
    current_trust: float
    total_changes: int


# ========== Maintenance Response Models ==========


class MaintenanceResponse(BaseModel):
    """Response for maintenance operation."""

    size_before_bytes: int
    size_after_bytes: int
    bytes_reclaimed: int
    memory_count: int
    vector_count: int
    schema_version: int
    auto_demoted_count: int = 0
    auto_demoted_ids: list[int] = Field(default_factory=list)


class AuditEntryResponse(BaseModel):
    """Single audit log entry."""

    id: int
    operation: str
    target_type: str | None
    target_id: int | None
    details: str | None
    timestamp: str


class AuditHistoryResponse(BaseModel):
    """Audit history response."""

    entries: list[AuditEntryResponse]
    count: int


class VectorRebuildResponse(BaseModel):
    """Response from vector rebuild operation."""

    success: bool
    vectors_cleared: int
    memories_total: int
    memories_embedded: int
    memories_failed: int
    new_dimension: int
    new_model: str
    message: str


# ========== Contradiction Response Models ==========


class ContradictionResponse(BaseModel):
    """Response for a single potential contradiction."""

    memory_a: MemoryResponse
    memory_b: MemoryResponse
    similarity: float
    already_linked: bool


class ContradictionPairResponse(BaseModel):
    """Response for marked contradiction pair."""

    memory_a: MemoryResponse
    memory_b: MemoryResponse
    relationship: RelationshipResponse


# ========== Predictive Cache Response Models ==========


class AccessPatternResponse(BaseModel):
    """Response for access pattern between memories."""

    from_memory_id: int
    to_memory_id: int
    count: int
    probability: float
    last_seen: str


class PredictionResponse(BaseModel):
    """Response for memory prediction."""

    memory: MemoryResponse
    probability: float
    source_memory_id: int


# ========== Converter Functions ==========


def session_to_response(s: Session) -> SessionResponse:
    """Convert Session to response model."""
    return SessionResponse(
        id=s.id,
        started_at=s.started_at.isoformat(),
        last_activity_at=s.last_activity_at.isoformat(),
        topic=s.topic,
        project_path=s.project_path,
        memory_count=s.memory_count,
        log_count=s.log_count,
    )


def relation_to_response(r: MemoryRelation) -> RelationshipResponse:
    """Convert MemoryRelation to response model."""
    return RelationshipResponse(
        id=r.id,
        from_memory_id=r.from_memory_id,
        to_memory_id=r.to_memory_id,
        relation_type=r.relation_type.value,
        created_at=r.created_at.isoformat(),
    )


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
        session_id=m.session_id,
        project_id=m.project_id,
        similarity=m.similarity,
        hot_score=m.hot_score,
        salience_score=m.salience_score,
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
