"""Data models for memory MCP.

This module contains all enums and dataclasses used throughout the memory system.
These are separated from storage.py for cleaner imports and better organization.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# ========== Enums ==========


class MemoryType(str, Enum):
    """Types of memories."""

    PROJECT = "project"  # Project-specific facts
    PATTERN = "pattern"  # Reusable code patterns
    REFERENCE = "reference"  # External docs, API notes
    CONVERSATION = "conversation"  # Facts from discussions
    EPISODIC = "episodic"  # Session-bound context (short-term)


class MemorySource(str, Enum):
    """How memory was created."""

    MANUAL = "manual"  # Explicitly stored by user
    MINED = "mined"  # Extracted from output logs


class PromotionSource(str, Enum):
    """How a memory was promoted to hot cache."""

    MANUAL = "manual"  # Explicitly promoted by user
    AUTO_THRESHOLD = "auto_threshold"  # Auto-promoted based on access count
    MINED_APPROVED = "mined_approved"  # Approved from mining candidates
    PREDICTED = "predicted"  # Pre-warmed based on access pattern prediction
    SESSION_END = "session_end"  # Promoted during session consolidation


class RecallMode(str, Enum):
    """Recall mode presets with different threshold/weight configurations."""

    PRECISION = "precision"  # High threshold, few results, prioritize similarity
    BALANCED = "balanced"  # Default balanced mode
    EXPLORATORY = "exploratory"  # Low threshold, more results, diverse factors


class TrustReason(str, Enum):
    """Reasons for trust adjustments with default boost/penalty values."""

    # Strengthening reasons (positive)
    USED_CORRECTLY = "used_correctly"
    EXPLICITLY_CONFIRMED = "explicitly_confirmed"
    HIGH_SIMILARITY_HIT = "high_similarity_hit"
    CROSS_VALIDATED = "cross_validated"

    # Weakening reasons (negative)
    OUTDATED = "outdated"
    PARTIALLY_INCORRECT = "partially_incorrect"
    FACTUALLY_WRONG = "factually_wrong"
    SUPERSEDED = "superseded"
    LOW_UTILITY = "low_utility"
    CONTRADICTION_RESOLVED = "contradiction_resolved"


class RelationType(str, Enum):
    """Types of relationships between memories."""

    RELATES_TO = "relates_to"  # General association
    DEPENDS_ON = "depends_on"  # Prerequisite knowledge
    SUPERSEDES = "supersedes"  # Newer version replaces older
    REFINES = "refines"  # More specific version
    CONTRADICTS = "contradicts"  # Conflicting information
    ELABORATES = "elaborates"  # Provides more detail
    MENTIONS = "mentions"  # Source content mentions an entity (weaker than elaborates)


class PatternStatus(str, Enum):
    """Status of mined patterns in the approval workflow."""

    PENDING = "pending"  # Awaiting review
    APPROVED = "approved"  # Approved for promotion
    REJECTED = "rejected"  # Rejected (won't be promoted)
    PROMOTED = "promoted"  # Already promoted to memory


class AuditOperation(str, Enum):
    """Destructive operations tracked in audit log."""

    DELETE_MEMORY = "delete_memory"
    DEMOTE_MEMORY = "demote_memory"
    DEMOTE_STALE = "demote_stale"
    DELETE_PATTERN = "delete_pattern"
    EXPIRE_PATTERNS = "expire_patterns"
    CLEANUP_MEMORIES = "cleanup_memories"
    MAINTENANCE = "maintenance"
    UNLINK_MEMORIES = "unlink_memories"


# ========== Constants ==========


# Default boost/penalty amounts per trust reason
TRUST_REASON_DEFAULTS: dict[TrustReason, float] = {
    TrustReason.USED_CORRECTLY: 0.05,
    TrustReason.EXPLICITLY_CONFIRMED: 0.15,
    TrustReason.HIGH_SIMILARITY_HIT: 0.03,
    TrustReason.CROSS_VALIDATED: 0.20,
    TrustReason.OUTDATED: -0.10,
    TrustReason.PARTIALLY_INCORRECT: -0.15,
    TrustReason.FACTUALLY_WRONG: -0.30,
    TrustReason.SUPERSEDED: -0.05,
    TrustReason.LOW_UTILITY: -0.03,
    TrustReason.CONTRADICTION_RESOLVED: -0.20,
}


# ========== Dataclasses ==========


@dataclass
class RecallModeConfig:
    """Configuration for a recall mode preset."""

    threshold: float
    limit: int
    similarity_weight: float
    recency_weight: float
    access_weight: float


@dataclass
class Memory:
    """A stored memory."""

    id: int
    content: str
    content_hash: str
    memory_type: MemoryType
    source: MemorySource
    is_hot: bool
    is_pinned: bool
    promotion_source: PromotionSource | None
    tags: list[str]
    access_count: int
    last_accessed_at: datetime | None
    created_at: datetime
    # Trust and provenance
    trust_score: float = 1.0  # Base trust (decays over time)
    importance_score: float = 0.5  # Admission-time importance (MemGPT-inspired)
    source_log_id: int | None = None  # For mined memories: originating log
    extracted_at: datetime | None = None  # When pattern was extracted
    session_id: str | None = None  # Conversation session this came from
    project_id: str | None = None  # Project this memory belongs to (e.g., "github/owner/repo")
    category: str | None = (
        None  # Subcategory within type (e.g., "decision", "architecture", "import")
    )
    # Helpfulness tracking (feedback loop)
    retrieved_count: int = 0  # Times returned in recall results
    used_count: int = 0  # Times marked as actually helpful
    last_used_at: datetime | None = None  # When last marked as used
    utility_score: float = 0.25  # Precomputed Bayesian helpfulness (α=1, β=3 prior)
    # Computed scores (populated during search/recall)
    similarity: float | None = None  # Populated during search
    hot_score: float | None = None  # Computed score for LRU ranking
    salience_score: float | None = None  # Unified metric for promotion/eviction (Engram-inspired)
    # Recall scoring components (populated during recall)
    recency_score: float | None = None  # 0-1 based on age with decay
    trust_score_decayed: float | None = None  # Trust with time decay applied
    composite_score: float | None = None  # Combined ranking score
    # Weighted component breakdown (for debugging/transparency)
    similarity_component: float | None = None  # similarity * weight
    recency_component: float | None = None  # recency_score * weight
    access_component: float | None = None  # access_score * weight
    trust_component: float | None = None  # trust * weight
    helpfulness_component: float | None = None  # utility_score * weight
    keyword_score: float | None = None  # FTS keyword match score (hybrid search)
    intent_boost: float | None = None  # Intent-based category boost (query intent matching)


@dataclass
class TrustEvent:
    """Record of a trust score change for audit trail."""

    id: int
    memory_id: int
    reason: TrustReason
    old_trust: float
    new_trust: float
    delta: float  # new_trust - old_trust
    similarity: float | None  # For confidence-weighted updates
    note: str | None  # Optional human-readable note
    created_at: datetime


@dataclass
class MemoryRelation:
    """A relationship between two memories."""

    id: int
    from_memory_id: int
    to_memory_id: int
    relation_type: RelationType
    created_at: datetime


@dataclass
class Session:
    """A conversation session for provenance tracking."""

    id: str  # UUID or transcript path hash
    started_at: datetime
    last_activity_at: datetime
    topic: str | None  # Auto-detected or user-provided
    project_path: str | None  # Working directory
    memory_count: int
    log_count: int


@dataclass
class MinedPattern:
    """A pattern extracted from output logs."""

    id: int
    pattern: str
    pattern_hash: str
    pattern_type: str
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    status: PatternStatus = PatternStatus.PENDING
    source_log_id: int | None = None  # Originating output_log ID
    confidence: float = 0.5  # Extraction confidence (0-1)
    score: float = 0.0  # Computed promotion score
    memory_id: int | None = None  # Linked memory ID for exact-match promotion


@dataclass
class ScoreBreakdown:
    """Weighted component breakdown for transparency."""

    total: float  # Combined composite score
    similarity_component: float  # similarity * weight
    recency_component: float  # recency_score * weight
    access_component: float  # access_score * weight
    trust_component: float  # trust * weight
    helpfulness_component: float = 0.0  # utility_score * weight


@dataclass
class RecallResult:
    """Result from recall operation with confidence gating."""

    memories: list[Memory]
    confidence: str  # "high", "medium", "low"
    gated_count: int  # How many results filtered by threshold
    mode: RecallMode | None = None  # Mode used for this recall
    guidance: str | None = None  # Hallucination prevention guidance


@dataclass
class HotCacheMetrics:
    """Metrics for hot cache observability."""

    hits: int = 0  # Times hot cache resource was read with content
    misses: int = 0  # Recalls that returned no hot cache results
    evictions: int = 0  # Items removed to make space for new ones
    promotions: int = 0  # Items added to hot cache

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "promotions": self.promotions,
        }


@dataclass
class PotentialContradiction:
    """A pair of memories that may contain conflicting information."""

    memory_a: Memory
    memory_b: Memory
    similarity: float  # High similarity suggests same topic
    already_linked: bool  # Whether contradiction relationship exists


@dataclass
class AccessPattern:
    """A learned access pattern between memories."""

    from_memory_id: int
    to_memory_id: int
    count: int
    probability: float  # Transition probability
    last_seen: datetime


@dataclass
class PredictionResult:
    """A predicted memory that may be needed next."""

    memory: Memory
    probability: float
    source_memory_id: int  # Which memory triggered this prediction


@dataclass
class SemanticMergeResult:
    """Result of semantic deduplication during store."""

    memory_id: int
    merged: bool  # True if merged with existing memory
    merged_with_id: int | None  # ID of memory merged into (if merged)
    similarity: float | None  # Similarity score (if merged)
    content_updated: bool  # True if content was updated (longer/richer)


@dataclass
class AuditEntry:
    """An entry in the audit log for a destructive operation."""

    id: int
    operation: str
    target_type: str | None
    target_id: int | None
    details: str | None
    timestamp: str


@dataclass
class RetrievalEvent:
    """A record of a memory being retrieved (RAG-inspired tracking)."""

    id: int
    query_hash: str  # Hash of the query that retrieved this
    memory_id: int
    similarity: float
    was_used: bool  # Whether the LLM actually used this memory
    feedback: str | None  # Optional feedback (helpful/not helpful)
    created_at: datetime


@dataclass
class ConsolidationCluster:
    """A cluster of similar memories for consolidation."""

    representative_id: int  # Best memory to keep as representative
    member_ids: list[int]  # All memories in this cluster
    avg_similarity: float  # Average pairwise similarity
    total_access_count: int  # Sum of access counts
    combined_tags: list[str]  # Union of all tags


@dataclass
class DisplayCluster:
    """A semantic cluster for display purposes (RePo-inspired).

    Used to group semantically similar memories in hot cache and recall
    results to reduce cognitive load.
    """

    label: str  # Human-readable label (e.g., "Python Development")
    members: list[Memory]  # Memories in this cluster (sorted by score)
    avg_similarity: float  # Average pairwise similarity within cluster

    @property
    def size(self) -> int:
        """Number of memories in this cluster."""
        return len(self.members)
