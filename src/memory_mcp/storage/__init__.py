"""Storage package for Memory MCP.

This package provides the Storage class for SQLite-based memory persistence
with vector search, hot cache, trust management, and knowledge graph features.
"""

from memory_mcp.migrations import EmbeddingDimensionError, SchemaVersionError
from memory_mcp.models import (
    TRUST_REASON_DEFAULTS,
    AccessPattern,
    AuditEntry,
    AuditOperation,
    ConsolidationCluster,
    HotCacheMetrics,
    Memory,
    MemoryRelation,
    MemorySource,
    MemoryType,
    MinedPattern,
    PatternStatus,
    PotentialContradiction,
    PredictionResult,
    PromotionSource,
    RecallMode,
    RecallModeConfig,
    RecallResult,
    RelationType,
    RetrievalEvent,
    ScoreBreakdown,
    SemanticMergeResult,
    Session,
    TrustEvent,
    TrustReason,
)
from memory_mcp.storage.core import Storage
from memory_mcp.storage.memory_crud import ValidationError

__all__ = [
    # Enums
    "MemoryType",
    "MemorySource",
    "PromotionSource",
    "RecallMode",
    "TrustReason",
    "RelationType",
    "PatternStatus",
    "AuditOperation",
    # Dataclasses
    "RecallModeConfig",
    "Memory",
    "TrustEvent",
    "MemoryRelation",
    "Session",
    "MinedPattern",
    "ScoreBreakdown",
    "RecallResult",
    "HotCacheMetrics",
    "PotentialContradiction",
    "AccessPattern",
    "PredictionResult",
    "SemanticMergeResult",
    "AuditEntry",
    "RetrievalEvent",
    "ConsolidationCluster",
    # Constants
    "TRUST_REASON_DEFAULTS",
    # Classes
    "Storage",
    "ValidationError",
    "SchemaVersionError",
    "EmbeddingDimensionError",
]
