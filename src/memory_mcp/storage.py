"""SQLite storage with sqlite-vec for vector search."""

import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
import sqlite_vec

from memory_mcp.config import Settings, ensure_data_dir, get_settings
from memory_mcp.embeddings import EmbeddingEngine, content_hash
from memory_mcp.logging import get_logger

log = get_logger("storage")


class MemoryType(str, Enum):
    """Types of memories."""

    PROJECT = "project"  # Project-specific facts
    PATTERN = "pattern"  # Reusable code patterns
    REFERENCE = "reference"  # External docs, API notes
    CONVERSATION = "conversation"  # Facts from discussions


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


class PatternStatus(str, Enum):
    """Status of mined patterns in the approval workflow."""

    PENDING = "pending"  # Awaiting review
    APPROVED = "approved"  # Approved for promotion
    REJECTED = "rejected"  # Rejected (won't be promoted)
    PROMOTED = "promoted"  # Already promoted to memory


# Default boost/penalty amounts per reason
TRUST_REASON_DEFAULTS: dict[TrustReason, float] = {
    TrustReason.USED_CORRECTLY: 0.05,
    TrustReason.EXPLICITLY_CONFIRMED: 0.15,
    TrustReason.HIGH_SIMILARITY_HIT: 0.03,
    TrustReason.CROSS_VALIDATED: 0.20,
    TrustReason.OUTDATED: -0.10,
    TrustReason.PARTIALLY_INCORRECT: -0.15,
    TrustReason.FACTUALLY_WRONG: -0.30,
    TrustReason.SUPERSEDED: -0.05,
    TrustReason.LOW_UTILITY: -0.05,
    TrustReason.CONTRADICTION_RESOLVED: -0.20,
}


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
    source_log_id: int | None = None  # For mined memories: originating log
    extracted_at: datetime | None = None  # When pattern was extracted
    session_id: str | None = None  # Conversation session this came from
    # Computed scores (populated during search/recall)
    similarity: float | None = None  # Populated during search
    hot_score: float | None = None  # Computed score for LRU ranking
    # Recall scoring components (populated during recall)
    recency_score: float | None = None  # 0-1 based on age with decay
    trust_score_decayed: float | None = None  # Trust with time decay applied
    composite_score: float | None = None  # Combined ranking score
    # Weighted component breakdown (for debugging/transparency)
    similarity_component: float | None = None  # similarity * weight
    recency_component: float | None = None  # recency_score * weight
    access_component: float | None = None  # access_score * weight
    trust_component: float | None = None  # trust * weight


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


@dataclass
class ScoreBreakdown:
    """Weighted component breakdown for transparency."""

    total: float  # Combined composite score
    similarity_component: float  # similarity * weight
    recency_component: float  # recency_score * weight
    access_component: float  # access_score * weight
    trust_component: float  # trust * weight


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


# Current schema version - increment when making breaking changes
SCHEMA_VERSION = 8

SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- All memories (hot + cold)
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE,
    memory_type TEXT NOT NULL,
    source TEXT NOT NULL,
    is_hot INTEGER DEFAULT 0,
    is_pinned INTEGER DEFAULT 0,
    promotion_source TEXT,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Tags
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);

-- Output log (7-day rolling)
CREATE TABLE IF NOT EXISTS output_log (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Mined patterns (candidates for promotion)
CREATE TABLE IF NOT EXISTS mined_patterns (
    id INTEGER PRIMARY KEY,
    pattern TEXT NOT NULL,
    pattern_hash TEXT UNIQUE,
    pattern_type TEXT NOT NULL,
    occurrence_count INTEGER DEFAULT 1,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Trust history (audit trail for trust changes)
CREATE TABLE IF NOT EXISTS trust_history (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    old_trust REAL NOT NULL,
    new_trust REAL NOT NULL,
    delta REAL NOT NULL,
    similarity REAL,
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Memory relationships (knowledge graph edges)
CREATE TABLE IF NOT EXISTS memory_relationships (
    id INTEGER PRIMARY KEY,
    from_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_memory_id, to_memory_id, relation_type)
);

-- Sessions (conversation provenance tracking)
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,  -- UUID or transcript path hash
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TEXT DEFAULT CURRENT_TIMESTAMP,
    topic TEXT,  -- Auto-detected or user-provided topic
    project_path TEXT,  -- Working directory path
    memory_count INTEGER DEFAULT 0,
    log_count INTEGER DEFAULT 0
);

-- Key-value metadata (for embedding model tracking, etc.)
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_hot ON memories(is_hot);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_output_log_timestamp ON output_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_mined_patterns_hash ON mined_patterns(pattern_hash);
CREATE INDEX IF NOT EXISTS idx_trust_history_memory ON trust_history(memory_id);
CREATE INDEX IF NOT EXISTS idx_trust_history_reason ON trust_history(reason);
CREATE INDEX IF NOT EXISTS idx_relationships_from ON memory_relationships(from_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON memory_relationships(to_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON memory_relationships(relation_type);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity_at);
"""


def get_vector_schema(dim: int) -> str:
    """Generate vector schema with correct dimension."""
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
    embedding FLOAT[{dim}]
);
"""


class SchemaVersionError(Exception):
    """Raised when database schema is incompatible."""

    pass


class EmbeddingDimensionError(Exception):
    """Raised when embedding dimension doesn't match database."""

    pass


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


class Storage:
    """SQLite storage manager with thread-safe connection handling."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        # Use settings-aware embedding engine, not global singleton
        self._embedding_engine = EmbeddingEngine(self.settings)
        self._hot_cache_metrics = HotCacheMetrics()
        log.info("Storage initialized with db_path={}", self.settings.db_path)

    @property
    def db_path(self) -> Path:
        return self.settings.db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection. Must be called with lock held."""
        if self._conn is None:
            ensure_data_dir(self.settings)
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,  # Wait up to 30s for locks
            )
            self._conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
            self._conn.execute("PRAGMA foreign_keys=ON")  # Enable cascade deletes

            # Load sqlite-vec extension
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

            # Initialize schema
            self._init_schema()
        return self._conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for read operations with thread safety."""
        with self._lock:
            yield self._get_connection()

    def _init_schema(self) -> None:
        """Initialize database schema with version tracking."""
        conn = self._conn
        if conn is None:
            return

        # Check existing schema version before applying migrations
        self._check_schema_version(conn)

        # Apply base schema first (uses IF NOT EXISTS, safe to re-run)
        conn.executescript(SCHEMA)
        conn.execute(get_vector_schema(self.settings.embedding_dim))

        # Get current version for migrations (now table exists)
        existing_version = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        current_version = existing_version[0] if existing_version else 0

        # Run migrations
        self._run_migrations(conn, current_version)

        # Record new schema version
        if current_version < SCHEMA_VERSION:
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            log.info("Database migrated from v{} to v{}", current_version, SCHEMA_VERSION)

        conn.commit()

        # Validate embedding dimension (fail fast, not just warn)
        self._validate_vector_dimension(conn)
        log.debug("Database schema initialized (version={})", SCHEMA_VERSION)

    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations from from_version to SCHEMA_VERSION."""
        if from_version < 2:
            self._migrate_v1_to_v2(conn)
        if from_version < 3:
            self._migrate_v2_to_v3(conn)
        if from_version < 4:
            self._migrate_v3_to_v4(conn)
        if from_version < 5:
            self._migrate_v4_to_v5(conn)
        if from_version < 6:
            self._migrate_v5_to_v6(conn)
        if from_version < 7:
            self._migrate_v6_to_v7(conn)
        if from_version < 8:
            self._migrate_v7_to_v8(conn)

    def _get_table_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        """Get set of column names for a table."""
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _add_column_if_missing(
        self, conn: sqlite3.Connection, table: str, column: str, definition: str
    ) -> bool:
        """Add column if it doesn't exist. Returns True if added."""
        columns = self._get_table_columns(conn, table)
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            log.debug("Added {} column to {} table", column, table)
            return True
        return False

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Add is_pinned and promotion_source columns for hot cache LRU."""
        self._add_column_if_missing(conn, "memories", "is_pinned", "INTEGER DEFAULT 0")
        self._add_column_if_missing(conn, "memories", "promotion_source", "TEXT")

    def _migrate_v2_to_v3(self, conn: sqlite3.Connection) -> None:
        """Add trust_score and provenance columns."""
        self._add_column_if_missing(conn, "memories", "trust_score", "REAL DEFAULT 1.0")
        self._add_column_if_missing(conn, "memories", "source_log_id", "INTEGER")
        self._add_column_if_missing(conn, "memories", "extracted_at", "TEXT")

        # Set initial trust scores based on source type
        conn.execute(
            """
            UPDATE memories
            SET trust_score = CASE
                WHEN source = 'manual' THEN ?
                WHEN source = 'mined' THEN ?
                ELSE 1.0
            END
            WHERE trust_score IS NULL OR trust_score = 1.0
            """,
            (self.settings.trust_score_manual, self.settings.trust_score_mined),
        )

    def _migrate_v3_to_v4(self, conn: sqlite3.Connection) -> None:
        """Add trust_history table for audit trail."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trust_history (
                id INTEGER PRIMARY KEY,
                memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
                reason TEXT NOT NULL,
                old_trust REAL NOT NULL,
                new_trust REAL NOT NULL,
                delta REAL NOT NULL,
                similarity REAL,
                note TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trust_history_memory ON trust_history(memory_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trust_history_reason ON trust_history(reason)")
        log.info("Created trust_history table for audit trail")

    def _migrate_v4_to_v5(self, conn: sqlite3.Connection) -> None:
        """Add memory_relationships table for knowledge graph."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_relationships (
                id INTEGER PRIMARY KEY,
                from_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                to_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(from_memory_id, to_memory_id, relation_type)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_from "
            "ON memory_relationships(from_memory_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_to ON memory_relationships(to_memory_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_type "
            "ON memory_relationships(relation_type)"
        )
        log.info("Created memory_relationships table for knowledge graph")

    def _migrate_v5_to_v6(self, conn: sqlite3.Connection) -> None:
        """Add sessions table and session_id columns for conversation provenance."""
        # Create sessions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_activity_at TEXT DEFAULT CURRENT_TIMESTAMP,
                topic TEXT,
                project_path TEXT,
                memory_count INTEGER DEFAULT 0,
                log_count INTEGER DEFAULT 0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity_at)"
        )

        # Add session_id to memories table
        self._add_column_if_missing(conn, "memories", "session_id", "TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")

        # Add session_id to output_log table
        self._add_column_if_missing(conn, "output_log", "session_id", "TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_output_log_session ON output_log(session_id)")

        log.info("Created sessions table and session tracking columns")

    def _migrate_v6_to_v7(self, conn: sqlite3.Connection) -> None:
        """Add access_sequences table for predictive hot cache warming."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS access_sequences (
                from_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                to_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                count INTEGER DEFAULT 1,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_memory_id, to_memory_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sequences_from ON access_sequences(from_memory_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sequences_last ON access_sequences(last_seen)")
        log.info("Created access_sequences table for predictive cache")

    def _migrate_v7_to_v8(self, conn: sqlite3.Connection) -> None:
        """Enhance mined_patterns table for mining quality improvements."""
        # Add status column for approval workflow
        self._add_column_if_missing(conn, "mined_patterns", "status", "TEXT DEFAULT 'pending'")
        # Add source_log_id for provenance tracking
        self._add_column_if_missing(
            conn, "mined_patterns", "source_log_id", "INTEGER REFERENCES output_log(id)"
        )
        # Add confidence score from extraction
        self._add_column_if_missing(conn, "mined_patterns", "confidence", "REAL DEFAULT 0.5")
        # Add computed promotion score
        self._add_column_if_missing(conn, "mined_patterns", "score", "REAL DEFAULT 0.0")
        # Index for filtering by status
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mined_patterns_status ON mined_patterns(status)"
        )
        log.info("Enhanced mined_patterns table with status, provenance, and scoring")

    def _check_schema_version(self, conn: sqlite3.Connection) -> None:
        """Check schema version compatibility."""
        # Check if schema_version table exists
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()

        if not table_exists:
            # New database or pre-versioning database
            # Check if other tables exist (pre-versioning database)
            memories_exist = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
            if memories_exist:
                log.info("Upgrading pre-versioning database to version {}", SCHEMA_VERSION)
            return

        # Get current version
        current_version = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()

        if current_version and current_version[0] > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Database schema version {current_version[0]} is newer than "
                f"supported version {SCHEMA_VERSION}. Please upgrade memory-mcp."
            )

    def _validate_vector_dimension(self, conn: sqlite3.Connection) -> None:
        """Check that existing vector table matches configured dimension. Fails fast on mismatch."""
        result = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_vectors'"
        ).fetchone()
        if result and result[0]:
            schema_sql = result[0]
            expected_dim = self.settings.embedding_dim

            # Check if dimension in schema matches
            if f"FLOAT[{expected_dim}]" not in schema_sql.upper():
                # Check if there are any existing vectors
                count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]

                if count > 0:
                    raise EmbeddingDimensionError(
                        f"Embedding dimension mismatch: database has vectors with different "
                        f"dimension than configured ({expected_dim}). "
                        f"Delete the database or set MEMORY_MCP_EMBEDDING_DIM to match. "
                        f"Database: {self.db_path}"
                    )
                else:
                    log.warning(
                        "Vector table dimension mismatch but no data. Consider recreating: {}",
                        self.db_path,
                    )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for transactions with thread safety."""
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_schema_version(self) -> int:
        """Get current database schema version."""
        with self._connection() as conn:
            result = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return result[0] if result else 0

    def vacuum(self) -> None:
        """Compact the database, reclaiming unused space."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("VACUUM")
            log.info("Database vacuumed")

    def analyze(self) -> None:
        """Update query planner statistics for better performance."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("ANALYZE")
            log.info("Database analyzed")

    def maintenance(self) -> dict:
        """Run full maintenance: vacuum, analyze, and auto-demote stale hot memories.

        Returns stats including demoted count.
        """
        with self._connection() as conn:
            # Get size before
            size_before = os.path.getsize(self.db_path) if self.db_path.exists() else 0

        # Auto-demote stale hot memories (if enabled)
        demoted_ids = self.demote_stale_hot_memories()

        self.vacuum()
        self.analyze()

        with self._connection() as conn:
            size_after = os.path.getsize(self.db_path) if self.db_path.exists() else 0
            memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            vector_count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]

        return {
            "size_before_bytes": size_before,
            "size_after_bytes": size_after,
            "bytes_reclaimed": size_before - size_after,
            "memory_count": memory_count,
            "vector_count": vector_count,
            "schema_version": self.get_schema_version(),
            "auto_demoted_count": len(demoted_ids),
            "auto_demoted_ids": demoted_ids,
        }

    def _get_retention_days(self, memory_type: MemoryType) -> int:
        """Get retention days for a memory type (0 = never expire)."""
        retention_map = {
            MemoryType.PROJECT: self.settings.retention_project_days,
            MemoryType.PATTERN: self.settings.retention_pattern_days,
            MemoryType.REFERENCE: self.settings.retention_reference_days,
            MemoryType.CONVERSATION: self.settings.retention_conversation_days,
        }
        return retention_map.get(memory_type, 0)

    def cleanup_stale_memories(self) -> dict:
        """Delete memories that exceed their type-specific retention period.

        Only deletes memories that:
        - Have a non-zero retention policy
        - Haven't been accessed within the retention period
        - Are not in hot cache

        Returns dict with counts per type and total deleted.
        """
        deleted_counts: dict[str, int] = {}

        for mem_type in MemoryType:
            retention_days = self._get_retention_days(mem_type)
            if retention_days == 0:
                continue  # Never expire

            cutoff = f"-{retention_days} days"

            with self.transaction() as conn:
                # Find stale memories of this type
                rows = conn.execute(
                    """
                    SELECT id FROM memories
                    WHERE memory_type = ?
                      AND is_hot = 0
                      AND (last_accessed_at IS NULL
                           OR last_accessed_at < datetime('now', ?))
                      AND created_at < datetime('now', ?)
                    """,
                    (mem_type.value, cutoff, cutoff),
                ).fetchall()

                stale_ids = [row["id"] for row in rows]

            # Delete outside transaction to avoid long locks
            for memory_id in stale_ids:
                self.delete_memory(memory_id)

            if stale_ids:
                deleted_counts[mem_type.value] = len(stale_ids)
                log.info(
                    "Cleaned up {} stale {} memories (retention: {} days)",
                    len(stale_ids),
                    mem_type.value,
                    retention_days,
                )

        return {
            "deleted_by_type": deleted_counts,
            "total_deleted": sum(deleted_counts.values()),
        }

    def cleanup_old_logs(self) -> int:
        """Delete output logs older than log_retention_days.

        Returns count of deleted logs.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM output_log WHERE timestamp < datetime('now', ?)",
                (f"-{self.settings.log_retention_days} days",),
            )
            deleted = cursor.rowcount

        if deleted > 0:
            log.info(
                "Cleaned up {} old output logs (retention: {} days)",
                deleted,
                self.settings.log_retention_days,
            )
        return deleted

    def get_embedding_model_info(self) -> dict:
        """Get stored embedding model info for validation."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM metadata WHERE key IN ('embedding_model', 'embedding_dim')"
            ).fetchall()

        info = {row["key"]: row["value"] for row in rows}
        return {
            "model": info.get("embedding_model"),
            "dimension": int(info["embedding_dim"]) if "embedding_dim" in info else None,
        }

    def set_embedding_model_info(self, model: str, dimension: int) -> None:
        """Store embedding model info for future validation."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('embedding_model', ?)
                """,
                (model,),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('embedding_dim', ?)
                """,
                (str(dimension),),
            )

    def validate_embedding_model(self, current_model: str, current_dim: int) -> dict:
        """Check if embedding model has changed since last use.

        Returns validation result with mismatch details if any.
        """
        stored = self.get_embedding_model_info()

        if stored["model"] is None:
            # First time - store current model info
            self.set_embedding_model_info(current_model, current_dim)
            return {
                "valid": True,
                "first_run": True,
                "model": current_model,
                "dimension": current_dim,
            }

        model_match = stored["model"] == current_model
        dim_match = stored["dimension"] == current_dim

        if model_match and dim_match:
            return {"valid": True, "model": current_model, "dimension": current_dim}

        return {
            "valid": False,
            "stored_model": stored["model"],
            "stored_dimension": stored["dimension"],
            "current_model": current_model,
            "current_dimension": current_dim,
            "model_changed": not model_match,
            "dimension_changed": not dim_match,
        }

    def run_full_cleanup(self) -> dict:
        """Run comprehensive cleanup: stale memories, old logs, patterns.

        Orchestrates all maintenance tasks in one call.

        Returns combined stats from all cleanup operations.
        """
        # 1. Demote stale hot memories
        demoted_ids = self.demote_stale_hot_memories()

        # 2. Expire stale mining patterns
        expired_patterns = self.expire_stale_patterns(days=30)

        # 3. Clean up old output logs
        deleted_logs = self.cleanup_old_logs()

        # 4. Clean up stale memories by retention policy
        memory_cleanup = self.cleanup_stale_memories()

        # 5. Decay access sequences (for predictive cache)
        if self.settings.predictive_cache_enabled:
            self.decay_old_sequences()

        return {
            "hot_cache_demoted": len(demoted_ids),
            "patterns_expired": expired_patterns,
            "logs_deleted": deleted_logs,
            "memories_deleted": memory_cleanup["total_deleted"],
            "memories_deleted_by_type": memory_cleanup["deleted_by_type"],
        }

    # ========== Memory CRUD ==========

    def _validate_content(self, content: str, field_name: str = "content") -> None:
        """Validate content length against settings.

        Raises:
            ValidationError: If content is empty or exceeds max length.
        """
        if not content or not content.strip():
            raise ValidationError(f"{field_name} cannot be empty")
        if len(content) > self.settings.max_content_length:
            raise ValidationError(
                f"{field_name} too long: {len(content)} chars "
                f"(max: {self.settings.max_content_length})"
            )

    def _validate_tags(self, tags: list[str] | None) -> list[str]:
        """Validate and normalize tags.

        Returns:
            Normalized tag list (stripped, non-empty).

        Raises:
            ValidationError: If too many tags provided.
        """
        if not tags:
            return []
        # Strip whitespace and filter empty
        normalized = [t.strip() for t in tags if t and t.strip()]
        if len(normalized) > self.settings.max_tags:
            raise ValidationError(
                f"Too many tags: {len(normalized)} (max: {self.settings.max_tags})"
            )
        return normalized

    def _find_semantic_duplicate(
        self,
        conn: sqlite3.Connection,
        embedding: np.ndarray,
        threshold: float,
    ) -> tuple[int, float] | None:
        """Find the most similar existing memory above threshold.

        Args:
            conn: Database connection
            embedding: Embedding of new content
            threshold: Minimum similarity to consider a duplicate

        Returns:
            Tuple of (memory_id, similarity) if found, None otherwise.
        """
        row = conn.execute(
            """
            SELECT m.id, vec_distance_cosine(v.embedding, ?) as distance
            FROM memory_vectors v
            JOIN memories m ON m.id = v.rowid
            ORDER BY distance ASC
            LIMIT 1
            """,
            (embedding.tobytes(),),
        ).fetchone()

        if row is None:
            return None

        similarity = 1 - row["distance"]
        if similarity >= threshold:
            return row["id"], similarity
        return None

    def _get_memory_by_id(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
    ) -> Memory | None:
        """Get a memory by ID using an existing connection."""
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row, conn)

    def _merge_with_existing(
        self,
        conn: sqlite3.Connection,
        new_content: str,
        existing: Memory,
        new_tags: list[str],
        similarity: float,
    ) -> tuple[int, bool]:
        """Merge new content with an existing similar memory.

        Strategy:
        - If new content is longer/richer: update content and embedding
        - Always merge tags
        - Increment access count

        Returns:
            Tuple of (memory_id, False) - False because not a new memory.
        """
        new_is_longer = len(new_content) > len(existing.content)

        if new_is_longer:
            # New content is richer - update content and embedding
            new_embedding = self._embedding_engine.embed(new_content)
            new_hash = content_hash(new_content)
            conn.execute(
                """
                UPDATE memories
                SET content = ?,
                    content_hash = ?,
                    access_count = access_count + 1,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_content, new_hash, existing.id),
            )
            conn.execute(
                "UPDATE memory_vectors SET embedding = ? WHERE rowid = ?",
                (new_embedding.tobytes(), existing.id),
            )
            log.info(
                "Merged memory #{}: updated content (similarity={:.2f}, {} -> {} chars)",
                existing.id,
                similarity,
                len(existing.content),
                len(new_content),
            )
        else:
            # Existing content is richer - just increment access
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (existing.id,),
            )
            log.info(
                "Merged memory #{}: kept existing content (similarity={:.2f})",
                existing.id,
                similarity,
            )

        # Merge tags
        if new_tags:
            conn.executemany(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                [(existing.id, tag) for tag in new_tags],
            )

        return existing.id, False

    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source: MemorySource = MemorySource.MANUAL,
        tags: list[str] | None = None,
        is_hot: bool = False,
        source_log_id: int | None = None,
        session_id: str | None = None,
    ) -> tuple[int, bool]:
        """Store a new memory with embedding.

        Args:
            content: The memory content
            memory_type: Type of memory (project, pattern, etc.)
            source: How memory was created (manual, mined)
            tags: Optional tags for categorization
            is_hot: Whether to add to hot cache immediately
            source_log_id: For mined memories, the originating output_log ID
            session_id: Conversation session ID for provenance tracking

        Returns:
            Tuple of (memory_id, is_new) where is_new indicates if a new
            memory was created (False means existing memory was updated).

        Raises:
            ValidationError: If content too long or too many tags.

        On duplicate content:
            - Increments access_count and updates last_accessed_at
            - Merges new tags with existing tags
            - Does NOT change memory_type or source (first write wins)
        """
        # Defense-in-depth validation
        self._validate_content(content)
        tags = self._validate_tags(tags)

        hash_val = content_hash(content)

        # Trust score based on source type
        trust_scores = {
            MemorySource.MANUAL: self.settings.trust_score_manual,
            MemorySource.MINED: self.settings.trust_score_mined,
        }
        trust_score = trust_scores.get(source, self.settings.trust_score_mined)

        with self.transaction() as conn:
            # Check if memory already exists
            existing = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (hash_val,)
            ).fetchone()

            if existing:
                # Update existing memory - merge tags, increment access
                memory_id = existing[0]
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (memory_id,),
                )

                # Merge new tags (INSERT OR IGNORE handles duplicates)
                if tags:
                    conn.executemany(
                        "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                        [(memory_id, tag) for tag in tags],
                    )
                    log.debug(
                        "Updated existing memory id={} (merged {} tags)",
                        memory_id,
                        len(tags),
                    )
                else:
                    log.debug("Updated existing memory id={}", memory_id)

                return memory_id, False

            # Compute embedding for new content (needed for dedup check and insert)
            embedding = self._embedding_engine.embed(content)

            # Semantic deduplication: check for very similar existing memories
            if self.settings.semantic_dedup_enabled:
                similar = self._find_semantic_duplicate(
                    conn, embedding, self.settings.semantic_dedup_threshold
                )
                if similar:
                    existing_id, similarity = similar
                    existing_memory = self._get_memory_by_id(conn, existing_id)
                    if existing_memory:
                        return self._merge_with_existing(
                            conn, content, existing_memory, tags, similarity
                        )

            # No duplicate found - insert new memory
            extracted_at = datetime.now().isoformat() if source == MemorySource.MINED else None
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    content, content_hash, memory_type, source, is_hot,
                    trust_score, source_log_id, extracted_at, session_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    content,
                    hash_val,
                    memory_type.value,
                    source.value,
                    int(is_hot),
                    trust_score,
                    source_log_id,
                    extracted_at,
                    session_id,
                ),
            )
            memory_id = cursor.fetchone()[0]

            # Update session memory count if session provided
            if session_id:
                self._update_session_activity(conn, session_id, memory_delta=1)

            # Insert embedding only for new memories
            conn.execute(
                "INSERT INTO memory_vectors (rowid, embedding) VALUES (?, ?)",
                (memory_id, embedding.tobytes()),
            )

            # Insert tags only for new memories
            if tags:
                conn.executemany(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    [(memory_id, tag) for tag in tags],
                )

            log.info(
                "Stored new memory id={} type={} trust={}",
                memory_id,
                memory_type.value,
                trust_score,
            )

            return memory_id, True

    def _compute_hot_score(self, access_count: int, last_accessed_at: datetime | None) -> float:
        """Compute hot cache score for LRU ranking.

        Score = (access_count * access_weight) + (recency_boost * recency_weight)

        Recency boost uses exponential decay with configurable half-life.
        """
        access_weight = self.settings.hot_score_access_weight
        recency_weight = self.settings.hot_score_recency_weight
        halflife_days = self.settings.hot_score_recency_halflife_days

        # Access count component
        access_score = access_count * access_weight

        # Recency component with exponential decay
        recency_boost = 0.0
        if last_accessed_at:
            days_since_access = (datetime.now() - last_accessed_at).total_seconds() / 86400
            # Exponential decay: 1.0 at time 0, 0.5 at half-life, approaches 0
            recency_boost = 2 ** (-days_since_access / halflife_days) * recency_weight

        return access_score + recency_boost

    def _get_row_value(self, row: sqlite3.Row, column: str, default=None):
        """Get column value from row, returning default if column doesn't exist."""
        return row[column] if column in row.keys() else default

    def _row_to_memory(
        self,
        row: sqlite3.Row,
        conn: sqlite3.Connection,
        tags: list[str] | None = None,
        similarity: float | None = None,
    ) -> Memory:
        """Convert a database row to a Memory object. Must be called with lock held."""
        if tags is None:
            tags = [
                r["tag"]
                for r in conn.execute(
                    "SELECT tag FROM memory_tags WHERE memory_id = ?", (row["id"],)
                )
            ]

        last_accessed = row["last_accessed_at"]
        last_accessed_dt = datetime.fromisoformat(last_accessed) if last_accessed else None

        # Parse optional columns (may not exist before migrations)
        promo_source_str = self._get_row_value(row, "promotion_source")
        promotion_source = PromotionSource(promo_source_str) if promo_source_str else None

        is_pinned = bool(self._get_row_value(row, "is_pinned", False))
        trust_score_val = self._get_row_value(row, "trust_score", 1.0)
        trust_score = trust_score_val if trust_score_val is not None else 1.0
        source_log_id = self._get_row_value(row, "source_log_id")
        extracted_at_str = self._get_row_value(row, "extracted_at")
        extracted_at = datetime.fromisoformat(extracted_at_str) if extracted_at_str else None
        session_id = self._get_row_value(row, "session_id")

        hot_score = self._compute_hot_score(row["access_count"], last_accessed_dt)

        return Memory(
            id=row["id"],
            content=row["content"],
            content_hash=row["content_hash"],
            memory_type=MemoryType(row["memory_type"]),
            source=MemorySource(row["source"]),
            is_hot=bool(row["is_hot"]),
            is_pinned=is_pinned,
            promotion_source=promotion_source,
            tags=tags,
            access_count=row["access_count"],
            last_accessed_at=last_accessed_dt,
            created_at=datetime.fromisoformat(row["created_at"]),
            trust_score=trust_score,
            source_log_id=source_log_id,
            extracted_at=extracted_at,
            session_id=session_id,
            similarity=similarity,
            hot_score=hot_score,
        )

    def get_memory(self, memory_id: int) -> Memory | None:
        """Get a memory by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not row:
                return None
            return self._row_to_memory(row, conn)

    def get_memory_by_hash(self, hash_val: str) -> Memory | None:
        """O(1) exact lookup by content hash."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (hash_val,)
            ).fetchone()
            if not row:
                return None
            # Re-fetch with full data using get_memory (RLock allows nested calls)
            return self.get_memory(row["id"])

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (memory_id,))
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                log.info("Deleted memory id={}", memory_id)
            return deleted

    def update_access(self, memory_id: int) -> None:
        """Update access count and timestamp."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (memory_id,),
            )

    def adjust_trust(
        self,
        memory_id: int,
        reason: TrustReason,
        delta: float | None = None,
        similarity: float | None = None,
        note: str | None = None,
    ) -> float | None:
        """Adjust trust score with reason tracking and audit history.

        Args:
            memory_id: ID of memory to adjust
            reason: Why trust is being adjusted (from TrustReason enum)
            delta: Trust change amount. If None, uses default for reason.
            similarity: Optional similarity score for confidence-weighted updates.
            note: Optional human-readable note for audit.

        Returns:
            New trust score, or None if memory not found.
        """
        if delta is None:
            delta = TRUST_REASON_DEFAULTS.get(reason, 0.0)

        # Confidence-weighted scaling: higher similarity = larger boost (0.5x to 1.0x)
        if similarity is not None and delta > 0:
            delta = delta * (0.5 + 0.5 * similarity)

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT trust_score FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not row:
                return None

            old_trust = row["trust_score"] if row["trust_score"] is not None else 1.0
            new_trust = max(0.0, min(1.0, old_trust + delta))

            # Update memory
            conn.execute(
                """
                UPDATE memories
                SET trust_score = ?,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_trust, memory_id),
            )

            # Record in history
            conn.execute(
                """
                INSERT INTO trust_history
                    (memory_id, reason, old_trust, new_trust, delta, similarity, note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (memory_id, reason.value, old_trust, new_trust, delta, similarity, note),
            )

            log.info(
                "Adjusted trust for memory id={}: {:.2f} -> {:.2f} (reason={}, delta={:.3f})",
                memory_id,
                old_trust,
                new_trust,
                reason.value,
                delta,
            )
            return new_trust

    def strengthen_trust(
        self,
        memory_id: int,
        boost: float = 0.1,
        reason: TrustReason = TrustReason.USED_CORRECTLY,
        similarity: float | None = None,
        note: str | None = None,
    ) -> float | None:
        """Strengthen trust score when memory is validated/confirmed useful.

        Increases trust_score by boost amount, capped at 1.0.
        Also updates last_accessed_at to refresh the decay timer.

        Args:
            memory_id: ID of memory to strengthen
            boost: Amount to increase trust (default 0.1, so 10 validations = full trust)
            reason: Why trust is being strengthened (for audit)
            similarity: Optional similarity score for confidence weighting
            note: Optional note for audit trail

        Returns:
            New trust score, or None if memory not found.
        """
        return self.adjust_trust(
            memory_id,
            reason=reason,
            delta=boost,
            similarity=similarity,
            note=note,
        )

    def weaken_trust(
        self,
        memory_id: int,
        penalty: float = 0.1,
        reason: TrustReason = TrustReason.OUTDATED,
        note: str | None = None,
    ) -> float | None:
        """Weaken trust score when memory is found incorrect/outdated.

        Decreases trust_score by penalty amount, floored at 0.0.

        Args:
            memory_id: ID of memory to weaken
            penalty: Amount to decrease trust (default 0.1)
            reason: Why trust is being weakened (for audit)
            note: Optional note for audit trail

        Returns:
            New trust score, or None if memory not found.
        """
        return self.adjust_trust(
            memory_id,
            reason=reason,
            delta=-abs(penalty),  # Ensure negative
            note=note,
        )

    def get_trust_history(self, memory_id: int | None = None, limit: int = 50) -> list[TrustEvent]:
        """Get trust change history for audit/debugging.

        Args:
            memory_id: Optional filter by memory ID. If None, returns all.
            limit: Maximum events to return.

        Returns:
            List of TrustEvent objects, most recent first.
        """
        with self._connection() as conn:
            if memory_id is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM trust_history
                    WHERE memory_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (memory_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM trust_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [
                TrustEvent(
                    id=row["id"],
                    memory_id=row["memory_id"],
                    reason=TrustReason(row["reason"]),
                    old_trust=row["old_trust"],
                    new_trust=row["new_trust"],
                    delta=row["delta"],
                    similarity=row["similarity"],
                    note=row["note"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    def _get_trust_decay_halflife(self, memory_type: MemoryType | None) -> float:
        """Get trust decay half-life for a specific memory type."""
        if memory_type is None:
            return self.settings.trust_decay_halflife_days

        type_halflife_days = {
            MemoryType.PROJECT: self.settings.trust_decay_project_days,
            MemoryType.PATTERN: self.settings.trust_decay_pattern_days,
            MemoryType.REFERENCE: self.settings.trust_decay_reference_days,
            MemoryType.CONVERSATION: self.settings.trust_decay_conversation_days,
        }
        return type_halflife_days.get(memory_type, self.settings.trust_decay_halflife_days)

    def check_auto_promote(self, memory_id: int) -> bool:
        """Check if memory should be auto-promoted and do so if eligible.

        Auto-promotes if:
        - auto_promote is enabled in settings
        - memory is not already hot
        - access_count >= promotion_threshold

        Returns True if memory was promoted.
        """
        if not self.settings.auto_promote:
            return False

        with self._connection() as conn:
            row = conn.execute(
                "SELECT is_hot, access_count FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()

            if not row or row["is_hot"]:
                return False

            if row["access_count"] >= self.settings.promotion_threshold:
                promoted = self.promote_to_hot(memory_id, PromotionSource.AUTO_THRESHOLD)
                if promoted:
                    log.info(
                        "Auto-promoted memory id={} (access_count={} >= threshold={})",
                        memory_id,
                        row["access_count"],
                        self.settings.promotion_threshold,
                    )
                return promoted

        return False

    # ========== Vector Search ==========

    def get_recall_mode_config(self, mode: RecallMode) -> RecallModeConfig:
        """Get configuration for a recall mode preset."""
        if mode == RecallMode.PRECISION:
            return RecallModeConfig(
                threshold=self.settings.precision_threshold,
                limit=self.settings.precision_limit,
                similarity_weight=self.settings.precision_similarity_weight,
                recency_weight=self.settings.precision_recency_weight,
                access_weight=self.settings.precision_access_weight,
            )
        elif mode == RecallMode.EXPLORATORY:
            return RecallModeConfig(
                threshold=self.settings.exploratory_threshold,
                limit=self.settings.exploratory_limit,
                similarity_weight=self.settings.exploratory_similarity_weight,
                recency_weight=self.settings.exploratory_recency_weight,
                access_weight=self.settings.exploratory_access_weight,
            )
        else:  # BALANCED (default)
            return RecallModeConfig(
                threshold=self.settings.default_confidence_threshold,
                limit=self.settings.default_recall_limit,
                similarity_weight=self.settings.recall_similarity_weight,
                recency_weight=self.settings.recall_recency_weight,
                access_weight=self.settings.recall_access_weight,
            )

    def _compute_recency_score(self, created_at: datetime) -> float:
        """Compute recency score (0-1) with exponential decay.

        Returns 1.0 for just-created items, decaying to 0.5 at half-life.
        """
        halflife_days = self.settings.recall_recency_halflife_days
        days_old = (datetime.now() - created_at).total_seconds() / 86400
        return 2 ** (-days_old / halflife_days)

    def _compute_trust_decay(
        self,
        base_trust: float,
        created_at: datetime,
        last_accessed_at: datetime | None = None,
        memory_type: MemoryType | None = None,
    ) -> float:
        """Compute time-decayed trust score.

        Trust decays based on time since last meaningful interaction:
        - If recently accessed, decay is based on last_accessed_at (refresh on use)
        - Otherwise, decay is based on created_at
        - Decay rate varies by memory type (project decays slowest, conversation fastest)

        This means memories that are actively used maintain their trust,
        while unused memories slowly decay - aligning with Engram's principle
        that frequently-used patterns should remain reliable.

        Args:
            base_trust: Initial trust (1.0 for manual, 0.7 for mined by default)
            created_at: When the memory was created
            last_accessed_at: When the memory was last accessed (optional)
            memory_type: Type of memory for per-type decay rate

        Returns:
            Trust score with exponential decay applied based on staleness.
        """
        halflife_days = self._get_trust_decay_halflife(memory_type)
        reference_time = last_accessed_at if last_accessed_at else created_at
        days_since_activity = (datetime.now() - reference_time).total_seconds() / 86400

        decay = 2 ** (-days_since_activity / halflife_days)
        return base_trust * decay

    def _compute_access_score(self, access_count: int, max_access: int) -> float:
        """Normalize access count to 0-1 range."""
        if max_access <= 0:
            return 0.0
        return min(1.0, access_count / max_access)

    def _generate_recall_guidance(
        self,
        confidence: str,
        result_count: int,
        gated_count: int,
        mode: RecallMode,
    ) -> str:
        """Generate hallucination prevention guidance based on recall results.

        Provides explicit instructions on how to use (or not use) the results.
        """
        if confidence == "high" and result_count > 0:
            return (
                "HIGH CONFIDENCE: Use these memories directly. "
                "The top result closely matches your query."
            )
        elif confidence == "medium" and result_count > 0:
            return (
                "MEDIUM CONFIDENCE: Verify these memories apply to current context. "
                "Results are relevant but may need validation."
            )
        elif result_count == 0 and gated_count > 0:
            # Had results but all were below threshold
            return (
                f"NO CONFIDENT MATCH: {gated_count} memories were found but filtered "
                "due to low similarity. Reason from first principles or try a "
                "different query. Do NOT guess or hallucinate information."
            )
        elif result_count == 0:
            # No results at all
            suggestions = []
            if mode == RecallMode.PRECISION:
                suggestions.append("try 'exploratory' mode for broader search")
            suggestions.append("try rephrasing your query")
            suggestions.append("store relevant information with 'remember' first")

            return (
                "NO MATCH FOUND: No relevant memories exist for this query. "
                f"Suggestions: {'; '.join(suggestions)}. "
                "Do NOT fabricate or guess information."
            )
        else:
            # Low confidence with some results
            return (
                "LOW CONFIDENCE: Results have weak similarity to your query. "
                "Use with caution and verify independently. Consider that the "
                "information you need may not be stored yet."
            )

    def _compute_composite_score(
        self,
        similarity: float,
        recency_score: float,
        access_score: float,
        trust_score: float = 1.0,
        weights: RecallModeConfig | None = None,
    ) -> ScoreBreakdown:
        """Compute weighted composite score for ranking.

        Combines semantic similarity with recency, access frequency, and trust.
        Trust weight is optional (default 0) for backwards compatibility.

        Args:
            similarity: Semantic similarity score (0-1)
            recency_score: Time-decayed recency score (0-1)
            access_score: Normalized access count (0-1)
            trust_score: Trust score with decay (0-1)
            weights: Optional custom weights from recall mode preset

        Returns:
            ScoreBreakdown with total and individual weighted components.
        """
        if weights:
            sim_weight = weights.similarity_weight
            rec_weight = weights.recency_weight
            acc_weight = weights.access_weight
        else:
            sim_weight = self.settings.recall_similarity_weight
            rec_weight = self.settings.recall_recency_weight
            acc_weight = self.settings.recall_access_weight

        trust_weight = self.settings.recall_trust_weight

        sim_component = similarity * sim_weight
        rec_component = recency_score * rec_weight
        acc_component = access_score * acc_weight
        trust_component = trust_score * trust_weight

        return ScoreBreakdown(
            total=sim_component + rec_component + acc_component + trust_component,
            similarity_component=sim_component,
            recency_component=rec_component,
            access_component=acc_component,
            trust_component=trust_component,
        )

    def recall(
        self,
        query: str,
        limit: int | None = None,
        threshold: float | None = None,
        mode: RecallMode | None = None,
        memory_types: list[MemoryType] | None = None,
    ) -> RecallResult:
        """Semantic search with confidence gating and composite ranking.

        Args:
            query: Search query for semantic similarity
            limit: Maximum results (overrides mode preset if set)
            threshold: Minimum similarity (overrides mode preset if set)
            mode: Recall mode preset (precision, balanced, exploratory)
            memory_types: Filter to specific memory types

        Results are ranked by composite score combining:
        - Semantic similarity (weight varies by mode)
        - Recency with exponential decay
        - Access frequency
        - Trust score with decay (optional)
        """
        # Get mode config (or default balanced)
        mode_config = self.get_recall_mode_config(mode or RecallMode.BALANCED)

        # Allow explicit overrides
        effective_limit = limit if limit is not None else mode_config.limit
        effective_threshold = threshold if threshold is not None else mode_config.threshold

        query_embedding = self._embedding_engine.embed(query)

        with self._connection() as conn:
            # Build query with optional type filter
            # Include all Memory fields for accurate response mapping
            if memory_types:
                type_placeholders = ",".join("?" * len(memory_types))
                type_values = [t.value for t in memory_types]
                query_sql = f"""
                    SELECT
                        m.id,
                        m.content,
                        m.content_hash,
                        m.memory_type,
                        m.source,
                        m.is_hot,
                        m.is_pinned,
                        m.promotion_source,
                        m.access_count,
                        m.last_accessed_at,
                        m.created_at,
                        m.trust_score,
                        m.source_log_id,
                        m.extracted_at,
                        m.session_id,
                        vec_distance_cosine(v.embedding, ?) as distance
                    FROM memory_vectors v
                    JOIN memories m ON m.id = v.rowid
                    WHERE m.memory_type IN ({type_placeholders})
                    ORDER BY distance ASC
                    LIMIT ?
                """
                params = (
                    query_embedding.tobytes(),
                    *type_values,
                    effective_limit * 3,
                )
            else:
                query_sql = """
                    SELECT
                        m.id,
                        m.content,
                        m.content_hash,
                        m.memory_type,
                        m.source,
                        m.is_hot,
                        m.is_pinned,
                        m.promotion_source,
                        m.access_count,
                        m.last_accessed_at,
                        m.created_at,
                        m.trust_score,
                        m.source_log_id,
                        m.extracted_at,
                        m.session_id,
                        vec_distance_cosine(v.embedding, ?) as distance
                    FROM memory_vectors v
                    JOIN memories m ON m.id = v.rowid
                    ORDER BY distance ASC
                    LIMIT ?
                """
                params = (query_embedding.tobytes(), effective_limit * 3)

            rows = conn.execute(query_sql, params).fetchall()

            # Find max access count for normalization
            max_access = max((row["access_count"] for row in rows), default=1)

            # Convert distance to similarity and compute scores
            candidates = []
            gated_count = 0

            for row in rows:
                similarity = 1 - row["distance"]  # cosine distance to similarity

                if similarity >= effective_threshold:
                    created_at = datetime.fromisoformat(row["created_at"])
                    last_accessed_str = row["last_accessed_at"]
                    last_accessed_at = (
                        datetime.fromisoformat(last_accessed_str) if last_accessed_str else None
                    )
                    recency_score = self._compute_recency_score(created_at)
                    access_score = self._compute_access_score(row["access_count"], max_access)

                    # Compute trust with time decay (refreshed by access, per-type rate)
                    memory_type_enum = (
                        MemoryType(row["memory_type"]) if row["memory_type"] else None
                    )
                    base_trust = row["trust_score"] if row["trust_score"] is not None else 1.0
                    trust_decayed = self._compute_trust_decay(
                        base_trust, created_at, last_accessed_at, memory_type_enum
                    )

                    score_breakdown = self._compute_composite_score(
                        similarity,
                        recency_score,
                        access_score,
                        trust_decayed,
                        weights=mode_config,
                    )

                    memory = self._row_to_memory(row, conn, similarity=similarity)
                    memory.recency_score = recency_score
                    memory.trust_score_decayed = trust_decayed
                    memory.composite_score = score_breakdown.total
                    # Populate weighted components for transparency
                    memory.similarity_component = score_breakdown.similarity_component
                    memory.recency_component = score_breakdown.recency_component
                    memory.access_component = score_breakdown.access_component
                    memory.trust_component = score_breakdown.trust_component
                    candidates.append(memory)
                else:
                    gated_count += 1

            # Re-rank by composite score
            candidates.sort(key=lambda m: m.composite_score or 0, reverse=True)

            # Take top results and update access counts
            memories = candidates[:effective_limit]
            memory_ids_to_check = []
            for memory in memories:
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (memory.id,),
                )
                # Track for auto-promotion check (if not already hot)
                if not memory.is_hot:
                    memory_ids_to_check.append(memory.id)
            conn.commit()

        # Check for auto-promotion outside the transaction
        for memory_id in memory_ids_to_check:
            self.check_auto_promote(memory_id)

        # Auto-strengthen trust for high-similarity matches (confidence-weighted)
        if self.settings.trust_auto_strengthen_on_recall:
            for memory in memories:
                if (
                    memory.similarity
                    and memory.similarity >= self.settings.trust_high_similarity_threshold
                ):
                    # Confidence-weighted: similarity scales the boost
                    self.adjust_trust(
                        memory.id,
                        reason=TrustReason.HIGH_SIMILARITY_HIT,
                        similarity=memory.similarity,
                    )

        # Determine confidence level based on top result's similarity
        if not memories:
            confidence = "low"
        elif (
            memories[0].similarity
            and memories[0].similarity > self.settings.high_confidence_threshold
        ):
            confidence = "high"
        else:
            confidence = "medium"

        # Generate hallucination prevention guidance
        effective_mode = mode or RecallMode.BALANCED
        guidance = self._generate_recall_guidance(
            confidence, len(memories), gated_count, effective_mode
        )

        log.debug(
            "Recall query='{}' mode={} returned {} results (confidence={}, gated={})",
            query[:50],
            effective_mode.value,
            len(memories),
            confidence,
            gated_count,
        )

        return RecallResult(
            memories=memories,
            confidence=confidence,
            gated_count=gated_count,
            mode=effective_mode,
            guidance=guidance,
        )

    def _get_memories_by_ids(self, ids: list[int]) -> list[Memory]:
        """Fetch multiple memories by ID, filtering out None results."""
        return [m for mid in ids if (m := self.get_memory(mid)) is not None]

    def recall_with_fallback(
        self,
        query: str,
        fallback_types: list[list[MemoryType]] | None = None,
        mode: RecallMode | None = None,
        min_results: int = 1,
    ) -> RecallResult:
        """Recall with multi-query fallback through different memory type filters.

        Tries each type filter in sequence until min_results are found or
        all fallbacks exhausted. Default fallback order:
        1. patterns only (code snippets)
        2. project facts
        3. all types (no filter)

        Args:
            query: Search query
            fallback_types: List of type filters to try in order
            mode: Recall mode preset
            min_results: Minimum results needed before stopping fallback

        Returns:
            RecallResult from first successful search, or last attempt
        """
        if fallback_types is None:
            # Default fallback: patterns -> project -> all
            fallback_types = [
                [MemoryType.PATTERN],
                [MemoryType.PROJECT],
                None,  # All types
            ]

        best_result: RecallResult | None = None

        for type_filter in fallback_types:
            result = self.recall(
                query=query,
                mode=mode,
                memory_types=type_filter,
            )

            # Track best result (most memories found)
            if best_result is None or len(result.memories) > len(best_result.memories):
                best_result = result

            # Stop if we have enough high-quality results
            if len(result.memories) >= min_results and result.confidence != "low":
                log.debug(
                    "Fallback succeeded with type_filter={} ({} results)",
                    [t.value for t in type_filter] if type_filter else "all",
                    len(result.memories),
                )
                return result

        # Return best result found (or empty)
        log.debug(
            "Fallback exhausted, returning best result ({} memories)",
            len(best_result.memories) if best_result else 0,
        )
        return best_result or RecallResult(memories=[], confidence="low", gated_count=0)

    def recall_by_tag(self, tag: str, limit: int | None = None) -> list[Memory]:
        """Get memories by tag."""
        limit = limit or self.settings.default_recall_limit

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT m.id FROM memories m
                JOIN memory_tags t ON t.memory_id = m.id
                WHERE t.tag = ?
                ORDER BY m.access_count DESC, m.created_at DESC
                LIMIT ?
                """,
                (tag, limit),
            ).fetchall()
            ids = [row["id"] for row in rows]

        return self._get_memories_by_ids(ids)

    # ========== Hot Cache ==========

    def get_hot_memories(self) -> list[Memory]:
        """Get all memories in hot cache, ordered by hot_score descending."""
        memories = []
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM memories WHERE is_hot = 1").fetchall()
            for row in rows:
                memories.append(self._row_to_memory(row, conn))

        # Sort by hot_score descending (highest score first)
        memories.sort(key=lambda m: m.hot_score or 0, reverse=True)
        return memories

    def record_hot_cache_hit(self) -> None:
        """Record a hot cache hit (resource was read with content)."""
        self._hot_cache_metrics.hits += 1

    def record_hot_cache_miss(self) -> None:
        """Record a hot cache miss (resource was read but empty)."""
        self._hot_cache_metrics.misses += 1

    def get_hot_cache_metrics(self) -> HotCacheMetrics:
        """Get current hot cache metrics."""
        return self._hot_cache_metrics

    def get_hot_cache_stats(self) -> dict:
        """Get hot cache statistics including metrics and computed values."""
        hot_memories = self.get_hot_memories()
        metrics = self._hot_cache_metrics.to_dict()

        # Compute average hot score
        avg_score = 0.0
        if hot_memories:
            scores = [m.hot_score for m in hot_memories if m.hot_score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            **metrics,
            "current_count": len(hot_memories),
            "max_items": self.settings.hot_cache_max_items,
            "avg_hot_score": round(avg_score, 3),
            "pinned_count": sum(1 for m in hot_memories if m.is_pinned),
        }

    def _find_eviction_candidate(self, conn: sqlite3.Connection) -> int | None:
        """Find the lowest-scoring non-pinned hot memory for eviction."""
        rows = conn.execute(
            """
            SELECT id, access_count, last_accessed_at
            FROM memories
            WHERE is_hot = 1 AND is_pinned = 0
            """
        ).fetchall()

        if not rows:
            return None

        # Compute scores and find minimum
        candidates = []
        for row in rows:
            last_accessed = row["last_accessed_at"]
            last_accessed_dt = datetime.fromisoformat(last_accessed) if last_accessed else None
            score = self._compute_hot_score(row["access_count"], last_accessed_dt)
            candidates.append((row["id"], score))

        if not candidates:
            return None

        # Return ID with lowest score
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def promote_to_hot(
        self,
        memory_id: int,
        promotion_source: PromotionSource = PromotionSource.MANUAL,
        pin: bool = False,
    ) -> bool:
        """Promote a memory to hot cache with score-based eviction.

        Args:
            memory_id: ID of memory to promote
            promotion_source: How the memory is being promoted
            pin: If True, memory won't be auto-evicted

        Returns:
            True if promoted successfully
        """
        with self.transaction() as conn:
            # Check if already hot
            existing = conn.execute(
                "SELECT is_hot FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not existing:
                return False
            if existing["is_hot"]:
                # Already hot, just update pinned status if requested
                if pin:
                    conn.execute("UPDATE memories SET is_pinned = 1 WHERE id = ?", (memory_id,))
                return True

            # Check hot cache limit
            hot_count = conn.execute("SELECT COUNT(*) FROM memories WHERE is_hot = 1").fetchone()[0]

            if hot_count >= self.settings.hot_cache_max_items:
                # Find lowest-scoring non-pinned memory to evict
                evict_id = self._find_eviction_candidate(conn)
                if evict_id is None:
                    log.warning(
                        "Cannot promote memory id={}: hot cache full and all items pinned",
                        memory_id,
                    )
                    return False

                conn.execute(
                    "UPDATE memories SET is_hot = 0, promotion_source = NULL WHERE id = ?",
                    (evict_id,),
                )
                self._hot_cache_metrics.evictions += 1
                log.debug("Evicted memory id={} from hot cache (lowest score)", evict_id)

            # Promote the memory
            cursor = conn.execute(
                """
                UPDATE memories
                SET is_hot = 1, is_pinned = ?, promotion_source = ?
                WHERE id = ?
                """,
                (int(pin), promotion_source.value, memory_id),
            )
            promoted = cursor.rowcount > 0
            if promoted:
                self._hot_cache_metrics.promotions += 1
                log.info(
                    "Promoted memory id={} to hot cache (source={}, pinned={})",
                    memory_id,
                    promotion_source.value,
                    pin,
                )
            return promoted

    def demote_from_hot(self, memory_id: int) -> bool:
        """Remove a memory from hot cache (ignores pinned status)."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """UPDATE memories
                   SET is_hot = 0, is_pinned = 0, promotion_source = NULL
                   WHERE id = ?""",
                (memory_id,),
            )
            demoted = cursor.rowcount > 0
            if demoted:
                log.info("Demoted memory id={} from hot cache", memory_id)
            return demoted

    def demote_stale_hot_memories(self) -> list[int]:
        """Demote hot memories that haven't been accessed in demotion_days.

        Skips pinned memories. Called during maintenance if auto_demote is enabled.
        Uses created_at as fallback when last_accessed_at is NULL (newly promoted).

        Returns list of demoted memory IDs.
        """
        if not self.settings.auto_demote:
            return []

        demoted_ids = []
        cutoff = f"-{self.settings.demotion_days} days"

        with self._connection() as conn:
            # Find stale non-pinned hot memories
            # Use COALESCE to fall back to created_at for newly promoted memories
            rows = conn.execute(
                """
                SELECT id FROM memories
                WHERE is_hot = 1
                  AND is_pinned = 0
                  AND COALESCE(last_accessed_at, created_at) < datetime('now', ?)
                """,
                (cutoff,),
            ).fetchall()

            stale_ids = [row["id"] for row in rows]

        # Demote each (outside the read transaction)
        for memory_id in stale_ids:
            if self.demote_from_hot(memory_id):
                demoted_ids.append(memory_id)
                log.info(
                    "Auto-demoted stale memory id={} (not accessed in {} days)",
                    memory_id,
                    self.settings.demotion_days,
                )

        return demoted_ids

    def pin_memory(self, memory_id: int) -> bool:
        """Pin a hot memory so it won't be auto-evicted."""
        with self.transaction() as conn:
            # Only pin if already in hot cache
            cursor = conn.execute(
                "UPDATE memories SET is_pinned = 1 WHERE id = ? AND is_hot = 1",
                (memory_id,),
            )
            pinned = cursor.rowcount > 0
            if pinned:
                log.info("Pinned memory id={}", memory_id)
            return pinned

    def unpin_memory(self, memory_id: int) -> bool:
        """Unpin a memory, making it eligible for auto-eviction."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE memories SET is_pinned = 0 WHERE id = ?",
                (memory_id,),
            )
            unpinned = cursor.rowcount > 0
            if unpinned:
                log.info("Unpinned memory id={}", memory_id)
            return unpinned

    # ========== Statistics ==========

    def get_stats(self) -> dict:
        """Get memory statistics."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            hot = conn.execute("SELECT COUNT(*) FROM memories WHERE is_hot = 1").fetchone()[0]
            by_type = {
                row["memory_type"]: row["count"]
                for row in conn.execute(
                    "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
                )
            }
            by_source = {
                row["source"]: row["count"]
                for row in conn.execute(
                    "SELECT source, COUNT(*) as count FROM memories GROUP BY source"
                )
            }

            return {
                "total_memories": total,
                "hot_cache_count": hot,
                "by_type": by_type,
                "by_source": by_source,
            }

    def list_memories(
        self,
        limit: int = 20,
        offset: int = 0,
        memory_type: MemoryType | None = None,
    ) -> list[Memory]:
        """List memories with pagination."""
        with self._connection() as conn:
            if memory_type:
                rows = conn.execute(
                    """
                    SELECT id FROM memories
                    WHERE memory_type = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (memory_type.value, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            ids = [row["id"] for row in rows]

        return self._get_memories_by_ids(ids)

    # ========== Output Logging ==========

    def log_output(self, content: str, session_id: str | None = None) -> int:
        """Log an output for pattern mining.

        Args:
            content: The output content to log.
            session_id: Optional session ID for provenance tracking.

        Raises:
            ValidationError: If content is empty or exceeds max length.
        """
        # Defense-in-depth validation
        self._validate_content(content, "output content")

        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO output_log (content, session_id) VALUES (?, ?)",
                (content, session_id),
            )

            # Update session log count if session_id provided (upsert creates if needed)
            if session_id:
                self._update_session_activity(conn, session_id, log_delta=1)

            # Cleanup old logs
            conn.execute(
                "DELETE FROM output_log WHERE timestamp < datetime('now', ?)",
                (f"-{self.settings.log_retention_days} days",),
            )

            log_id = cursor.lastrowid or 0
            log.debug(
                "Logged output id={} ({} chars) session={}",
                log_id,
                len(content),
                session_id,
            )
            return log_id

    def get_recent_outputs(self, hours: int = 24) -> list[tuple[int, str, datetime]]:
        """Get recent output logs."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, content, timestamp FROM output_log
                WHERE timestamp > datetime('now', ?)
                ORDER BY timestamp DESC
                """,
                (f"-{hours} hours",),
            ).fetchall()

            return [
                (row["id"], row["content"], datetime.fromisoformat(row["timestamp"]))
                for row in rows
            ]

    # ========== Mined Patterns ==========

    def upsert_mined_pattern(
        self,
        pattern: str,
        pattern_type: str,
        source_log_id: int | None = None,
        confidence: float = 0.5,
    ) -> int:
        """Insert or update a mined pattern.

        Args:
            pattern: The extracted pattern content.
            pattern_type: Type of pattern (import, fact, command, code).
            source_log_id: ID of the output_log this pattern was extracted from.
            confidence: Extraction confidence score (0-1).

        Returns:
            The pattern ID.

        Raises:
            ValidationError: If pattern is empty or exceeds max length.
        """
        # Defense-in-depth validation
        self._validate_content(pattern, "pattern")

        hash_val = content_hash(pattern)

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO mined_patterns
                    (pattern, pattern_hash, pattern_type, source_log_id, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pattern_hash) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen = CURRENT_TIMESTAMP,
                    confidence = MAX(confidence, excluded.confidence)
                RETURNING id
                """,
                (pattern, hash_val, pattern_type, source_log_id, confidence),
            )
            pattern_id = cursor.fetchone()[0]

            # Recalculate score on update
            self._update_pattern_score(conn, pattern_id)

            return pattern_id

    def _update_pattern_score(self, conn: sqlite3.Connection, pattern_id: int) -> None:
        """Recalculate and update the promotion score for a pattern."""
        row = conn.execute(
            """
            SELECT occurrence_count, first_seen, last_seen, confidence
            FROM mined_patterns WHERE id = ?
            """,
            (pattern_id,),
        ).fetchone()

        if not row:
            return

        # Score = frequency * recency_factor * confidence
        # Recency: decay based on days since last seen
        last_seen = datetime.fromisoformat(row["last_seen"])
        days_since = (datetime.now() - last_seen).days
        recency_factor = 0.5 ** (days_since / 7.0)  # Half-life of 7 days

        frequency_score = min(row["occurrence_count"] / 10.0, 1.0)  # Cap at 10 occurrences
        confidence = row["confidence"] or 0.5

        score = frequency_score * recency_factor * confidence

        conn.execute(
            "UPDATE mined_patterns SET score = ? WHERE id = ?",
            (score, pattern_id),
        )

    def _row_to_mined_pattern(self, row: sqlite3.Row) -> MinedPattern:
        """Convert a database row to a MinedPattern."""
        status = PatternStatus(row["status"]) if row["status"] else PatternStatus.PENDING
        return MinedPattern(
            id=row["id"],
            pattern=row["pattern"],
            pattern_hash=row["pattern_hash"],
            pattern_type=row["pattern_type"],
            occurrence_count=row["occurrence_count"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            status=status,
            source_log_id=row["source_log_id"],
            confidence=row["confidence"] or 0.5,
            score=row["score"] or 0.0,
        )

    def get_promotion_candidates(
        self,
        threshold: int | None = None,
        status: PatternStatus | None = None,
    ) -> list[MinedPattern]:
        """Get mined patterns ready for promotion.

        Args:
            threshold: Minimum occurrence count (defaults to settings.promotion_threshold).
            status: Filter by status (defaults to PENDING or APPROVED).

        Returns:
            List of MinedPattern sorted by score descending.
        """
        threshold = threshold or self.settings.promotion_threshold

        # Build status filter: specific status or default to pending/approved
        params: tuple[int, ...] | tuple[int, str]
        if status:
            status_filter = "status = ?"
            params = (threshold, status.value)
        else:
            status_filter = "status IN ('pending', 'approved')"
            params = (threshold,)

        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM mined_patterns
                WHERE occurrence_count >= ? AND {status_filter}
                ORDER BY score DESC, occurrence_count DESC
                """,
                params,
            ).fetchall()

            return [self._row_to_mined_pattern(row) for row in rows]

    def get_mined_pattern(self, pattern_id: int) -> MinedPattern | None:
        """Get a mined pattern by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM mined_patterns WHERE id = ?", (pattern_id,)
            ).fetchone()

        if not row:
            return None
        return self._row_to_mined_pattern(row)

    def update_pattern_status(
        self,
        pattern_id: int,
        status: PatternStatus,
    ) -> bool:
        """Update the status of a mined pattern.

        Args:
            pattern_id: Pattern ID to update.
            status: New status (pending, approved, rejected, promoted).

        Returns:
            True if updated, False if pattern not found.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE mined_patterns SET status = ? WHERE id = ?",
                (status.value, pattern_id),
            )
            return cursor.rowcount > 0

    def mined_pattern_exists(self, pattern_hash: str) -> bool:
        """Check if a mined pattern exists by hash."""
        with self._connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM mined_patterns WHERE pattern_hash = ?", (pattern_hash,)
            ).fetchone()
            return result is not None

    def delete_mined_pattern(self, pattern_id: int) -> bool:
        """Delete a mined pattern (after promotion or rejection)."""
        with self.transaction() as conn:
            cursor = conn.execute("DELETE FROM mined_patterns WHERE id = ?", (pattern_id,))
            return cursor.rowcount > 0

    def expire_stale_patterns(self, days: int = 30) -> int:
        """Expire pending patterns that haven't been seen in N days.

        Args:
            days: Number of days without activity before expiration.

        Returns:
            Number of patterns expired (marked as rejected).
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE mined_patterns
                SET status = 'rejected'
                WHERE status = 'pending'
                  AND last_seen < datetime('now', ?)
                """,
                (f"-{days} days",),
            )
            expired = cursor.rowcount
            if expired > 0:
                log.info("Expired {} stale patterns (inactive for {} days)", expired, days)
            return expired

    # ========== Memory Relationships (Knowledge Graph) ==========

    def link_memories(
        self,
        from_memory_id: int,
        to_memory_id: int,
        relation_type: RelationType,
    ) -> MemoryRelation | None:
        """Create a typed relationship between two memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation_type: Type of relationship

        Returns:
            MemoryRelation if created, None if memories don't exist or already linked.
        """
        if from_memory_id == to_memory_id:
            log.warning("Cannot link memory to itself: id={}", from_memory_id)
            return None

        with self.transaction() as conn:
            # Verify both memories exist
            from_exists = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?", (from_memory_id,)
            ).fetchone()
            to_exists = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?", (to_memory_id,)
            ).fetchone()

            if not from_exists or not to_exists:
                log.warning(
                    "Cannot link: memory {} or {} does not exist",
                    from_memory_id,
                    to_memory_id,
                )
                return None

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO memory_relationships (from_memory_id, to_memory_id, relation_type)
                    VALUES (?, ?, ?)
                    RETURNING id, created_at
                    """,
                    (from_memory_id, to_memory_id, relation_type.value),
                )
                row = cursor.fetchone()
                log.info(
                    "Linked memories: {} -[{}]-> {}",
                    from_memory_id,
                    relation_type.value,
                    to_memory_id,
                )
                return MemoryRelation(
                    id=row[0],
                    from_memory_id=from_memory_id,
                    to_memory_id=to_memory_id,
                    relation_type=relation_type,
                    created_at=datetime.fromisoformat(row[1]),
                )
            except sqlite3.IntegrityError:
                # Already linked with this relation type
                log.debug(
                    "Relationship already exists: {} -[{}]-> {}",
                    from_memory_id,
                    relation_type.value,
                    to_memory_id,
                )
                return None

    def unlink_memories(
        self,
        from_memory_id: int,
        to_memory_id: int,
        relation_type: RelationType | None = None,
    ) -> int:
        """Remove relationship(s) between memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation_type: If specified, only remove this type. Otherwise remove all.

        Returns:
            Number of relationships removed.
        """
        with self.transaction() as conn:
            if relation_type:
                cursor = conn.execute(
                    """
                    DELETE FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ? AND relation_type = ?
                    """,
                    (from_memory_id, to_memory_id, relation_type.value),
                )
            else:
                cursor = conn.execute(
                    """
                    DELETE FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ?
                    """,
                    (from_memory_id, to_memory_id),
                )

            count = cursor.rowcount
            if count > 0:
                log.info(
                    "Unlinked {} relationship(s): {} -> {}",
                    count,
                    from_memory_id,
                    to_memory_id,
                )
            return count

    def get_related(
        self,
        memory_id: int,
        relation_type: RelationType | None = None,
        direction: str = "both",
    ) -> list[tuple[Memory, MemoryRelation]]:
        """Get memories related to a given memory.

        Args:
            memory_id: The memory to find relationships for
            relation_type: Filter by relationship type (optional)
            direction: "outgoing" (from this memory), "incoming" (to this memory), or "both"

        Returns:
            List of (related_memory, relationship) tuples.
        """
        results: list[tuple[Memory, MemoryRelation]] = []

        with self._connection() as conn:
            queries: list[tuple[str, tuple[int, ...] | tuple[int, str], str]] = []

            if direction in ("outgoing", "both"):
                # Relationships FROM this memory
                if relation_type:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'outgoing' as direction
                            FROM memory_relationships r
                            WHERE r.from_memory_id = ? AND r.relation_type = ?
                            """,
                            (memory_id, relation_type.value),
                            "to_memory_id",
                        )
                    )
                else:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'outgoing' as direction
                            FROM memory_relationships r
                            WHERE r.from_memory_id = ?
                            """,
                            (memory_id,),
                            "to_memory_id",
                        )
                    )

            if direction in ("incoming", "both"):
                # Relationships TO this memory
                if relation_type:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'incoming' as direction
                            FROM memory_relationships r
                            WHERE r.to_memory_id = ? AND r.relation_type = ?
                            """,
                            (memory_id, relation_type.value),
                            "from_memory_id",
                        )
                    )
                else:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'incoming' as direction
                            FROM memory_relationships r
                            WHERE r.to_memory_id = ?
                            """,
                            (memory_id,),
                            "from_memory_id",
                        )
                    )

            for query, params, related_id_col in queries:
                rows = conn.execute(query, params).fetchall()
                for row in rows:
                    related_id = row[related_id_col]
                    relation = MemoryRelation(
                        id=row["id"],
                        from_memory_id=row["from_memory_id"],
                        to_memory_id=row["to_memory_id"],
                        relation_type=RelationType(row["relation_type"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                    # Fetch the related memory
                    memory = self.get_memory(related_id)
                    if memory:
                        results.append((memory, relation))

        return results

    def get_relationship(
        self,
        from_memory_id: int,
        to_memory_id: int,
        relation_type: RelationType | None = None,
    ) -> list[MemoryRelation]:
        """Get specific relationship(s) between two memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation_type: Filter by type (optional)

        Returns:
            List of matching relationships.
        """
        with self._connection() as conn:
            if relation_type:
                rows = conn.execute(
                    """
                    SELECT * FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ? AND relation_type = ?
                    """,
                    (from_memory_id, to_memory_id, relation_type.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ?
                    """,
                    (from_memory_id, to_memory_id),
                ).fetchall()

            return [
                MemoryRelation(
                    id=row["id"],
                    from_memory_id=row["from_memory_id"],
                    to_memory_id=row["to_memory_id"],
                    relation_type=RelationType(row["relation_type"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    def get_relationship_stats(self) -> dict:
        """Get statistics about memory relationships."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memory_relationships").fetchone()[0]
            by_type = {
                row["relation_type"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT relation_type, COUNT(*) as count
                    FROM memory_relationships
                    GROUP BY relation_type
                    """
                )
            }
            # Count memories with at least one relationship
            linked_memories = conn.execute(
                """
                SELECT COUNT(DISTINCT memory_id) FROM (
                    SELECT from_memory_id as memory_id FROM memory_relationships
                    UNION
                    SELECT to_memory_id as memory_id FROM memory_relationships
                )
                """
            ).fetchone()[0]

            return {
                "total_relationships": total,
                "by_type": by_type,
                "linked_memories": linked_memories,
            }

    # ========== Contradiction Detection ==========

    def find_contradictions(
        self,
        memory_id: int,
        similarity_threshold: float = 0.75,
        limit: int = 5,
    ) -> list[PotentialContradiction]:
        """Find memories that may contradict a given memory.

        Looks for memories that are semantically similar (same topic)
        but might contain conflicting information.

        Args:
            memory_id: The memory to check for contradictions
            similarity_threshold: Minimum similarity to consider (default 0.75)
            limit: Maximum contradictions to return

        Returns:
            List of potential contradictions sorted by similarity.
        """
        with self._connection() as conn:
            # Get the target memory and its embedding
            memory_row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not memory_row:
                return []

            embedding_row = conn.execute(
                "SELECT embedding FROM memory_vectors WHERE rowid = ?", (memory_id,)
            ).fetchone()
            if not embedding_row:
                return []

            # Find similar memories (excluding self)
            rows = conn.execute(
                """
                SELECT
                    m.id,
                    m.content,
                    m.content_hash,
                    m.memory_type,
                    m.source,
                    m.is_hot,
                    m.access_count,
                    m.last_accessed_at,
                    m.created_at,
                    m.trust_score,
                    vec_distance_cosine(v.embedding, ?) as distance
                FROM memory_vectors v
                JOIN memories m ON m.id = v.rowid
                WHERE m.id != ?
                ORDER BY distance ASC
                LIMIT ?
                """,
                (embedding_row["embedding"], memory_id, limit * 2),
            ).fetchall()

            # Check which are already linked as contradictions
            existing_contradictions = set()
            contradicts_type = RelationType.CONTRADICTS.value
            for row in conn.execute(
                """
                SELECT to_memory_id FROM memory_relationships
                WHERE from_memory_id = ? AND relation_type = ?
                UNION
                SELECT from_memory_id FROM memory_relationships
                WHERE to_memory_id = ? AND relation_type = ?
                """,
                (memory_id, contradicts_type, memory_id, contradicts_type),
            ):
                existing_contradictions.add(row[0])

            source_memory = self._row_to_memory(memory_row, conn)
            results: list[PotentialContradiction] = []

            for row in rows:
                similarity = 1 - row["distance"]
                if similarity < similarity_threshold:
                    continue

                other_memory = self._row_to_memory(row, conn)
                results.append(
                    PotentialContradiction(
                        memory_a=source_memory,
                        memory_b=other_memory,
                        similarity=similarity,
                        already_linked=other_memory.id in existing_contradictions,
                    )
                )

                if len(results) >= limit:
                    break

            return results

    def get_all_contradictions(self) -> list[tuple[Memory, Memory, MemoryRelation]]:
        """Get all memory pairs marked as contradictions.

        Returns:
            List of (memory_a, memory_b, relationship) tuples.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, from_memory_id, to_memory_id, relation_type, created_at
                FROM memory_relationships
                WHERE relation_type = ?
                ORDER BY created_at DESC
                """,
                (RelationType.CONTRADICTS.value,),
            ).fetchall()

            results: list[tuple[Memory, Memory, MemoryRelation]] = []
            for row in rows:
                relation = MemoryRelation(
                    id=row["id"],
                    from_memory_id=row["from_memory_id"],
                    to_memory_id=row["to_memory_id"],
                    relation_type=RelationType(row["relation_type"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )

                m1 = self.get_memory(row["from_memory_id"])
                m2 = self.get_memory(row["to_memory_id"])

                if m1 and m2:
                    results.append((m1, m2, relation))

            return results

    def mark_contradiction(
        self,
        memory_id_a: int,
        memory_id_b: int,
    ) -> MemoryRelation | None:
        """Mark two memories as contradicting each other.

        Creates a CONTRADICTS relationship between the memories.

        Args:
            memory_id_a: First memory ID
            memory_id_b: Second memory ID

        Returns:
            The created relationship, or None if already exists or memories don't exist.
        """
        return self.link_memories(memory_id_a, memory_id_b, RelationType.CONTRADICTS)

    def resolve_contradiction(
        self,
        memory_id_a: int,
        memory_id_b: int,
        keep_id: int,
        resolution: str = "supersedes",
    ) -> bool:
        """Resolve a contradiction by keeping one memory and handling the other.

        Args:
            memory_id_a: First memory in contradiction
            memory_id_b: Second memory in contradiction
            keep_id: Which memory to keep (must be one of the two)
            resolution: How to handle the discarded memory:
                - "supersedes": Keep memory supersedes the other (default)
                - "delete": Delete the other memory
                - "weaken": Weaken trust in the other memory

        Returns:
            True if resolution succeeded.
        """
        if keep_id not in (memory_id_a, memory_id_b):
            log.warning("keep_id must be one of the contradicting memories")
            return False

        discard_id = memory_id_b if keep_id == memory_id_a else memory_id_a

        # Remove the contradiction relationship
        self.unlink_memories(memory_id_a, memory_id_b, RelationType.CONTRADICTS)
        self.unlink_memories(memory_id_b, memory_id_a, RelationType.CONTRADICTS)

        if resolution == "delete":
            return self.delete_memory(discard_id)
        elif resolution == "supersedes":
            self.link_memories(keep_id, discard_id, RelationType.SUPERSEDES)
            # Weaken trust in superseded memory
            self.weaken_trust(discard_id, 0.2, TrustReason.CONTRADICTION_RESOLVED)
            return True
        elif resolution == "weaken":
            self.weaken_trust(discard_id, 0.3, TrustReason.CONTRADICTION_RESOLVED)
            return True
        else:
            log.warning("Unknown resolution type: {}", resolution)
            return False

    # ========== Predictive Hot Cache Warming ==========

    def record_access_sequence(
        self,
        from_memory_id: int,
        to_memory_id: int,
    ) -> None:
        """Record that to_memory was accessed after from_memory.

        Builds a Markov chain of access patterns for prediction.
        """
        if not self.settings.predictive_cache_enabled:
            return

        if from_memory_id == to_memory_id:
            return

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO access_sequences (from_memory_id, to_memory_id, count, last_seen)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(from_memory_id, to_memory_id) DO UPDATE SET
                    count = count + 1,
                    last_seen = CURRENT_TIMESTAMP
                """,
                (from_memory_id, to_memory_id),
            )

    def get_access_patterns(
        self,
        memory_id: int,
        limit: int = 10,
    ) -> list[AccessPattern]:
        """Get learned access patterns from a memory.

        Returns patterns sorted by probability (count / total outgoing).
        """
        with self._connection() as conn:
            # Get total outgoing count for this memory
            total_row = conn.execute(
                "SELECT SUM(count) FROM access_sequences WHERE from_memory_id = ?",
                (memory_id,),
            ).fetchone()
            total = total_row[0] if total_row[0] else 0

            if total == 0:
                return []

            rows = conn.execute(
                """
                SELECT from_memory_id, to_memory_id, count, last_seen
                FROM access_sequences
                WHERE from_memory_id = ?
                ORDER BY count DESC
                LIMIT ?
                """,
                (memory_id, limit),
            ).fetchall()

            return [
                AccessPattern(
                    from_memory_id=row["from_memory_id"],
                    to_memory_id=row["to_memory_id"],
                    count=row["count"],
                    probability=row["count"] / total,
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                )
                for row in rows
            ]

    def predict_next_memories(
        self,
        memory_id: int,
        threshold: float | None = None,
        limit: int | None = None,
    ) -> list[PredictionResult]:
        """Predict which memories might be needed after accessing memory_id.

        Args:
            memory_id: The memory just accessed
            threshold: Minimum probability for prediction (default from settings)
            limit: Maximum predictions (default from settings)

        Returns:
            List of predicted memories with probabilities.
        """
        if not self.settings.predictive_cache_enabled:
            return []

        threshold = threshold if threshold is not None else self.settings.prediction_threshold
        limit = limit if limit is not None else self.settings.max_predictions

        patterns = self.get_access_patterns(memory_id, limit=limit * 2)

        results: list[PredictionResult] = []
        for pattern in patterns:
            if pattern.probability < threshold:
                continue

            memory = self.get_memory(pattern.to_memory_id)
            if memory is None:
                continue

            results.append(
                PredictionResult(
                    memory=memory,
                    probability=pattern.probability,
                    source_memory_id=memory_id,
                )
            )

            if len(results) >= limit:
                break

        return results

    def warm_predicted_cache(
        self,
        memory_id: int,
    ) -> list[int]:
        """Pre-warm hot cache with predicted next memories.

        Returns list of memory IDs that were promoted.
        """
        if not self.settings.predictive_cache_enabled:
            return []

        predictions = self.predict_next_memories(memory_id)
        promoted_ids: list[int] = []

        for pred in predictions:
            # Skip if already hot
            if pred.memory.is_hot:
                continue

            # Promote to hot cache
            if self.promote_to_hot(pred.memory.id, PromotionSource.PREDICTED):
                promoted_ids.append(pred.memory.id)
                log.debug(
                    "Predictively promoted memory {} (prob={:.2f} from {})",
                    pred.memory.id,
                    pred.probability,
                    memory_id,
                )

        return promoted_ids

    def get_all_access_patterns(
        self,
        min_count: int = 2,
        limit: int = 50,
    ) -> list[AccessPattern]:
        """Get all learned access patterns across all memories.

        Args:
            min_count: Minimum access count to include
            limit: Maximum patterns to return

        Returns:
            Patterns sorted by count descending.
        """
        with self._connection() as conn:
            # First get totals per source memory for probability calculation
            rows = conn.execute(
                """
                SELECT
                    s.from_memory_id,
                    s.to_memory_id,
                    s.count,
                    s.last_seen,
                    (SELECT SUM(count) FROM access_sequences
                     WHERE from_memory_id = s.from_memory_id) as total
                FROM access_sequences s
                WHERE s.count >= ?
                ORDER BY s.count DESC
                LIMIT ?
                """,
                (min_count, limit),
            ).fetchall()

            return [
                AccessPattern(
                    from_memory_id=row["from_memory_id"],
                    to_memory_id=row["to_memory_id"],
                    count=row["count"],
                    probability=row["count"] / row["total"] if row["total"] else 0,
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                )
                for row in rows
            ]

    def decay_old_sequences(self) -> int:
        """Decay access sequences older than sequence_decay_days.

        Reduces count by half for old sequences. Removes if count drops to 0.
        Returns number of sequences affected.
        """
        cutoff = datetime.now() - timedelta(days=self.settings.sequence_decay_days)

        with self.transaction() as conn:
            # Halve counts for old sequences
            conn.execute(
                """
                UPDATE access_sequences
                SET count = count / 2
                WHERE last_seen < ?
                """,
                (cutoff.isoformat(),),
            )
            affected = conn.execute("SELECT changes()").fetchone()[0]

            # Remove sequences with count = 0
            conn.execute("DELETE FROM access_sequences WHERE count = 0")
            deleted = conn.execute("SELECT changes()").fetchone()[0]

            if affected > 0 or deleted > 0:
                log.info("Decayed {} sequences, removed {} with zero count", affected, deleted)

            return affected

    # ========== Session (Conversation Provenance) Methods ==========

    def _update_session_activity(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        memory_delta: int = 0,
        log_delta: int = 0,
    ) -> None:
        """Update session activity counters. Creates session if needed."""
        # Upsert: insert or update session
        conn.execute(
            """
            INSERT INTO sessions (id, memory_count, log_count)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_activity_at = CURRENT_TIMESTAMP,
                memory_count = memory_count + excluded.memory_count,
                log_count = log_count + excluded.log_count
            """,
            (session_id, memory_delta, log_delta),
        )

    def create_or_get_session(
        self,
        session_id: str,
        topic: str | None = None,
        project_path: str | None = None,
    ) -> Session:
        """Create a new session or return existing one.

        Args:
            session_id: Unique session identifier (UUID or transcript hash)
            topic: Optional topic description
            project_path: Working directory for the session

        Returns:
            Session object (created or existing)
        """
        with self.transaction() as conn:
            # Try to get existing
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()

            if row:
                return self._row_to_session(row)

            # Create new session
            conn.execute(
                """
                INSERT INTO sessions (id, topic, project_path)
                VALUES (?, ?, ?)
                """,
                (session_id, topic, project_path),
            )

            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            log.info("Created session id={} topic={}", session_id, topic)
            return self._row_to_session(row)

    def update_session_topic(self, session_id: str, topic: str) -> bool:
        """Update the topic for a session."""
        with self.transaction() as conn:
            cursor = conn.execute("UPDATE sessions SET topic = ? WHERE id = ?", (topic, session_id))
            return cursor.rowcount > 0

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return self._row_to_session(row) if row else None

    def get_sessions(
        self,
        limit: int = 20,
        project_path: str | None = None,
    ) -> list[Session]:
        """Get recent sessions, optionally filtered by project.

        Args:
            limit: Maximum sessions to return
            project_path: Filter to sessions from this project

        Returns:
            List of sessions ordered by last activity (most recent first)
        """
        with self._connection() as conn:
            if project_path:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE project_path = ?
                    ORDER BY last_activity_at DESC
                    LIMIT ?
                    """,
                    (project_path, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    ORDER BY last_activity_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [self._row_to_session(row) for row in rows]

    def get_session_memories(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories from a specific session.

        Args:
            session_id: Session to get memories from
            limit: Maximum memories to return

        Returns:
            List of memories from the session, ordered by creation time
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT m.*, GROUP_CONCAT(t.tag, ',') as tags_str
                FROM memories m
                LEFT JOIN memory_tags t ON m.id = t.memory_id
                WHERE m.session_id = ?
                GROUP BY m.id
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

            memories = []
            for row in rows:
                tags_str = row["tags_str"]
                tags = tags_str.split(",") if tags_str else []
                memories.append(self._row_to_memory(row, conn, tags=tags))

            return memories

    def get_cross_session_patterns(self, min_sessions: int = 2) -> list[dict]:
        """Find content patterns that appear across multiple sessions.

        Useful for identifying frequently-discussed topics that might
        warrant promotion to hot cache.

        Args:
            min_sessions: Minimum sessions a pattern must appear in

        Returns:
            List of dicts with pattern info and session counts
        """
        with self._connection() as conn:
            # Find memories that appear in multiple sessions via similar content
            # This uses a simple approach: group by content_hash
            rows = conn.execute(
                """
                SELECT
                    content,
                    memory_type,
                    COUNT(DISTINCT session_id) as session_count,
                    SUM(access_count) as total_accesses,
                    GROUP_CONCAT(DISTINCT session_id) as sessions
                FROM memories
                WHERE session_id IS NOT NULL
                GROUP BY content_hash
                HAVING COUNT(DISTINCT session_id) >= ?
                ORDER BY session_count DESC, total_accesses DESC
                LIMIT 50
                """,
                (min_sessions,),
            ).fetchall()

            return [
                {
                    "content": row["content"][:200],  # Truncate for display
                    "memory_type": row["memory_type"],
                    "session_count": row["session_count"],
                    "total_accesses": row["total_accesses"],
                    "sessions": row["sessions"].split(",") if row["sessions"] else [],
                }
                for row in rows
            ]

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert a database row to a Session object."""
        return Session(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            last_activity_at=datetime.fromisoformat(row["last_activity_at"]),
            topic=row["topic"],
            project_path=row["project_path"],
            memory_count=row["memory_count"],
            log_count=row["log_count"],
        )

    # ========== Bootstrap Methods ==========

    def bootstrap_from_files(
        self,
        file_paths: list[Path],
        memory_type: MemoryType = MemoryType.PROJECT,
        promote_to_hot: bool = True,
        tags: list[str] | None = None,
    ) -> dict:
        """Seed memories from multiple files with deduplication.

        Reads each file, parses into chunks, and stores as memories.
        Handles edge cases gracefully (empty files, permission errors, etc.).

        Args:
            file_paths: List of file paths to process.
            memory_type: Type to assign to all created memories.
            promote_to_hot: Whether to promote new memories to hot cache.
            tags: Optional tags to apply to all memories.

        Returns:
            Dict with:
                - success: Always True (errors are reported, not raised)
                - files_found: Number of files in input
                - files_processed: Files successfully read
                - memories_created: New memories stored
                - memories_skipped: Duplicates (already existed)
                - hot_cache_promoted: Memories added to hot cache
                - errors: List of error messages
                - message: Human-readable summary
        """
        from memory_mcp.text_parsing import parse_content_into_chunks

        # Track results with explicit types to satisfy mypy
        errors: list[str] = []
        files_processed = 0
        memories_created = 0
        memories_skipped = 0
        hot_cache_promoted = 0

        tag_list = tags or []

        files_found = len(file_paths)

        for path in file_paths:
            # Check file exists
            if not path.exists():
                errors.append(f"{path.name}: file not found")
                continue

            # Check it's a file
            if not path.is_file():
                errors.append(f"{path.name}: not a file")
                continue

            # Detect binary files (skip them)
            if self._is_binary_file(path):
                errors.append(f"{path.name}: binary file skipped")
                continue

            # Read file content
            try:
                content = path.read_text(encoding="utf-8")
            except PermissionError:
                errors.append(f"{path.name}: permission denied")
                continue
            except UnicodeDecodeError:
                errors.append(f"{path.name}: encoding error (not UTF-8)")
                continue
            except OSError as e:
                errors.append(f"{path.name}: read error ({e})")
                continue

            # Skip empty files
            if not content.strip():
                log.debug("Skipping empty file: {}", path.name)
                continue

            files_processed += 1

            # Parse into chunks
            chunks = parse_content_into_chunks(content)

            for chunk in chunks:
                # Skip chunks that are too long
                if len(chunk) > self.settings.max_content_length:
                    errors.append(f"{path.name}: chunk too long ({len(chunk)} chars), skipped")
                    continue

                # Store the memory
                memory_id, is_new = self.store_memory(
                    content=chunk,
                    memory_type=memory_type,
                    source=MemorySource.MANUAL,
                    tags=tag_list,
                )

                if is_new:
                    memories_created += 1
                    if promote_to_hot:
                        if self.promote_to_hot(memory_id):
                            hot_cache_promoted += 1
                else:
                    memories_skipped += 1

        # Build summary message
        if files_found == 0:
            message = "No files provided. Pass file paths or use auto-detection."
        elif files_processed == 0 and files_found > 0:
            message = (
                f"No files could be processed from {files_found} provided. "
                "Check file permissions and paths."
            )
        elif memories_created == 0 and files_processed > 0:
            if memories_skipped > 0:
                message = (
                    f"All {memories_skipped} memories already exist from {files_processed} file(s)."
                )
            else:
                message = (
                    f"No memories extracted from {files_processed} file(s). "
                    "Files may be empty or contain only non-extractable content."
                )
        else:
            message = f"Bootstrapped {memories_created} memories from {files_processed} file(s)"
            if hot_cache_promoted > 0:
                message += f" ({hot_cache_promoted} promoted to hot cache)"

        log.info(
            "bootstrap_from_files: files={}/{} created={} skipped={} errors={}",
            files_processed,
            files_found,
            memories_created,
            memories_skipped,
            len(errors),
        )

        return {
            "success": True,
            "files_found": files_found,
            "files_processed": files_processed,
            "memories_created": memories_created,
            "memories_skipped": memories_skipped,
            "hot_cache_promoted": hot_cache_promoted,
            "errors": errors,
            "message": message,
        }

    def _is_binary_file(self, path: Path) -> bool:
        """Check if a file is binary by reading first bytes.

        Args:
            path: File path to check.

        Returns:
            True if file appears to be binary.
        """
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                # Check for null bytes (common in binary files)
                if b"\x00" in chunk:
                    return True
                # Check for very high ratio of non-text bytes
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
                non_text = sum(1 for byte in chunk if byte not in text_chars)
                return len(chunk) > 0 and non_text / len(chunk) > 0.30
        except OSError:
            return False  # Can't read, let the main read handle the error
