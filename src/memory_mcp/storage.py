"""SQLite storage with sqlite-vec for vector search."""

import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator

import sqlite_vec

from memory_mcp.config import Settings, ensure_data_dir, get_settings
from memory_mcp.embeddings import content_hash, get_embedding_engine
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


class RecallMode(str, Enum):
    """Recall mode presets with different threshold/weight configurations."""

    PRECISION = "precision"  # High threshold, few results, prioritize similarity
    BALANCED = "balanced"  # Default balanced mode
    EXPLORATORY = "exploratory"  # Low threshold, more results, diverse factors


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
    # Computed scores (populated during search/recall)
    similarity: float | None = None  # Populated during search
    hot_score: float | None = None  # Computed score for LRU ranking
    # Recall scoring components (populated during recall)
    recency_score: float | None = None  # 0-1 based on age with decay
    trust_score_decayed: float | None = None  # Trust with time decay applied
    composite_score: float | None = None  # Combined ranking score


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


@dataclass
class RecallResult:
    """Result from recall operation with confidence gating."""

    memories: list[Memory]
    confidence: str  # "high", "medium", "low"
    gated_count: int  # How many results filtered by threshold
    mode: RecallMode | None = None  # Mode used for this recall
    guidance: str | None = None  # Hallucination prevention guidance


# Current schema version - increment when making breaking changes
SCHEMA_VERSION = 3

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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_hot ON memories(is_hot);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_output_log_timestamp ON output_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_mined_patterns_hash ON mined_patterns(pattern_hash);
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
        self._embedding_engine = get_embedding_engine()
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
        """Run full maintenance: vacuum and analyze. Returns stats."""
        with self._connection() as conn:
            # Get size before
            size_before = os.path.getsize(self.db_path) if self.db_path.exists() else 0

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

    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source: MemorySource = MemorySource.MANUAL,
        tags: list[str] | None = None,
        is_hot: bool = False,
        source_log_id: int | None = None,
    ) -> tuple[int, bool]:
        """Store a new memory with embedding.

        Args:
            content: The memory content
            memory_type: Type of memory (project, pattern, etc.)
            source: How memory was created (manual, mined)
            tags: Optional tags for categorization
            is_hot: Whether to add to hot cache immediately
            source_log_id: For mined memories, the originating output_log ID

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
            else:
                # Insert new memory with trust and provenance
                embedding = self._embedding_engine.embed(content)
                extracted_at = datetime.now().isoformat() if source == MemorySource.MINED else None
                cursor = conn.execute(
                    """
                    INSERT INTO memories (
                        content, content_hash, memory_type, source, is_hot,
                        trust_score, source_log_id, extracted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )
                memory_id = cursor.fetchone()[0]

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
        trust_score = self._get_row_value(row, "trust_score", 1.0) or 1.0
        source_log_id = self._get_row_value(row, "source_log_id")
        extracted_at_str = self._get_row_value(row, "extracted_at")
        extracted_at = datetime.fromisoformat(extracted_at_str) if extracted_at_str else None

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

    def _compute_trust_decay(self, base_trust: float, created_at: datetime) -> float:
        """Compute time-decayed trust score.

        Args:
            base_trust: Initial trust (1.0 for manual, 0.7 for mined by default)
            created_at: When the memory was created

        Returns:
            Trust score with exponential decay applied based on age.
        """
        halflife_days = self.settings.trust_decay_halflife_days
        days_old = (datetime.now() - created_at).total_seconds() / 86400
        decay = 2 ** (-days_old / halflife_days)
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
    ) -> float:
        """Compute weighted composite score for ranking.

        Combines semantic similarity with recency, access frequency, and trust.
        Trust weight is optional (default 0) for backwards compatibility.

        Args:
            similarity: Semantic similarity score (0-1)
            recency_score: Time-decayed recency score (0-1)
            access_score: Normalized access count (0-1)
            trust_score: Trust score with decay (0-1)
            weights: Optional custom weights from recall mode preset
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

        return (
            similarity * sim_weight
            + recency_score * rec_weight
            + access_score * acc_weight
            + trust_score * trust_weight
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
                        m.access_count,
                        m.last_accessed_at,
                        m.created_at,
                        m.trust_score,
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
                        m.access_count,
                        m.last_accessed_at,
                        m.created_at,
                        m.trust_score,
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
                    recency_score = self._compute_recency_score(created_at)
                    access_score = self._compute_access_score(row["access_count"], max_access)

                    # Compute trust with time decay
                    base_trust = row["trust_score"] or 1.0
                    trust_decayed = self._compute_trust_decay(base_trust, created_at)

                    composite_score = self._compute_composite_score(
                        similarity,
                        recency_score,
                        access_score,
                        trust_decayed,
                        weights=mode_config,
                    )

                    memory = self._row_to_memory(row, conn, similarity=similarity)
                    memory.recency_score = recency_score
                    memory.trust_score_decayed = trust_decayed
                    memory.composite_score = composite_score
                    candidates.append(memory)
                else:
                    gated_count += 1

            # Re-rank by composite score
            candidates.sort(key=lambda m: m.composite_score or 0, reverse=True)

            # Take top results and update access counts
            memories = candidates[:effective_limit]
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
            conn.commit()

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

    def log_output(self, content: str) -> int:
        """Log an output for pattern mining.

        Raises:
            ValidationError: If content is empty or exceeds max length.
        """
        # Defense-in-depth validation
        self._validate_content(content, "output content")

        with self.transaction() as conn:
            cursor = conn.execute("INSERT INTO output_log (content) VALUES (?)", (content,))

            # Cleanup old logs
            conn.execute(
                "DELETE FROM output_log WHERE timestamp < datetime('now', ?)",
                (f"-{self.settings.log_retention_days} days",),
            )

            log_id = cursor.lastrowid or 0
            log.debug("Logged output id={} ({} chars)", log_id, len(content))
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

    def upsert_mined_pattern(self, pattern: str, pattern_type: str) -> int:
        """Insert or update a mined pattern.

        Raises:
            ValidationError: If pattern is empty or exceeds max length.
        """
        # Defense-in-depth validation
        self._validate_content(pattern, "pattern")

        hash_val = content_hash(pattern)

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO mined_patterns (pattern, pattern_hash, pattern_type)
                VALUES (?, ?, ?)
                ON CONFLICT(pattern_hash) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (pattern, hash_val, pattern_type),
            )
            return cursor.fetchone()[0]

    def get_promotion_candidates(self, threshold: int | None = None) -> list[MinedPattern]:
        """Get mined patterns ready for promotion."""
        threshold = threshold or self.settings.promotion_threshold

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM mined_patterns
                WHERE occurrence_count >= ?
                ORDER BY occurrence_count DESC
                """,
                (threshold,),
            ).fetchall()

            return [
                MinedPattern(
                    id=row["id"],
                    pattern=row["pattern"],
                    pattern_hash=row["pattern_hash"],
                    pattern_type=row["pattern_type"],
                    occurrence_count=row["occurrence_count"],
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                )
                for row in rows
            ]

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
