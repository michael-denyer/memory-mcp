"""SQLite storage with sqlite-vec for vector search."""

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


@dataclass
class Memory:
    """A stored memory."""

    id: int
    content: str
    content_hash: str
    memory_type: MemoryType
    source: MemorySource
    is_hot: bool
    tags: list[str]
    access_count: int
    last_accessed_at: datetime | None
    created_at: datetime
    similarity: float | None = None  # Populated during search


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


SCHEMA = """
-- All memories (hot + cold)
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE,
    memory_type TEXT NOT NULL,
    source TEXT NOT NULL,
    is_hot INTEGER DEFAULT 0,
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
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
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
        """Initialize database schema."""
        conn = self._conn
        if conn is None:
            return
        conn.executescript(SCHEMA)
        # Use embedding dimension from settings
        conn.execute(get_vector_schema(self.settings.embedding_dim))
        conn.commit()

        # Validate vector dimension matches settings
        self._validate_vector_dimension(conn)
        log.debug("Database schema initialized")

    def _validate_vector_dimension(self, conn: sqlite3.Connection) -> None:
        """Check that existing vector table matches configured dimension."""
        result = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_vectors'"
        ).fetchone()
        if result and result[0]:
            schema_sql = result[0]
            expected_dim = self.settings.embedding_dim
            # Check if dimension in schema matches
            if f"FLOAT[{expected_dim}]" not in schema_sql.upper():
                log.warning(
                    "Vector table dimension mismatch! Schema: {}, expected: {}. "
                    "Consider recreating the database or migrating.",
                    schema_sql,
                    expected_dim,
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

    # ========== Memory CRUD ==========

    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source: MemorySource = MemorySource.MANUAL,
        tags: list[str] | None = None,
        is_hot: bool = False,
    ) -> int:
        """Store a new memory with embedding."""
        hash_val = content_hash(content)

        with self.transaction() as conn:
            # Check if memory already exists
            existing = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (hash_val,)
            ).fetchone()

            if existing:
                # Update existing memory
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
                log.debug("Updated existing memory id={}", memory_id)
            else:
                # Insert new memory
                embedding = self._embedding_engine.embed(content)
                cursor = conn.execute(
                    """
                    INSERT INTO memories (content, content_hash, memory_type, source, is_hot)
                    VALUES (?, ?, ?, ?, ?)
                    RETURNING id
                    """,
                    (content, hash_val, memory_type.value, source.value, int(is_hot)),
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

                log.info("Stored new memory id={} type={}", memory_id, memory_type.value)

        return memory_id

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

        return Memory(
            id=row["id"],
            content=row["content"],
            content_hash=row["content_hash"],
            memory_type=MemoryType(row["memory_type"]),
            source=MemorySource(row["source"]),
            is_hot=bool(row["is_hot"]),
            tags=tags,
            access_count=row["access_count"],
            last_accessed_at=(
                datetime.fromisoformat(row["last_accessed_at"])
                if row["last_accessed_at"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            similarity=similarity,
        )

    def get_memory(self, memory_id: int) -> Memory | None:
        """Get a memory by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
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
            # Re-fetch with full data using get_memory (which acquires lock again, RLock allows this)
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

    def recall(
        self,
        query: str,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> RecallResult:
        """Semantic search with confidence gating."""
        # Use 'is None' to allow explicit 0 values
        if limit is None:
            limit = self.settings.default_recall_limit
        if threshold is None:
            threshold = self.settings.default_confidence_threshold

        query_embedding = self._embedding_engine.embed(query)

        with self._connection() as conn:
            # Vector similarity search - fetch extra for filtering
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
                    vec_distance_cosine(v.embedding, ?) as distance
                FROM memory_vectors v
                JOIN memories m ON m.id = v.rowid
                ORDER BY distance ASC
                LIMIT ?
                """,
                (query_embedding.tobytes(), limit * 2),
            ).fetchall()

            # Convert distance to similarity (cosine distance to similarity)
            memories = []
            gated_count = 0
            ids_to_update = []

            for row in rows:
                similarity = 1 - row["distance"]  # cosine distance to similarity

                if similarity >= threshold:
                    memories.append(self._row_to_memory(row, conn, similarity=similarity))
                    ids_to_update.append(row["id"])
                else:
                    gated_count += 1

            # Update access counts within the same lock
            for memory_id in ids_to_update:
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (memory_id,),
                )
            conn.commit()

        # Limit results
        memories = memories[:limit]

        # Determine confidence level
        if not memories:
            confidence = "low"
        elif memories[0].similarity and memories[0].similarity > self.settings.high_confidence_threshold:
            confidence = "high"
        else:
            confidence = "medium"

        log.debug(
            "Recall query='{}' returned {} results (confidence={}, gated={})",
            query[:50],
            len(memories),
            confidence,
            gated_count,
        )

        return RecallResult(
            memories=memories,
            confidence=confidence,
            gated_count=gated_count,
        )

    def _get_memories_by_ids(self, ids: list[int]) -> list[Memory]:
        """Fetch multiple memories by ID, filtering out None results."""
        return [m for mid in ids if (m := self.get_memory(mid)) is not None]

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
        """Get all memories in hot cache."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id FROM memories WHERE is_hot = 1 ORDER BY access_count DESC"
            ).fetchall()
            ids = [row["id"] for row in rows]

        return self._get_memories_by_ids(ids)

    def promote_to_hot(self, memory_id: int) -> bool:
        """Promote a memory to hot cache."""
        with self.transaction() as conn:
            # Check hot cache limit
            hot_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE is_hot = 1"
            ).fetchone()[0]

            if hot_count >= self.settings.hot_cache_max_items:
                # Demote least accessed hot memory
                conn.execute(
                    """
                    UPDATE memories SET is_hot = 0
                    WHERE id = (
                        SELECT id FROM memories
                        WHERE is_hot = 1
                        ORDER BY access_count ASC, last_accessed_at ASC
                        LIMIT 1
                    )
                    """
                )
                log.debug("Demoted least-accessed memory from hot cache (at limit)")

            cursor = conn.execute(
                "UPDATE memories SET is_hot = 1 WHERE id = ?", (memory_id,)
            )
            promoted = cursor.rowcount > 0
            if promoted:
                log.info("Promoted memory id={} to hot cache", memory_id)
            return promoted

    def demote_from_hot(self, memory_id: int) -> bool:
        """Remove a memory from hot cache."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE memories SET is_hot = 0 WHERE id = ?", (memory_id,)
            )
            demoted = cursor.rowcount > 0
            if demoted:
                log.info("Demoted memory id={} from hot cache", memory_id)
            return demoted

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
        """Log an output for pattern mining."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO output_log (content) VALUES (?)", (content,)
            )

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

    def upsert_mined_pattern(
        self, pattern: str, pattern_type: str
    ) -> int:
        """Insert or update a mined pattern."""
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
            cursor = conn.execute(
                "DELETE FROM mined_patterns WHERE id = ?", (pattern_id,)
            )
            return cursor.rowcount > 0
