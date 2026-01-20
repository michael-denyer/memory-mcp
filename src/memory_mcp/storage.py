"""SQLite storage with sqlite-vec for vector search."""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import sqlite_vec

from memory_mcp.config import Settings, ensure_data_dir, get_settings
from memory_mcp.embeddings import EmbeddingEngine, content_hash
from memory_mcp.logging import get_logger
from memory_mcp.migrations import (
    SCHEMA,
    SCHEMA_VERSION,
    EmbeddingDimensionError,
    SchemaVersionError,
    check_schema_version,
    get_vector_schema,
    run_migrations,
)

# Import all models from the dedicated module
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

# Re-export all models for backwards compatibility
# This allows existing code using `from memory_mcp.storage import MemoryType` to continue working
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
    "SchemaVersionError",
    "EmbeddingDimensionError",
    "ValidationError",
]

log = get_logger("storage")


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
        check_schema_version(conn)

        # Apply base schema first (uses IF NOT EXISTS, safe to re-run)
        conn.executescript(SCHEMA)
        conn.execute(get_vector_schema(self.settings.embedding_dim))

        # Get current version for migrations (now table exists)
        existing_version = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        current_version = existing_version[0] if existing_version else 0

        # Run migrations
        run_migrations(conn, current_version, self.settings)

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
                        f"Embedding dimension mismatch: database has {count} vectors with "
                        f"different dimension than configured ({expected_dim}).\n\n"
                        f"To fix this, use one of these options:\n"
                        f"  1. CLI: memory-mcp-cli db-rebuild-vectors\n"
                        f"  2. MCP tool: db_rebuild_vectors()\n"
                        f"  3. Set MEMORY_MCP_EMBEDDING_DIM to match existing vectors\n"
                        f"  4. Delete the database: {self.db_path}"
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

    # ========== Audit Logging ==========

    def _record_audit(
        self,
        conn: sqlite3.Connection,
        operation: AuditOperation,
        target_type: str | None = None,
        target_id: int | None = None,
        details: str | None = None,
    ) -> None:
        """Record a destructive operation in the audit log.

        Args:
            conn: Active database connection (should be in transaction).
            operation: The type of destructive operation.
            target_type: Type of target (memory, pattern, etc).
            target_id: ID of the affected target.
            details: JSON string with additional details (before/after state).
        """
        conn.execute(
            """
            INSERT INTO audit_log (operation, target_type, target_id, details)
            VALUES (?, ?, ?, ?)
            """,
            (operation.value, target_type, target_id, details),
        )

    def audit_history(
        self,
        limit: int = 50,
        operation: AuditOperation | None = None,
        target_type: str | None = None,
    ) -> list[AuditEntry]:
        """Get recent audit log entries.

        Args:
            limit: Maximum entries to return (default 50, max 500).
            operation: Filter by operation type.
            target_type: Filter by target type (memory, pattern, etc).

        Returns:
            List of audit entries, most recent first.
        """
        limit = min(limit, 500)

        with self._connection() as conn:
            query = (
                "SELECT id, operation, target_type, target_id, details, timestamp "
                "FROM audit_log WHERE 1=1"
            )
            params: list = []

            if operation:
                query += " AND operation = ?"
                params.append(operation.value)
            if target_type:
                query += " AND target_type = ?"
                params.append(target_type)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [
                AuditEntry(
                    id=row["id"],
                    operation=row["operation"],
                    target_type=row["target_type"],
                    target_id=row["target_id"],
                    details=row["details"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    def cleanup_old_audit_logs(self, retention_days: int = 30) -> int:
        """Delete audit log entries older than retention period.

        Args:
            retention_days: Days to keep audit logs (default 30).

        Returns:
            Number of entries deleted.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM audit_log WHERE timestamp < datetime('now', ?)",
                (f"-{retention_days} days",),
            )
            deleted = cursor.rowcount
            if deleted > 0:
                log.info(
                    "Deleted {} old audit log entries (older than {} days)", deleted, retention_days
                )
            return deleted

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

        result = {
            "size_before_bytes": size_before,
            "size_after_bytes": size_after,
            "bytes_reclaimed": size_before - size_after,
            "memory_count": memory_count,
            "vector_count": vector_count,
            "schema_version": self.get_schema_version(),
            "auto_demoted_count": len(demoted_ids),
            "auto_demoted_ids": demoted_ids,
        }

        # Record maintenance audit entry
        with self.transaction() as conn:
            self._record_audit(
                conn,
                AuditOperation.MAINTENANCE,
                details=json.dumps(
                    {
                        "bytes_reclaimed": result["bytes_reclaimed"],
                        "demoted_count": len(demoted_ids),
                    }
                ),
            )

        return result

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

        # Record summary audit entry
        total_deleted = sum(deleted_counts.values())
        if total_deleted > 0:
            with self.transaction() as conn:
                self._record_audit(
                    conn,
                    AuditOperation.CLEANUP_MEMORIES,
                    details=json.dumps({"deleted_by_type": deleted_counts, "total": total_deleted}),
                )

        return {
            "deleted_by_type": deleted_counts,
            "total_deleted": total_deleted,
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

    def clear_vectors(self) -> dict:
        """Clear all vectors from the database.

        Drops and recreates the vector table with current dimension.
        Memories are preserved but will have no vectors until rebuild.

        Returns:
            Stats about vectors cleared.
        """
        with self.transaction() as conn:
            # Count existing vectors
            count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]

            # Drop and recreate with current dimension
            # Use same schema as original (implicit rowid) for JOIN compatibility
            conn.execute("DROP TABLE IF EXISTS memory_vectors")
            dim = self.settings.embedding_dim
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE memory_vectors USING vec0(
                    embedding FLOAT[{dim}]
                )
                """
            )

            # Update stored model info
            model = self.settings.embedding_model
            self.set_embedding_model_info(model, dim)

            log.info("Cleared {} vectors, recreated table with dimension {}", count, dim)

            return {
                "vectors_cleared": count,
                "new_dimension": dim,
                "new_model": model,
            }

    def rebuild_vectors(self, batch_size: int = 100) -> dict:
        """Rebuild all vectors by re-embedding memories.

        Clears existing vectors and re-embeds all memories with the current
        embedding model. This is useful when changing embedding models or
        fixing dimension mismatches.

        Args:
            batch_size: Number of memories to embed per batch.

        Returns:
            Stats about the rebuild operation.
        """
        # First clear existing vectors
        clear_result = self.clear_vectors()

        # Get all memories that need embedding
        with self.transaction() as conn:
            memories = conn.execute("SELECT id, content FROM memories").fetchall()

        total = len(memories)
        embedded = 0
        errors = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch = memories[i : i + batch_size]
            memory_ids = [m[0] for m in batch]
            contents = [m[1] for m in batch]

            try:
                embeddings = self._embedding_engine.embed_batch(contents)

                with self.transaction() as conn:
                    for memory_id, embedding in zip(memory_ids, embeddings):
                        conn.execute(
                            "INSERT INTO memory_vectors (rowid, embedding) VALUES (?, ?)",
                            (memory_id, embedding.tobytes()),
                        )
                embedded += len(batch)
            except Exception as e:
                log.error("Failed to embed batch starting at {}: {}", i, e)
                errors += len(batch)

            if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                log.info("Rebuild progress: {}/{} memories", min(i + batch_size, total), total)

        log.info(
            "Vector rebuild complete: {} embedded, {} errors out of {} total",
            embedded,
            errors,
            total,
        )

        return {
            "vectors_cleared": clear_result["vectors_cleared"],
            "memories_total": total,
            "memories_embedded": embedded,
            "memories_failed": errors,
            "new_dimension": clear_result["new_dimension"],
            "new_model": clear_result["new_model"],
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

        # Compute importance score at admission time (MemGPT-inspired)
        importance_score = 0.5  # Default
        if self.settings.importance_scoring_enabled:
            from memory_mcp.helpers import compute_importance_score

            importance_score = compute_importance_score(
                content,
                self.settings.importance_length_weight,
                self.settings.importance_code_weight,
                self.settings.importance_entity_weight,
            )

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
                    trust_score, importance_score, source_log_id, extracted_at, session_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    content,
                    hash_val,
                    memory_type.value,
                    source.value,
                    int(is_hot),
                    trust_score,
                    importance_score,
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

    def _compute_salience_score(
        self,
        importance_score: float,
        trust_score: float,
        access_count: int,
        last_accessed_at: datetime | None,
    ) -> float:
        """Compute unified salience score for promotion/eviction decisions.

        Combines importance, trust, access frequency, and recency into a single
        metric (Engram-inspired). Used for smarter hot cache promotion than
        simple access count threshold.
        """
        from memory_mcp.helpers import compute_salience_score

        return compute_salience_score(
            importance_score=importance_score,
            trust_score=trust_score,
            access_count=access_count,
            last_accessed_at=last_accessed_at,
            importance_weight=self.settings.salience_importance_weight,
            trust_weight=self.settings.salience_trust_weight,
            access_weight=self.settings.salience_access_weight,
            recency_weight=self.settings.salience_recency_weight,
            recency_halflife_days=self.settings.salience_recency_halflife_days,
            max_access_count=self.settings.hot_cache_max_items,  # Normalize against cache size
        )

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
        importance_score_val = self._get_row_value(row, "importance_score", 0.5)
        importance_score = importance_score_val if importance_score_val is not None else 0.5
        source_log_id = self._get_row_value(row, "source_log_id")
        extracted_at_str = self._get_row_value(row, "extracted_at")
        extracted_at = datetime.fromisoformat(extracted_at_str) if extracted_at_str else None
        session_id = self._get_row_value(row, "session_id")

        hot_score = self._compute_hot_score(row["access_count"], last_accessed_dt)
        salience_score = self._compute_salience_score(
            importance_score, trust_score, row["access_count"], last_accessed_dt
        )

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
            importance_score=importance_score,
            source_log_id=source_log_id,
            extracted_at=extracted_at,
            session_id=session_id,
            similarity=similarity,
            hot_score=hot_score,
            salience_score=salience_score,
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
        # Capture content summary for audit before deletion
        memory = self.get_memory(memory_id)
        content_preview = memory.content[:100] if memory else None

        with self.transaction() as conn:
            conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (memory_id,))
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                self._record_audit(
                    conn,
                    AuditOperation.DELETE_MEMORY,
                    target_type="memory",
                    target_id=memory_id,
                    details=json.dumps({"content_preview": content_preview}),
                )
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
        - salience_score >= salience_promotion_threshold (Engram-inspired)
        - OR access_count >= promotion_threshold (legacy fallback)

        Returns True if memory was promoted.
        """
        if not self.settings.auto_promote:
            return False

        with self._connection() as conn:
            row = conn.execute(
                """SELECT is_hot, access_count, trust_score, importance_score, last_accessed_at
                   FROM memories WHERE id = ?""",
                (memory_id,),
            ).fetchone()

            if not row or row["is_hot"]:
                return False

            trust_score = row["trust_score"] or 1.0
            importance_score = row["importance_score"] or 0.5
            last_accessed_dt = (
                datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None
            )

            salience = self._compute_salience_score(
                importance_score, trust_score, row["access_count"], last_accessed_dt
            )

            meets_salience_threshold = salience >= self.settings.salience_promotion_threshold
            meets_access_threshold = row["access_count"] >= self.settings.promotion_threshold

            if not (meets_salience_threshold or meets_access_threshold):
                return False

            promoted = self.promote_to_hot(memory_id, PromotionSource.AUTO_THRESHOLD)
            if promoted:
                log.info(
                    "Auto-promoted memory id={} (salience={:.3f}, access_count={})",
                    memory_id,
                    salience,
                    row["access_count"],
                )
            return promoted

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
        expand_relations: bool | None = None,
    ) -> RecallResult:
        """Semantic search with confidence gating and composite ranking.

        Args:
            query: Search query for semantic similarity
            limit: Maximum results (overrides mode preset if set)
            threshold: Minimum similarity (overrides mode preset if set)
            mode: Recall mode preset (precision, balanced, exploratory)
            memory_types: Filter to specific memory types
            expand_relations: Expand results via knowledge graph (default from config)

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

        # Record access sequences for predictive cache (if enabled)
        if self.settings.predictive_cache_enabled and len(memories) >= 2:
            memory_ids = [m.id for m in memories]
            for i in range(len(memory_ids) - 1):
                self.record_access_sequence(memory_ids[i], memory_ids[i + 1])

        # Auto-warm hot cache with predicted next memories (based on top result)
        if self.settings.predictive_cache_enabled and memories:
            self.warm_predicted_cache(memories[0].id)

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

        # Expand via knowledge graph if enabled
        should_expand = (
            expand_relations
            if expand_relations is not None
            else self.settings.recall_expand_relations
        )
        if should_expand and memories:
            memories = self.expand_via_relations(memories)

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

    def expand_via_relations(
        self,
        memories: list[Memory],
        max_per_memory: int | None = None,
        decay_factor: float | None = None,
    ) -> list[Memory]:
        """Expand recall results by traversing knowledge graph relations.

        For each memory, finds related memories via the knowledge graph and adds
        them with a decayed score. This implements Engram-style associative recall
        where one memory activates related memories.

        Relation handling:
        - relates_to, depends_on, elaborates: Add related memory
        - contradicts: Add with flag for user awareness
        - supersedes: Prefer the superseding (newer) memory

        Args:
            memories: Initial recall results to expand
            max_per_memory: Max related memories per source (default from config)
            decay_factor: Score decay for expanded results (default from config)

        Returns:
            Expanded list with related memories appended (deduplicated).
        """
        max_expansion = max_per_memory or self.settings.recall_max_expansion
        decay = decay_factor or self.settings.recall_expansion_decay

        # Track already-included memory IDs to avoid duplicates
        seen_ids = {m.id for m in memories}
        expanded: list[Memory] = []

        for source_memory in memories:
            # Get related memories (1-hop only)
            related = self.get_related(source_memory.id, direction="both")

            added_count = 0
            for related_memory, relation in related:
                if added_count >= max_expansion:
                    break
                if related_memory.id in seen_ids:
                    continue

                # Handle supersedes specially - if this memory supersedes another,
                # we already have the newer version, skip the old one
                if relation.relation_type == RelationType.SUPERSEDES:
                    if relation.from_memory_id == source_memory.id:
                        # source supersedes related - skip related (it's outdated)
                        continue

                # Apply score decay from parent
                if source_memory.composite_score is not None:
                    related_memory.composite_score = source_memory.composite_score * decay
                if source_memory.similarity is not None:
                    related_memory.similarity = source_memory.similarity * decay

                expanded.append(related_memory)
                seen_ids.add(related_memory.id)
                added_count += 1

        return memories + expanded

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
                self._record_audit(
                    conn,
                    AuditOperation.DEMOTE_MEMORY,
                    target_type="memory",
                    target_id=memory_id,
                )
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

        # Record summary audit entry for batch demotion
        if demoted_ids:
            with self.transaction() as conn:
                self._record_audit(
                    conn,
                    AuditOperation.DEMOTE_STALE,
                    details=json.dumps(
                        {
                            "count": len(demoted_ids),
                            "memory_ids": demoted_ids,
                            "demotion_days": self.settings.demotion_days,
                        }
                    ),
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
        threshold = threshold if threshold is not None else self.settings.promotion_threshold

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
            deleted = cursor.rowcount > 0
            if deleted:
                self._record_audit(
                    conn,
                    AuditOperation.DELETE_PATTERN,
                    target_type="pattern",
                    target_id=pattern_id,
                )
            return deleted

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
                self._record_audit(
                    conn,
                    AuditOperation.EXPIRE_PATTERNS,
                    details=json.dumps({"count": expired, "days_inactive": days}),
                )
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
                self._record_audit(
                    conn,
                    AuditOperation.UNLINK_MEMORIES,
                    target_type="relationship",
                    details=json.dumps(
                        {
                            "from_memory_id": from_memory_id,
                            "to_memory_id": to_memory_id,
                            "relation_type": relation_type.value if relation_type else None,
                            "count_removed": count,
                        }
                    ),
                )
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
        # Use space-separated format to match SQLite CURRENT_TIMESTAMP
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

        with self.transaction() as conn:
            # Halve counts for old sequences
            conn.execute(
                """
                UPDATE access_sequences
                SET count = count / 2
                WHERE last_seen < ?
                """,
                (cutoff_str,),
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

                # Store the memory (catch validation errors to honor "errors reported not raised")
                try:
                    memory_id, is_new = self.store_memory(
                        content=chunk,
                        memory_type=memory_type,
                        source=MemorySource.MANUAL,
                        tags=tag_list,
                    )
                except ValidationError as e:
                    errors.append(f"{path.name}: {e}")
                    continue

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

    # ========== Retrieval Tracking (RAG-inspired) ==========

    def record_retrieval_event(
        self,
        query: str,
        memory_ids: list[int],
        similarities: list[float],
    ) -> list[int]:
        """Record which memories were retrieved for a query.

        Called after recall() to log which memories were returned.
        Enables tracking usage patterns for ranking feedback.

        Args:
            query: The recall query text
            memory_ids: IDs of memories returned
            similarities: Similarity scores for each memory

        Returns:
            List of retrieval event IDs
        """
        if not self.settings.retrieval_tracking_enabled:
            return []

        query_hash = content_hash(query)
        event_ids = []

        with self.transaction() as conn:
            for memory_id, similarity in zip(memory_ids, similarities):
                cursor = conn.execute(
                    """
                    INSERT INTO retrieval_events
                        (query_hash, memory_id, similarity, was_used, feedback)
                    VALUES (?, ?, ?, 0, NULL)
                    """,
                    (query_hash, memory_id, similarity),
                )
                event_ids.append(cursor.lastrowid)

        log.debug(
            "Recorded {} retrieval events for query_hash={}",
            len(event_ids),
            query_hash[:8],
        )
        return event_ids

    def mark_retrieval_used(
        self,
        memory_id: int,
        query: str | None = None,
        feedback: str | None = None,
    ) -> int:
        """Mark a retrieved memory as actually used by the LLM.

        Called when user/system confirms a memory was helpful.
        If query is provided, marks the specific retrieval event.
        Otherwise, marks the most recent retrieval for this memory.

        Args:
            memory_id: ID of the memory that was used
            query: Optional query to match specific retrieval
            feedback: Optional feedback (e.g., "helpful", "partially_helpful")

        Returns:
            Number of retrieval events updated
        """
        if not self.settings.retrieval_tracking_enabled:
            return 0

        with self.transaction() as conn:
            if query:
                query_hash = content_hash(query)
                cursor = conn.execute(
                    """
                    UPDATE retrieval_events
                    SET was_used = 1, feedback = COALESCE(?, feedback)
                    WHERE memory_id = ? AND query_hash = ?
                    """,
                    (feedback, memory_id, query_hash),
                )
            else:
                # Mark most recent retrieval for this memory
                cursor = conn.execute(
                    """
                    UPDATE retrieval_events
                    SET was_used = 1, feedback = COALESCE(?, feedback)
                    WHERE id = (
                        SELECT id FROM retrieval_events
                        WHERE memory_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    )
                    """,
                    (feedback, memory_id),
                )

            updated = cursor.rowcount
            if updated > 0:
                log.debug(
                    "Marked {} retrieval(s) as used for memory_id={}",
                    updated,
                    memory_id,
                )
            return updated

    def get_retrieval_stats(
        self,
        memory_id: int | None = None,
        days: int = 30,
    ) -> dict:
        """Get retrieval quality statistics.

        Args:
            memory_id: Optional memory ID to get stats for (None = all)
            days: How many days back to analyze

        Returns:
            Dictionary with retrieval quality metrics
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._connection() as conn:
            if memory_id:
                # Stats for specific memory
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_retrievals,
                        SUM(was_used) as used_count,
                        AVG(similarity) as avg_similarity,
                        AVG(CASE WHEN was_used = 1 THEN similarity END) as avg_used_sim
                    FROM retrieval_events
                    WHERE memory_id = ? AND created_at >= ?
                    """,
                    (memory_id, cutoff),
                ).fetchone()

                total = row["total_retrievals"] or 0
                used = row["used_count"] or 0
                usage_rate = used / total if total > 0 else 0.0

                return {
                    "memory_id": memory_id,
                    "days": days,
                    "total_retrievals": total,
                    "used_count": used,
                    "usage_rate": round(usage_rate, 3),
                    "avg_similarity": round(row["avg_similarity"] or 0, 3),
                    "avg_used_similarity": round(row["avg_used_sim"] or 0, 3),
                }
            else:
                # Global stats
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_retrievals,
                        SUM(was_used) as used_count,
                        COUNT(DISTINCT memory_id) as unique_memories,
                        COUNT(DISTINCT query_hash) as unique_queries,
                        AVG(similarity) as avg_similarity
                    FROM retrieval_events
                    WHERE created_at >= ?
                    """,
                    (cutoff,),
                ).fetchone()

                total = row["total_retrievals"] or 0
                used = row["used_count"] or 0
                usage_rate = used / total if total > 0 else 0.0

                # Top used memories
                top_used = conn.execute(
                    """
                    SELECT memory_id, COUNT(*) as retrieval_count,
                           SUM(was_used) as used_count
                    FROM retrieval_events
                    WHERE created_at >= ?
                    GROUP BY memory_id
                    ORDER BY used_count DESC
                    LIMIT 5
                    """,
                    (cutoff,),
                ).fetchall()

                # Least useful (retrieved but rarely used)
                least_useful = conn.execute(
                    """
                    SELECT memory_id, COUNT(*) as retrieval_count,
                           SUM(was_used) as used_count
                    FROM retrieval_events
                    WHERE created_at >= ?
                    GROUP BY memory_id
                    HAVING COUNT(*) >= 3 AND SUM(was_used) = 0
                    ORDER BY retrieval_count DESC
                    LIMIT 5
                    """,
                    (cutoff,),
                ).fetchall()

                return {
                    "days": days,
                    "total_retrievals": total,
                    "used_count": used,
                    "usage_rate": round(usage_rate, 3),
                    "unique_memories": row["unique_memories"] or 0,
                    "unique_queries": row["unique_queries"] or 0,
                    "avg_similarity": round(row["avg_similarity"] or 0, 3),
                    "top_used_memories": [
                        {
                            "memory_id": r["memory_id"],
                            "retrieval_count": r["retrieval_count"],
                            "used_count": r["used_count"],
                        }
                        for r in top_used
                    ],
                    "least_useful_memories": [
                        {
                            "memory_id": r["memory_id"],
                            "retrieval_count": r["retrieval_count"],
                        }
                        for r in least_useful
                    ],
                }

    def cleanup_old_retrieval_events(self, days: int = 90) -> int:
        """Remove old retrieval events to manage table size.

        Args:
            days: Delete events older than this many days

        Returns:
            Number of events deleted
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM retrieval_events WHERE created_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            if deleted > 0:
                log.info("Cleaned up {} old retrieval events (>{} days)", deleted, days)
            return deleted

    # ========== Memory Consolidation (MemoryBank-inspired) ==========

    def find_consolidation_clusters(
        self,
        memory_type: MemoryType | None = None,
        threshold: float | None = None,
        min_cluster_size: int | None = None,
    ) -> list[ConsolidationCluster]:
        """Find clusters of similar memories that could be consolidated.

        Uses vector similarity to group near-duplicates.

        Args:
            memory_type: Optional filter by memory type
            threshold: Similarity threshold (default from settings)
            min_cluster_size: Minimum memories to form cluster (default from settings)

        Returns:
            List of ConsolidationCluster objects
        """
        threshold = threshold or self.settings.consolidation_threshold
        min_size = min_cluster_size or self.settings.consolidation_min_cluster_size

        # Get all memories (optionally filtered)
        with self._connection() as conn:
            if memory_type:
                rows = conn.execute(
                    """
                    SELECT id, content, access_count, is_hot, is_pinned
                    FROM memories
                    WHERE memory_type = ?
                    ORDER BY access_count DESC
                    """,
                    (memory_type.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, content, access_count, is_hot, is_pinned
                    FROM memories
                    ORDER BY access_count DESC
                    """
                ).fetchall()

        if len(rows) < min_size:
            return []

        # Build similarity matrix using embeddings
        memory_ids = [r["id"] for r in rows]
        contents = [r["content"] for r in rows]
        access_counts = {r["id"]: r["access_count"] for r in rows}
        is_hot = {r["id"]: r["is_hot"] for r in rows}
        is_pinned = {r["id"]: r["is_pinned"] for r in rows}

        # Get embeddings for all memories
        embeddings = []
        for content in contents:
            emb = self._embedding_engine.embed(content)
            embeddings.append(emb)

        embeddings_array = np.array(embeddings)

        # Find clusters using greedy approach
        clusters: list[ConsolidationCluster] = []
        assigned: set[int] = set()

        for i, mem_id in enumerate(memory_ids):
            if mem_id in assigned:
                continue

            # Skip pinned memories as cluster seeds
            if is_pinned.get(mem_id):
                continue

            # Find similar memories
            cluster_members = [mem_id]
            similarities = []

            for j, other_id in enumerate(memory_ids):
                if i == j or other_id in assigned:
                    continue

                # Compute cosine similarity
                norm_i = np.linalg.norm(embeddings_array[i])
                norm_j = np.linalg.norm(embeddings_array[j])
                if norm_i > 0 and norm_j > 0:
                    sim = float(
                        np.dot(embeddings_array[i], embeddings_array[j]) / (norm_i * norm_j)
                    )
                else:
                    sim = 0.0

                if sim >= threshold:
                    cluster_members.append(other_id)
                    similarities.append(sim)

            if len(cluster_members) >= min_size:
                # Get tags for all members
                all_tags: set[str] = set()
                for mid in cluster_members:
                    memory = self.get_memory(mid)
                    if memory:
                        all_tags.update(memory.tags)

                # Choose representative: prefer hot, then highest access count
                representative = max(
                    cluster_members,
                    key=lambda mid: (is_hot.get(mid, 0), access_counts.get(mid, 0)),
                )

                avg_sim = sum(similarities) / len(similarities) if similarities else 1.0
                clusters.append(
                    ConsolidationCluster(
                        representative_id=representative,
                        member_ids=cluster_members,
                        avg_similarity=avg_sim,
                        total_access_count=sum(access_counts.get(m, 0) for m in cluster_members),
                        combined_tags=sorted(all_tags),
                    )
                )

                assigned.update(cluster_members)

        log.info(
            "Found {} consolidation clusters from {} memories",
            len(clusters),
            len(memory_ids),
        )
        return clusters

    def consolidate_cluster(
        self,
        cluster: ConsolidationCluster,
    ) -> dict:
        """Consolidate a cluster by merging members into representative.

        Args:
            cluster: The cluster to consolidate

        Returns:
            Dict with consolidation results
        """
        if len(cluster.member_ids) < 2:
            return {"success": False, "error": "Cluster too small"}

        # Get representative memory
        representative = self.get_memory(cluster.representative_id)
        if not representative:
            return {"success": False, "error": "Representative memory not found"}

        deleted_ids = []
        new_tags: set[str] = set()

        with self.transaction() as conn:
            # Update representative with combined tags
            existing_tags = set(representative.tags)
            new_tags = set(cluster.combined_tags) - existing_tags

            for tag in new_tags:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (cluster.representative_id, tag),
                )

            # Update access count to reflect combined usage
            conn.execute(
                """
                UPDATE memories
                SET access_count = ?,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (cluster.total_access_count, cluster.representative_id),
            )

            # Delete non-representative members
            for member_id in cluster.member_ids:
                if member_id != cluster.representative_id:
                    # Delete vectors
                    conn.execute(
                        "DELETE FROM memory_vectors WHERE rowid = ?",
                        (member_id,),
                    )
                    # Delete memory
                    conn.execute(
                        "DELETE FROM memories WHERE id = ?",
                        (member_id,),
                    )
                    deleted_ids.append(member_id)

            # Record in audit log
            self._record_audit(
                conn,
                AuditOperation.CLEANUP_MEMORIES,
                target_type="consolidation",
                target_id=cluster.representative_id,
                details=json.dumps(
                    {
                        "merged_count": len(deleted_ids),
                        "deleted_ids": deleted_ids,
                        "avg_similarity": cluster.avg_similarity,
                    }
                ),
            )

        log.info(
            "Consolidated cluster: kept id={}, deleted {} members",
            cluster.representative_id,
            len(deleted_ids),
        )

        return {
            "success": True,
            "representative_id": cluster.representative_id,
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "tags_added": list(new_tags),
        }

    def preview_consolidation(
        self,
        memory_type: MemoryType | None = None,
    ) -> dict:
        """Preview what consolidation would do without making changes.

        Args:
            memory_type: Optional filter by memory type

        Returns:
            Preview of consolidation results
        """
        clusters = self.find_consolidation_clusters(memory_type=memory_type)

        total_memories = sum(len(c.member_ids) for c in clusters)
        memories_to_delete = sum(len(c.member_ids) - 1 for c in clusters)

        pct = memories_to_delete / total_memories * 100 if total_memories > 0 else 0

        return {
            "cluster_count": len(clusters),
            "total_memories_in_clusters": total_memories,
            "memories_to_delete": memories_to_delete,
            "space_savings_pct": round(pct, 1),
            "clusters": [
                {
                    "representative_id": c.representative_id,
                    "member_count": len(c.member_ids),
                    "avg_similarity": round(c.avg_similarity, 3),
                    "total_access_count": c.total_access_count,
                }
                for c in clusters
            ],
        }

    def run_consolidation(
        self,
        memory_type: MemoryType | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Run consolidation on all eligible clusters.

        Args:
            memory_type: Optional filter by memory type
            dry_run: If True, only preview without making changes

        Returns:
            Consolidation results
        """
        if dry_run:
            return self.preview_consolidation(memory_type=memory_type)

        clusters = self.find_consolidation_clusters(memory_type=memory_type)

        results: dict = {
            "clusters_processed": 0,
            "memories_deleted": 0,
            "errors": [],
        }

        for cluster in clusters:
            result = self.consolidate_cluster(cluster)
            if result.get("success"):
                results["clusters_processed"] += 1
                results["memories_deleted"] += result.get("deleted_count", 0)
            else:
                results["errors"].append(result.get("error"))

        log.info(
            "Consolidation complete: {} clusters, {} memories deleted",
            results["clusters_processed"],
            results["memories_deleted"],
        )

        return results
