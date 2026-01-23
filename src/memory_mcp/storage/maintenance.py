"""Maintenance operations mixin for Storage class."""

from __future__ import annotations

import json
import os
import sqlite3

from memory_mcp.logging import get_logger
from memory_mcp.models import AuditOperation, MemoryType, TrustReason

log = get_logger("storage.maintenance")


class MaintenanceMixin:
    """Mixin providing maintenance methods for Storage."""

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
            MemoryType.EPISODIC: self.settings.retention_episodic_days,
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

    def _set_embedding_model_info(
        self, conn: sqlite3.Connection, model: str, dimension: int
    ) -> None:
        """Store embedding model info using existing connection.

        Internal method that accepts an existing connection to avoid
        nested transactions when called from within another transaction.
        """
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

    def set_embedding_model_info(self, model: str, dimension: int) -> None:
        """Store embedding model info for future validation."""
        with self.transaction() as conn:
            self._set_embedding_model_info(conn, model, dimension)

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

            # Update stored model info (using internal method to avoid nested transaction)
            model = self.settings.embedding_model
            self._set_embedding_model_info(conn, model, dim)

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
                        # Delete any existing vector first (vec0 doesn't support INSERT OR REPLACE)
                        conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (memory_id,))
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

    def penalize_low_utility_memories(self, min_retrievals: int = 5) -> list[int]:
        """Apply trust penalty to memories with poor helpfulness.

        Finds memories that have been retrieved at least min_retrievals times
        but never marked as used, and applies a LOW_UTILITY trust penalty.

        This helps demote memories that are frequently returned but never helpful,
        reducing their ranking in future recalls.

        Args:
            min_retrievals: Minimum retrieved_count to consider (default 5)

        Returns:
            List of memory IDs that were penalized
        """
        penalized_ids = []

        with self._connection() as conn:
            # Find low-utility memories: retrieved enough times but never used
            rows = conn.execute(
                """
                SELECT id FROM memories
                WHERE retrieved_count >= ?
                  AND used_count = 0
                """,
                (min_retrievals,),
            ).fetchall()

        for row in rows:
            memory_id = row["id"]
            result = self.adjust_trust(
                memory_id,
                reason=TrustReason.LOW_UTILITY,
                note=f"retrieved {min_retrievals}+ times but never used",
            )
            if result is not None:
                penalized_ids.append(memory_id)

        if penalized_ids:
            log.info(
                "Penalized {} low-utility memories (retrieved >= {} times, used = 0)",
                len(penalized_ids),
                min_retrievals,
            )

        return penalized_ids

    def run_full_cleanup(self) -> dict:
        """Run comprehensive cleanup: stale memories, old logs, patterns, injections.

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

        # 6. Clean up old injection records (7-day retention)
        deleted_injections = self.cleanup_old_injections(retention_days=7)

        # 7. Penalize low-utility memories (retrieved but never used)
        penalized_ids = self.penalize_low_utility_memories()

        # 8. Improve hot cache based on injection feedback (non-dry-run)
        injection_feedback = self.improve_hot_cache_from_injections(days=7, dry_run=False)

        return {
            "hot_cache_demoted": len(demoted_ids),
            "patterns_expired": expired_patterns,
            "logs_deleted": deleted_logs,
            "memories_deleted": memory_cleanup["total_deleted"],
            "memories_deleted_by_type": memory_cleanup["deleted_by_type"],
            "injections_deleted": deleted_injections,
            "low_utility_penalized": len(penalized_ids),
            "injection_feedback_promoted": len(injection_feedback.get("promoted", [])),
            "injection_feedback_warnings": len(injection_feedback.get("warnings", [])),
        }
