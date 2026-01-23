"""Hot cache mixin for Storage class."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from memory_mcp.logging import get_logger, get_promotion_rejection_summary
from memory_mcp.models import (
    AuditOperation,
    HotCacheMetrics,
    Memory,
    PromotionSource,
)

if TYPE_CHECKING:
    pass

log = get_logger("storage.hot_cache")


class HotCacheMixin:
    """Mixin providing hot cache methods for Storage."""

    # Type stub for attribute defined in Storage.__init__
    _hot_cache_metrics: HotCacheMetrics | None

    def _ensure_hot_cache_metrics(self) -> HotCacheMetrics:
        """Ensure hot cache metrics are loaded (lazy initialization from DB)."""
        if self._hot_cache_metrics is None:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'hot_cache_metrics'"
                ).fetchone()
                if row:
                    data = json.loads(row["value"])
                    self._hot_cache_metrics = HotCacheMetrics(
                        hits=data.get("hits", 0),
                        misses=data.get("misses", 0),
                        evictions=data.get("evictions", 0),
                        promotions=data.get("promotions", 0),
                    )
                else:
                    self._hot_cache_metrics = HotCacheMetrics()
        return self._hot_cache_metrics

    def _save_hot_cache_metrics(self) -> None:
        """Persist hot cache metrics to metadata table."""
        metrics = self._ensure_hot_cache_metrics()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('hot_cache_metrics', ?)
                """,
                (json.dumps(metrics.to_dict()),),
            )

    def get_promoted_memories(self, project_id: str | None = None) -> list[Memory]:
        """Get all memories in hot cache, ordered for optimal injection.

        Ordering prioritizes (most important first):
        1. last_used_at (session recency - memories recently marked helpful)
        2. decayed_trust (reliability signal with staleness penalty)
        3. real_usage_ratio (used_count / access_count - filters auto-marked noise)

        This provides cheap, non-embedding curation without per-request filtering.
        Uses decayed trust instead of raw trust_score to prevent stale items from
        dominating the injection despite not being used recently.

        Args:
            project_id: Optional project filter. If provided and project_filter_hot_cache
                is enabled, returns only memories for this project (+ global if
                project_include_global is enabled).
        """
        from memory_mcp.models import MemoryType

        memories = []
        with self._connection() as conn:
            # Build query with optional project filter
            if project_id and self.settings.project_filter_hot_cache:
                if self.settings.project_include_global:
                    query = """
                        SELECT * FROM memories
                        WHERE is_hot = 1 AND (project_id = ? OR project_id IS NULL)
                    """
                else:
                    query = "SELECT * FROM memories WHERE is_hot = 1 AND project_id = ?"
                rows = conn.execute(query, (project_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM memories WHERE is_hot = 1").fetchall()

            for row in rows:
                memories.append(self._row_to_memory(row, conn))

        # Sort by: session recency, decayed trust, real usage ratio
        # Uses decayed trust to penalize stale items even with high raw trust
        def sort_key(m: Memory) -> tuple:
            last_used_ts = m.last_used_at.timestamp() if m.last_used_at else 0

            # Compute decayed trust (penalizes staleness, rewards usage)
            memory_type = MemoryType(m.memory_type) if m.memory_type else None
            decayed_trust = self._compute_trust_decay(
                m.trust_score or 1.0,
                m.created_at,
                m.last_accessed_at,
                memory_type,
                m.access_count or 0,
            )

            # Real usage ratio: distinguish actual usage from auto-marking
            accessed = m.access_count or 1
            real_usage_ratio = (m.used_count or 0) / accessed

            return (-last_used_ts, -decayed_trust, -real_usage_ratio)

        memories.sort(key=sort_key)
        return memories

    # Alias for backwards compatibility
    def get_hot_memories(self, project_id: str | None = None) -> list[Memory]:
        """Alias for get_promoted_memories (backwards compatibility)."""
        return self.get_promoted_memories(project_id=project_id)

    def get_embeddings_for_memories(self, memory_ids: list[int]) -> dict[int, np.ndarray]:
        """Get embeddings for a list of memory IDs.

        Used for semantic clustering in display contexts.

        Args:
            memory_ids: List of memory IDs to fetch embeddings for.

        Returns:
            Dict mapping memory_id to embedding numpy array.
        """
        if not memory_ids:
            return {}

        with self._connection() as conn:
            placeholders = ",".join("?" * len(memory_ids))
            rows = conn.execute(
                f"SELECT rowid, embedding FROM memory_vectors WHERE rowid IN ({placeholders})",
                memory_ids,
            ).fetchall()

            return {row["rowid"]: np.frombuffer(row["embedding"], dtype=np.float32) for row in rows}

    def record_hot_cache_hit(self) -> None:
        """Record a hot cache hit (resource was read with content)."""
        self._ensure_hot_cache_metrics().hits += 1
        self._save_hot_cache_metrics()

    def record_hot_cache_miss(self) -> None:
        """Record a hot cache miss (resource was read but empty)."""
        self._ensure_hot_cache_metrics().misses += 1
        self._save_hot_cache_metrics()

    def get_hot_cache_metrics(self) -> HotCacheMetrics:
        """Get current hot cache metrics."""
        return self._ensure_hot_cache_metrics()

    def get_promoted_stats(self) -> dict:
        """Get promoted memories statistics including metrics and computed values."""
        promoted = self.get_promoted_memories()
        metrics = self._ensure_hot_cache_metrics().to_dict()

        # Compute average hot score
        avg_score = 0.0
        if promoted:
            scores = [m.hot_score for m in promoted if m.hot_score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            **metrics,
            "current_count": len(promoted),
            "max_items": self.settings.promoted_max_items,
            "avg_hot_score": round(avg_score, 3),
            "pinned_count": sum(1 for m in promoted if m.is_pinned),
            "promotion_rejections": get_promotion_rejection_summary(),
        }

    # Alias for backwards compatibility
    def get_hot_cache_stats(self) -> dict:
        """Alias for get_promoted_stats (backwards compatibility)."""
        return self.get_promoted_stats()

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
        def compute_score(row: sqlite3.Row) -> tuple[int, float]:
            last_accessed = row["last_accessed_at"]
            last_accessed_dt = datetime.fromisoformat(last_accessed) if last_accessed else None
            score = self._compute_hot_score(row["access_count"], last_accessed_dt)
            return (row["id"], score)

        candidates = [compute_score(row) for row in rows]
        return min(candidates, key=lambda x: x[1])[0]

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
                self._ensure_hot_cache_metrics().evictions += 1
                self._save_hot_cache_metrics()
                # Eviction indicates cache pressure - log as warning
                log.warning(
                    "Evicted memory id={} from hot cache (cache pressure, {} items)",
                    evict_id,
                    self.settings.hot_cache_max_items,
                )

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
                self._ensure_hot_cache_metrics().promotions += 1
                self._save_hot_cache_metrics()
                # Manual promotions are user actions (INFO), auto-promotions are routine (DEBUG)
                if promotion_source == PromotionSource.MANUAL:
                    log.info(
                        "Promoted memory id={} to hot cache (source={}, pinned={})",
                        memory_id,
                        promotion_source.value,
                        pin,
                    )
                else:
                    log.debug(
                        "Auto-promoted memory id={} to hot cache (source={}, pinned={})",
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

        Temporal-scope-aware demotion:
        - Durable memories (antipattern, landmine, convention, architecture): 2x base time
        - Stable memories (decision, preference, lesson, constraint): base time
        - Transient memories (context, bug, todo): 0.5x base time

        Returns list of demoted memory IDs.
        """
        if not self.settings.auto_demote:
            return []

        from memory_mcp.helpers import get_demotion_multiplier

        demoted_ids = []
        base_demotion_days = self.settings.demotion_days

        with self._connection() as conn:
            # Fetch all non-pinned hot memories with their category and last access
            rows = conn.execute(
                """
                SELECT id, category, COALESCE(last_accessed_at, created_at) as last_activity
                FROM memories
                WHERE is_hot = 1 AND is_pinned = 0
                """
            ).fetchall()

            stale_candidates = []
            for row in rows:
                category = row["category"]
                last_activity = datetime.fromisoformat(row["last_activity"])

                # Category-aware demotion time
                multiplier = get_demotion_multiplier(category)
                effective_demotion_days = int(base_demotion_days * multiplier)

                # Check if stale based on effective threshold
                days_since_access = (datetime.now() - last_activity).days
                if days_since_access >= effective_demotion_days:
                    stale_candidates.append(
                        (row["id"], category, effective_demotion_days, days_since_access)
                    )

        # Demote each (outside the read transaction)
        for memory_id, category, effective_days, days_since in stale_candidates:
            if self.demote_from_hot(memory_id):
                demoted_ids.append(memory_id)
                log.info(
                    "Auto-demoted stale memory id={} (category={}, days={}, threshold={})",
                    memory_id,
                    category,
                    days_since,
                    effective_days,
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
                            "base_demotion_days": base_demotion_days,
                            "note": "Category-aware thresholds applied",
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
