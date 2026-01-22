"""Retrieval tracking mixin for Storage class (RAG-inspired)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from memory_mcp.embeddings import content_hash
from memory_mcp.logging import get_logger
from memory_mcp.models import Memory

if TYPE_CHECKING:
    pass

log = get_logger("storage.retrieval")


class RetrievalMixin:
    """Mixin providing retrieval tracking methods for Storage."""

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

        Also updates denormalized counters on the memory:
        - Increments used_count
        - Sets last_used_at to current timestamp
        - Recomputes utility_score (Bayesian helpfulness)

        Args:
            memory_id: ID of the memory that was used
            query: Optional query to match specific retrieval
            feedback: Optional feedback (e.g., "helpful", "partially_helpful")

        Returns:
            Number of retrieval events updated (0 if tracking disabled, 1 if memory updated)
        """
        # Always update denormalized counters on the memory itself
        # even if retrieval_events tracking is disabled
        with self.transaction() as conn:
            # Update used_count and last_used_at on the memory
            conn.execute(
                """
                UPDATE memories
                SET used_count = used_count + 1,
                    last_used_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (memory_id,),
            )

            # Recompute Bayesian utility_score: (used + α) / (retrieved + α + β)
            # α=1 (prior successes), β=3 (assumes 25% base rate)
            conn.execute(
                """
                UPDATE memories
                SET utility_score = CAST(used_count + 1 AS REAL) / (retrieved_count + 4)
                WHERE id = ?
                """,
                (memory_id,),
            )

        if not self.settings.retrieval_tracking_enabled:
            log.debug("Marked memory {} as used", memory_id)
            return 1

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
            log.debug(
                "Marked memory {} as used ({} retrieval event(s))",
                memory_id,
                updated,
            )
            return max(1, updated)  # At least 1 for memory update

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

    def get_recent_recalls(self, limit: int = 5) -> list[Memory]:
        """Get memories from recent recall operations.

        Args:
            limit: Maximum memories to return

        Returns:
            List of recently recalled memories (most recent first)
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT m.*
                FROM memories m
                INNER JOIN (
                    SELECT DISTINCT memory_id, MAX(created_at) as last_used
                    FROM retrieval_events
                    WHERE was_used = 1
                    GROUP BY memory_id
                    ORDER BY last_used DESC
                    LIMIT ?
                ) r ON m.id = r.memory_id
                ORDER BY r.last_used DESC
                """,
                (limit,),
            ).fetchall()

            return [self._row_to_memory(row, conn) for row in rows]

    def get_working_set(self) -> list[Memory]:
        """Get the working set: recent recalls + predictions + top hot items.

        Combines:
        1. Recently recalled memories (from retrieval_events with was_used=1)
        2. Predicted next memories (from access patterns)
        3. Top salience hot memories (to fill remaining slots)

        Returns:
            List of memories for the working set, capped at working_set_max_items
        """
        if not self.settings.working_set_enabled:
            return []

        max_items = self.settings.working_set_max_items
        seen_ids: set[int] = set()
        working_set: list[Memory] = []

        def add_memory(memory: Memory) -> bool:
            """Add memory to working set if not seen and not full. Returns True if added."""
            if memory.id not in seen_ids and len(working_set) < max_items:
                working_set.append(memory)
                seen_ids.add(memory.id)
                return True
            return False

        # 1. Recent recalls (most valuable - user actually used these)
        recent_recalls = self.get_recent_recalls(
            limit=self.settings.working_set_recent_recalls_limit
        )
        for memory in recent_recalls:
            add_memory(memory)

        # 2. Predictions based on recent recalls (use top 3 as seeds)
        pred_limit = self.settings.working_set_predictions_limit
        for memory in recent_recalls[:3]:
            if len(working_set) >= max_items:
                break
            for pred in self.predict_next_memories(memory.id, limit=pred_limit):
                add_memory(pred.memory)

        # 3. Fill with top salience hot memories
        if len(working_set) < max_items:
            hot_memories = self._get_hot_memories_by_salience()
            for memory in hot_memories:
                add_memory(memory)

        log.debug(
            "Working set: {} recent recalls, {} total items",
            len(recent_recalls),
            len(working_set),
        )
        return working_set

    def _get_hot_memories_by_salience(self) -> list[Memory]:
        """Get hot memories sorted by salience score (highest first)."""
        hot_memories = self.get_hot_memories()
        for memory in hot_memories:
            memory.salience_score = self._compute_salience_score(
                importance_score=memory.importance_score,
                trust_score=memory.trust_score,
                access_count=memory.access_count,
                last_accessed_at=memory.last_accessed_at,
            )
        hot_memories.sort(key=lambda m: m.salience_score or 0, reverse=True)
        return hot_memories
