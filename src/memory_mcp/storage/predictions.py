"""Predictive hot cache warming mixin for Storage class."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.models import AccessPattern, PredictionResult, PromotionSource

if TYPE_CHECKING:
    pass

log = get_logger("storage.predictions")


class PredictionsMixin:
    """Mixin providing predictive cache warming methods for Storage."""

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
