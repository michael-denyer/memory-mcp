"""Mined patterns storage mixin for Storage class."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from memory_mcp.embeddings import content_hash
from memory_mcp.logging import get_logger
from memory_mcp.models import AuditOperation, MinedPattern, PatternStatus

if TYPE_CHECKING:
    pass

log = get_logger("storage.mining_store")


class MiningStoreMixin:
    """Mixin providing mined pattern storage methods for Storage."""

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
        # Handle memory_id column (may not exist in older databases before migration)
        memory_id = row["memory_id"] if "memory_id" in row.keys() else None
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
            memory_id=memory_id,
        )

    def link_pattern_to_memory(self, pattern_id: int, memory_id: int) -> bool:
        """Link a mined pattern to its created memory for exact-match promotion.

        Args:
            pattern_id: Pattern ID to update.
            memory_id: Memory ID that was created from this pattern.

        Returns:
            True if updated, False if pattern not found.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE mined_patterns SET memory_id = ? WHERE id = ?",
                (memory_id, pattern_id),
            )
            return cursor.rowcount > 0

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

    def delete_mined_patterns_bulk(
        self,
        pattern_ids: list[int] | None = None,
        pattern_type_prefix: str | None = None,
    ) -> int:
        """Delete multiple mined patterns at once.

        Args:
            pattern_ids: List of pattern IDs to delete. If provided, deletes
                these specific patterns.
            pattern_type_prefix: Delete patterns whose type starts with this
                prefix (e.g., 'entity_' matches entity_misc, entity_person).

        Returns:
            Number of patterns deleted.

        Raises:
            ValueError: If neither pattern_ids nor pattern_type_prefix is provided.
        """
        if not pattern_ids and not pattern_type_prefix:
            raise ValueError("Must provide either pattern_ids or pattern_type_prefix")

        with self.transaction() as conn:
            if pattern_ids:
                # Delete by specific IDs
                placeholders = ",".join("?" * len(pattern_ids))
                cursor = conn.execute(
                    f"DELETE FROM mined_patterns WHERE id IN ({placeholders})",
                    pattern_ids,
                )
            else:
                # Delete by pattern type prefix
                cursor = conn.execute(
                    "DELETE FROM mined_patterns WHERE pattern_type LIKE ?",
                    (f"{pattern_type_prefix}%",),
                )

            deleted = cursor.rowcount
            if deleted > 0:
                self._record_audit(
                    conn,
                    AuditOperation.DELETE_PATTERN,
                    details=json.dumps(
                        {
                            "bulk_delete": True,
                            "count": deleted,
                            "pattern_ids": pattern_ids[:10] if pattern_ids else None,
                            "pattern_type_prefix": pattern_type_prefix,
                        }
                    ),
                )
                log.info(
                    "Bulk deleted {} patterns (ids={}, type_prefix={})",
                    deleted,
                    pattern_ids[:5] if pattern_ids else None,
                    pattern_type_prefix,
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
