"""Trust score management mixin for Storage class."""

from __future__ import annotations

from datetime import datetime

from memory_mcp.logging import get_logger
from memory_mcp.models import (
    TRUST_REASON_DEFAULTS,
    MemoryType,
    PromotionSource,
    TrustEvent,
    TrustReason,
)

log = get_logger("storage.trust")


class TrustMixin:
    """Mixin providing trust score management methods for Storage."""

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
            MemoryType.EPISODIC: self.settings.trust_decay_episodic_days,
        }
        return type_halflife_days.get(memory_type, self.settings.trust_decay_halflife_days)

    def check_auto_promote(self, memory_id: int) -> bool:
        """Check if memory should be auto-promoted and do so if eligible.

        Auto-promotes if:
        - auto_promote is enabled in settings
        - memory is not already hot
        - salience_score >= threshold (category-aware: lower for antipattern/landmine)
        - OR access_count >= promotion_threshold (legacy fallback)
        - AND helpfulness check passes (if enough retrievals):
          - If retrieved_count >= 5: used_rate must be >= 0.25
          - If retrieved_count < 5: passes by default (cold start benefit of doubt)

        High-value categories (antipattern, landmine) use lower thresholds
        so critical warnings surface early in plans.

        Returns True if memory was promoted.
        """
        if not self.settings.auto_promote:
            return False

        with self._connection() as conn:
            row = conn.execute(
                """SELECT is_hot, access_count, trust_score, importance_score,
                          last_accessed_at, category, retrieved_count, used_count
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
            category = row["category"]

            salience = self._compute_salience_score(
                importance_score, trust_score, row["access_count"], last_accessed_dt
            )

            # Category-aware threshold: high-value categories (antipattern, landmine)
            # have lower thresholds to promote more eagerly
            from memory_mcp.helpers import get_promotion_salience_threshold

            effective_threshold = get_promotion_salience_threshold(
                category, self.settings.salience_promotion_threshold
            )

            meets_salience_threshold = salience >= effective_threshold
            meets_access_threshold = row["access_count"] >= self.settings.promotion_threshold

            if not (meets_salience_threshold or meets_access_threshold):
                return False

            # Helpfulness gate: if we have enough retrieval data, require minimum used_rate
            retrieved_count = row["retrieved_count"] or 0
            used_count = row["used_count"] or 0
            helpfulness_warmup_threshold = 5  # Require 5 retrievals before gating

            if retrieved_count >= helpfulness_warmup_threshold:
                used_rate = used_count / retrieved_count
                min_used_rate = 0.25  # Require 25% usage rate
                if used_rate < min_used_rate:
                    log.debug(
                        "Skipped promotion for memory id={} (used_rate={:.2f} < {:.2f})",
                        memory_id,
                        used_rate,
                        min_used_rate,
                    )
                    return False

            promoted = self.promote_to_hot(memory_id, PromotionSource.AUTO_THRESHOLD)
            if promoted:
                log.info(
                    "Auto-promoted memory id={} (salience={:.3f}, threshold={:.3f}, category={})",
                    memory_id,
                    salience,
                    effective_threshold,
                    category,
                )
            return promoted
