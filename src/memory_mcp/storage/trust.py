"""Trust score management mixin for Storage class."""

from __future__ import annotations

from datetime import datetime

from memory_mcp.logging import get_logger, record_promotion_rejection
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

            # Significant trust changes (|delta| >= 0.1) are INFO, minor tweaks are DEBUG
            if abs(delta) >= 0.1:
                log.info(
                    "Trust change id={}: {:.2f} -> {:.2f} (reason={}, delta={:.3f})",
                    memory_id,
                    old_trust,
                    new_trust,
                    reason.value,
                    delta,
                )
            else:
                log.debug(
                    "Trust tweak id={}: {:.2f} -> {:.2f} (reason={}, delta={:.3f})",
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
        - category is eligible for promotion (command, snippet never promoted)
        - salience_score >= threshold (category-aware: lower for antipattern/landmine/constraint)
        - OR access_count >= promotion_threshold (legacy fallback)
        - AND Bayesian helpfulness check passes:
          - Uses Beta-Binomial posterior for smooth cold-start handling
          - New memories (retrieved < 3) get benefit of doubt
          - Heavily-retrieved-but-unused memories (helpfulness < 0.20) are blocked

        High-value categories (antipattern, landmine, constraint) use lower thresholds
        so critical warnings and guardrails surface early in plans.

        Auto-pinning: High-value category memories with high trust (>= 0.8) are
        automatically pinned to prevent eviction.

        Low-value categories (command, snippet) are never promoted since they're
        easily discoverable or have low recall value.

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

            category = row["category"]

            # Import helper functions for promotion decisions
            from memory_mcp.helpers import (
                get_bayesian_helpfulness,
                get_promotion_salience_threshold,
                should_auto_pin,
                should_promote_category,
            )

            # Category gate: block low-value categories (command, snippet)
            if not should_promote_category(category):
                log.debug(
                    "Skipped promotion for memory id={} (category '{}' ineligible)",
                    memory_id,
                    category,
                )
                record_promotion_rejection("category_ineligible", memory_id)
                return False

            trust_score = row["trust_score"] or 1.0
            importance_score = row["importance_score"] or 0.5
            last_accessed_dt = (
                datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None
            )

            salience = self._compute_salience_score(
                importance_score, trust_score, row["access_count"], last_accessed_dt
            )

            # Category-aware threshold: high-value categories (antipattern, landmine, constraint)
            # have lower thresholds to promote more eagerly
            effective_threshold = get_promotion_salience_threshold(
                category, self.settings.salience_promotion_threshold
            )

            meets_salience_threshold = salience >= effective_threshold
            meets_access_threshold = row["access_count"] >= self.settings.promotion_threshold

            if not (meets_salience_threshold or meets_access_threshold):
                log.debug(
                    "Skipped promotion for memory id={} (salience={:.3f} < {:.3f}, "
                    "access={} < {}, category={})",
                    memory_id,
                    salience,
                    effective_threshold,
                    row["access_count"],
                    self.settings.promotion_threshold,
                    category,
                )
                record_promotion_rejection("threshold_not_met", memory_id)
                return False

            # Helpfulness gate: use Bayesian helpfulness to avoid cold-start trap
            # New memories get benefit of doubt via prior; heavily-retrieved-but-unused fail
            retrieved_count = row["retrieved_count"] or 0
            used_count = row["used_count"] or 0
            bayesian_helpfulness = get_bayesian_helpfulness(used_count, retrieved_count)
            min_helpfulness = 0.20  # Bayesian threshold (stricter than raw 25% due to prior)

            # Only gate if we have meaningful retrieval data (avoid blocking brand new memories)
            if retrieved_count >= 3 and bayesian_helpfulness < min_helpfulness:
                log.debug(
                    "Skipped promotion for memory id={} (bayesian_helpfulness={:.2f} < {:.2f}, "
                    "used={}, retrieved={})",
                    memory_id,
                    bayesian_helpfulness,
                    min_helpfulness,
                    used_count,
                    retrieved_count,
                )
                record_promotion_rejection("low_helpfulness", memory_id)
                return False

            # Check if this high-value memory should be auto-pinned
            auto_pin = should_auto_pin(category, trust_score)

            promoted = self.promote_to_hot(memory_id, PromotionSource.AUTO_THRESHOLD, pin=auto_pin)
            if promoted:
                log.debug(
                    "Auto-promoted memory id={} (salience={:.3f}, threshold={:.3f}, "
                    "category={}, pinned={})",
                    memory_id,
                    salience,
                    effective_threshold,
                    category,
                    auto_pin,
                )
            return promoted
