"""Injection tracking for hot cache and working set resources.

Tracks which memories were injected via MCP resources to enable:
- Feedback loop analysis (injection → used correlation)
- Dashboard injection history
- Auto-mark exploration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger("injection_tracking")


@dataclass
class InjectionRecord:
    """A record of a memory being injected via a resource."""

    id: int
    memory_id: int
    resource: str  # 'hot-cache' or 'working-set'
    injected_at: datetime
    session_id: str | None
    project_id: str | None


class InjectionTrackingMixin:
    """Mixin for injection tracking operations."""

    def log_injection(
        self,
        memory_id: int,
        resource: str,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> int:
        """Log that a memory was injected via a resource.

        Args:
            memory_id: ID of the injected memory
            resource: Resource name ('hot-cache' or 'working-set')
            session_id: Current session ID
            project_id: Current project ID

        Returns:
            ID of the injection log entry
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO injection_log (memory_id, resource, session_id, project_id)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, resource, session_id, project_id),
            )
            return cursor.lastrowid or 0

    def log_injections_batch(
        self,
        memory_ids: list[int],
        resource: str,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> int:
        """Log multiple memory injections in a single transaction.

        Args:
            memory_ids: IDs of injected memories
            resource: Resource name ('hot-cache' or 'working-set')
            session_id: Current session ID
            project_id: Current project ID

        Returns:
            Number of injections logged
        """
        if not memory_ids:
            return 0

        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO injection_log (memory_id, resource, session_id, project_id)
                VALUES (?, ?, ?, ?)
                """,
                [(mid, resource, session_id, project_id) for mid in memory_ids],
            )
            return len(memory_ids)

    def get_recent_injections(
        self,
        memory_id: int | None = None,
        days: int = 7,
        resource: str | None = None,
        limit: int = 100,
    ) -> list[InjectionRecord]:
        """Get recent injection records.

        Args:
            memory_id: Filter to specific memory (None for all)
            days: How many days back to look
            resource: Filter by resource type
            limit: Maximum records to return

        Returns:
            List of InjectionRecord objects
        """
        with self._connection() as conn:
            query = """
                SELECT id, memory_id, resource, injected_at, session_id, project_id
                FROM injection_log
                WHERE injected_at >= datetime('now', ?)
            """
            params: list = [f"-{days} days"]

            if memory_id is not None:
                query += " AND memory_id = ?"
                params.append(memory_id)

            if resource is not None:
                query += " AND resource = ?"
                params.append(resource)

            query += " ORDER BY injected_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

            return [
                InjectionRecord(
                    id=row["id"],
                    memory_id=row["memory_id"],
                    resource=row["resource"],
                    injected_at=datetime.fromisoformat(row["injected_at"]),
                    session_id=row["session_id"],
                    project_id=row["project_id"],
                )
                for row in rows
            ]

    def was_recently_injected(
        self,
        memory_id: int,
        days: int = 7,
        resource: str | None = None,
    ) -> bool:
        """Check if a memory was recently injected.

        Args:
            memory_id: Memory to check
            days: How many days back to look
            resource: Filter by resource type

        Returns:
            True if memory was injected within the time window
        """
        with self._connection() as conn:
            query = """
                SELECT 1 FROM injection_log
                WHERE memory_id = ?
                AND injected_at >= datetime('now', ?)
            """
            params: list = [memory_id, f"-{days} days"]

            if resource is not None:
                query += " AND resource = ?"
                params.append(resource)

            query += " LIMIT 1"

            return conn.execute(query, params).fetchone() is not None

    def get_injection_count(
        self,
        memory_id: int,
        days: int = 7,
    ) -> int:
        """Get count of injections for a memory.

        Args:
            memory_id: Memory to count injections for
            days: How many days back to look

        Returns:
            Number of times the memory was injected
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count FROM injection_log
                WHERE memory_id = ?
                AND injected_at >= datetime('now', ?)
                """,
                (memory_id, f"-{days} days"),
            ).fetchone()
            return row["count"] if row else 0

    def get_injection_stats(self, days: int = 7) -> dict:
        """Get injection statistics.

        Args:
            days: How many days back to analyze

        Returns:
            Dictionary with injection stats
        """
        with self._connection() as conn:
            # Total injections
            total = conn.execute(
                """
                SELECT COUNT(*) as count FROM injection_log
                WHERE injected_at >= datetime('now', ?)
                """,
                (f"-{days} days",),
            ).fetchone()

            # By resource
            by_resource = conn.execute(
                """
                SELECT resource, COUNT(*) as count FROM injection_log
                WHERE injected_at >= datetime('now', ?)
                GROUP BY resource
                """,
                (f"-{days} days",),
            ).fetchall()

            # Unique memories injected
            unique_memories = conn.execute(
                """
                SELECT COUNT(DISTINCT memory_id) as count FROM injection_log
                WHERE injected_at >= datetime('now', ?)
                """,
                (f"-{days} days",),
            ).fetchone()

            # Today's injections
            today = conn.execute(
                """
                SELECT COUNT(*) as count FROM injection_log
                WHERE injected_at >= datetime('now', 'start of day')
                """,
            ).fetchone()

            return {
                "total_injections": total["count"] if total else 0,
                "by_resource": {row["resource"]: row["count"] for row in by_resource},
                "unique_memories": unique_memories["count"] if unique_memories else 0,
                "today": today["count"] if today else 0,
                "days": days,
            }

    def cleanup_old_injections(self, retention_days: int = 7) -> int:
        """Remove injection records older than retention period.

        Args:
            retention_days: How many days to keep

        Returns:
            Number of records deleted
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM injection_log
                WHERE injected_at < datetime('now', ?)
                """,
                (f"-{retention_days} days",),
            )
            deleted = cursor.rowcount
            if deleted > 0:
                log.info("Cleaned up {} old injection records", deleted)
            return deleted

    def analyze_injection_patterns(self, days: int = 7) -> dict:
        """Analyze injection patterns to identify high-value and low-utility memories.

        Correlates injection counts with actual usage to identify:
        1. High-value: Frequently injected AND frequently used (should stay hot)
        2. Low-utility: Frequently injected but rarely/never used (consider demotion)
        3. Co-injected pairs: Memories that frequently appear together

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with pattern analysis results
        """
        with self._connection() as conn:
            # Get memories with injection counts and usage correlation
            memory_stats = conn.execute(
                """
                SELECT
                    i.memory_id,
                    COUNT(*) as injection_count,
                    m.used_count,
                    m.retrieved_count,
                    m.utility_score,
                    m.is_hot,
                    m.category
                FROM injection_log i
                JOIN memories m ON m.id = i.memory_id
                WHERE i.injected_at >= datetime('now', ?)
                GROUP BY i.memory_id
                ORDER BY injection_count DESC
                """,
                (f"-{days} days",),
            ).fetchall()

            # Categorize memories
            high_value = []  # Injected AND used
            low_utility = []  # Injected but not used
            candidates_for_promotion = []  # Not hot but high injection+usage

            for row in memory_stats:
                injection_count = row["injection_count"]
                used_count = row["used_count"] or 0
                utility_score = row["utility_score"] or 0.25

                entry = {
                    "memory_id": row["memory_id"],
                    "injection_count": injection_count,
                    "used_count": used_count,
                    "utility_score": round(utility_score, 3),
                    "is_hot": bool(row["is_hot"]),
                    "category": row["category"],
                }

                # High value: injected 3+ times and used at least once
                if injection_count >= 3 and used_count > 0:
                    high_value.append(entry)

                # Low utility: injected 5+ times but never used
                if injection_count >= 5 and used_count == 0:
                    low_utility.append(entry)

                # Candidate for promotion: not hot but high injection AND usage
                if not row["is_hot"] and injection_count >= 5 and utility_score >= 0.3:
                    candidates_for_promotion.append(entry)

            # Find co-injected pairs (memories that appear together within same minute)
            co_injected = conn.execute(
                """
                SELECT
                    a.memory_id as memory_a,
                    b.memory_id as memory_b,
                    COUNT(*) as co_occurrence
                FROM injection_log a
                JOIN injection_log b ON a.resource = b.resource
                    AND a.memory_id < b.memory_id
                    AND a.injected_at >= datetime('now', ?)
                    AND b.injected_at >= datetime('now', ?)
                    AND abs(julianday(a.injected_at) - julianday(b.injected_at)) < 0.0007
                GROUP BY a.memory_id, b.memory_id
                HAVING COUNT(*) >= 3
                ORDER BY co_occurrence DESC
                LIMIT 10
                """,
                (f"-{days} days", f"-{days} days"),
            ).fetchall()

            co_pairs = [
                {
                    "memory_a": row["memory_a"],
                    "memory_b": row["memory_b"],
                    "co_occurrence": row["co_occurrence"],
                }
                for row in co_injected
            ]

            return {
                "days": days,
                "total_memories_analyzed": len(memory_stats),
                "high_value": high_value[:20],  # Top 20
                "low_utility": low_utility[:20],  # Top 20
                "candidates_for_promotion": candidates_for_promotion[:10],
                "co_injected_pairs": co_pairs,
                "summary": {
                    "high_value_count": len(high_value),
                    "low_utility_count": len(low_utility),
                    "promotion_candidates": len(candidates_for_promotion),
                    "co_injection_pairs": len(co_pairs),
                },
            }

    def improve_hot_cache_from_injections(self, days: int = 7, dry_run: bool = True) -> dict:
        """Use injection patterns to improve hot cache quality.

        Takes action based on injection analysis:
        1. Promote frequently-injected, high-utility memories
        2. Log warnings for injected-but-never-used memories (optional demotion)

        Args:
            days: Number of days to analyze
            dry_run: If True, only report what would be done (default True)

        Returns:
            Dictionary with actions taken or recommended
        """
        from memory_mcp.models import PromotionSource

        analysis = self.analyze_injection_patterns(days)
        actions = {
            "dry_run": dry_run,
            "promoted": [],
            "warnings": [],
        }

        # Promote candidates that are frequently injected with good utility
        for candidate in analysis["candidates_for_promotion"]:
            memory_id = candidate["memory_id"]
            if dry_run:
                actions["promoted"].append(
                    {
                        "action": "would_promote",
                        **candidate,
                    }
                )
            else:
                if self.promote_to_hot(memory_id, PromotionSource.FEEDBACK):
                    actions["promoted"].append(
                        {
                            "action": "promoted",
                            **candidate,
                        }
                    )
                    log.info(
                        "Promoted memory {} based on injection feedback "
                        "(injected={}, utility={:.2f})",
                        memory_id,
                        candidate["injection_count"],
                        candidate["utility_score"],
                    )

        # Warn about low-utility memories (don't auto-demote - too risky)
        for low_util in analysis["low_utility"]:
            memory_id = low_util["memory_id"]
            actions["warnings"].append(
                {
                    "memory_id": memory_id,
                    "injection_count": low_util["injection_count"],
                    "message": "Injected but never used - consider demotion",
                }
            )
            log.warning(
                "Memory {} injected {} times but never used (is_hot={})",
                memory_id,
                low_util["injection_count"],
                low_util["is_hot"],
            )

        return actions
