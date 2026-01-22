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
