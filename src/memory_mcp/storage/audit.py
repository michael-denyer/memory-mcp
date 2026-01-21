"""Audit logging mixin for Storage class."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.models import AuditEntry, AuditOperation

if TYPE_CHECKING:
    pass

log = get_logger("storage.audit")


class AuditMixin:
    """Mixin providing audit logging methods for Storage."""

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
