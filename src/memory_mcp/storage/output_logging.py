"""Output logging mixin for Storage class."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.redaction import redact_secrets

if TYPE_CHECKING:
    pass

log = get_logger("storage.output_logging")


class OutputLoggingMixin:
    """Mixin providing output logging methods for Storage."""

    def log_output(
        self, content: str, session_id: str | None = None, project_id: str | None = None
    ) -> int:
        """Log an output for pattern mining.

        Args:
            content: The output content to log.
            session_id: Optional session ID for provenance tracking.
            project_id: Optional project ID for project-scoped mining.

        Raises:
            ValidationError: If content is empty or exceeds max length.

        Note:
            Secrets are redacted BEFORE storage to prevent persistence.
            This ensures secrets never appear in dashboard, recall, or exports.
        """
        # Defense-in-depth validation
        self._validate_content(content, "output content")

        # Redact secrets BEFORE storage to prevent persistence
        # This is critical - secrets should never be stored in output_log
        redacted_content = redact_secrets(content)

        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO output_log (content, session_id, project_id) VALUES (?, ?, ?)",
                (redacted_content, session_id, project_id),
            )

            # Update session log count if session_id provided (upsert creates if needed)
            if session_id:
                self._update_session_activity(conn, session_id, log_delta=1)

            # Cleanup old logs
            conn.execute(
                "DELETE FROM output_log WHERE timestamp < datetime('now', ?)",
                (f"-{self.settings.log_retention_days} days",),
            )

            log_id = cursor.lastrowid or 0
            log.debug(
                "Logged output id={} ({} chars) session={}",
                log_id,
                len(content),
                session_id,
            )
            return log_id

    def get_recent_outputs(
        self, hours: int = 24, project_id: str | None = None
    ) -> list[tuple[int, str, datetime, str | None]]:
        """Get recent output logs, optionally filtered by project.

        Args:
            hours: How many hours back to look.
            project_id: If provided, only return logs from this project.
                        If None, returns all logs (backwards compatible).

        Returns:
            List of tuples: (log_id, content, timestamp, project_id).
            The project_id is preserved for each log so mining can use the
            source project rather than the current session's project.
        """
        with self._connection() as conn:
            if project_id:
                rows = conn.execute(
                    """
                    SELECT id, content, timestamp, project_id FROM output_log
                    WHERE timestamp > datetime('now', ?)
                      AND project_id = ?
                    ORDER BY timestamp DESC
                    """,
                    (f"-{hours} hours", project_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, content, timestamp, project_id FROM output_log
                    WHERE timestamp > datetime('now', ?)
                    ORDER BY timestamp DESC
                    """,
                    (f"-{hours} hours",),
                ).fetchall()

            return [
                (
                    row["id"],
                    row["content"],
                    datetime.fromisoformat(row["timestamp"]),
                    row["project_id"],
                )
                for row in rows
            ]
