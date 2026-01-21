"""Output logging mixin for Storage class."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger("storage.output_logging")


class OutputLoggingMixin:
    """Mixin providing output logging methods for Storage."""

    def log_output(self, content: str, session_id: str | None = None) -> int:
        """Log an output for pattern mining.

        Args:
            content: The output content to log.
            session_id: Optional session ID for provenance tracking.

        Raises:
            ValidationError: If content is empty or exceeds max length.
        """
        # Defense-in-depth validation
        self._validate_content(content, "output content")

        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO output_log (content, session_id) VALUES (?, ?)",
                (content, session_id),
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

    def get_recent_outputs(self, hours: int = 24) -> list[tuple[int, str, datetime]]:
        """Get recent output logs."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, content, timestamp FROM output_log
                WHERE timestamp > datetime('now', ?)
                ORDER BY timestamp DESC
                """,
                (f"-{hours} hours",),
            ).fetchall()

            return [
                (row["id"], row["content"], datetime.fromisoformat(row["timestamp"]))
                for row in rows
            ]
