"""Mining run recording and loop health mixin for Storage class."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory_mcp.logging import get_logger

log = get_logger("storage.mining_runs")

STALENESS_DAYS = 7
ERROR_STREAK = 3
PROBE_SESSION_ID = "memory-mcp-probe"


class MiningRunsMixin:
    """Mixin providing mining run recording and loop health queries for Storage."""

    def record_mining_run(
        self,
        started_at: str,
        finished_at: str,
        stats: dict,
        error: str | None = None,
    ) -> int:
        """Record a single mining loop run.

        Args:
            started_at: UTC timestamp the run started, in the same format as
                `output_log.timestamp` (e.g. "2026-07-01 12:00:00").
            finished_at: UTC timestamp the run finished, same format.
            stats: Run statistics. Reads keys `outputs_processed`,
                `patterns_found`, and `new_memories`; missing keys default to 0.
            error: Error message if the run failed, else None.

        Returns:
            The id of the inserted `mining_runs` row.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO mining_runs (
                    started_at, finished_at, outputs_processed,
                    patterns_found, memories_created, error
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    finished_at,
                    stats.get("outputs_processed", 0),
                    stats.get("patterns_found", 0),
                    stats.get("new_memories", 0),
                    error,
                ),
            )
            run_id = cursor.lastrowid or 0
            log.debug(
                "Recorded mining run id={} error={}",
                run_id,
                error,
            )
            return run_id

    def get_loop_health(self) -> dict:
        """Compute learning-loop health from recorded mining runs and output logs.

        State precedence:
            1. "red" if the most recent `ERROR_STREAK` runs all have a
               non-null `error` (requires at least `ERROR_STREAK` runs total).
            2. "amber" if there is no successful run, or the last successful
               run is older than `STALENESS_DAYS`.
            3. "green" otherwise.

        Returns:
            Dict with keys `state`, `last_success_at`, `last_run_at`,
            `consecutive_errors`, `total_runs`, `outputs_24h`, `outputs_7d`,
            `patterns_7d`, `memories_7d`, `days_since_success`.
        """
        with self._connection() as conn:
            total_runs = conn.execute("SELECT COUNT(*) FROM mining_runs").fetchone()[0]

            last_run_row = conn.execute(
                "SELECT started_at FROM mining_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            last_run_at = last_run_row[0] if last_run_row else None

            last_success_row = conn.execute(
                """
                SELECT started_at FROM mining_runs
                WHERE error IS NULL
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
            last_success_at = last_success_row[0] if last_success_row else None

            recent_errors = conn.execute(
                "SELECT error FROM mining_runs ORDER BY id DESC LIMIT ?",
                (ERROR_STREAK,),
            ).fetchall()

            outputs_24h = conn.execute(
                """
                SELECT COUNT(*) FROM output_log
                WHERE timestamp > datetime('now', '-24 hours')
                  AND (session_id IS NULL OR session_id != ?)
                """,
                (PROBE_SESSION_ID,),
            ).fetchone()[0]

            outputs_7d = conn.execute(
                """
                SELECT COUNT(*) FROM output_log
                WHERE timestamp > datetime('now', '-7 days')
                  AND (session_id IS NULL OR session_id != ?)
                """,
                (PROBE_SESSION_ID,),
            ).fetchone()[0]

            window_row = conn.execute(
                """
                SELECT COALESCE(SUM(patterns_found), 0), COALESCE(SUM(memories_created), 0)
                FROM mining_runs
                WHERE error IS NULL
                  AND started_at > datetime('now', ?)
                """,
                (f"-{STALENESS_DAYS} days",),
            ).fetchone()
            patterns_7d, memories_7d = window_row[0], window_row[1]

        consecutive_errors = 0
        for row in recent_errors:
            if row[0] is not None:
                consecutive_errors += 1
            else:
                break

        # days_since_success stays whole-day for display (status/docs show
        # it), but amber onset must trigger at the exact STALENESS_DAYS
        # boundary, not at the day *after* whole-day truncation rounds down
        # to it - e.g. a 7.4-day-old success must already read amber ("no
        # successful run in 7 days"), which floor(7.4) > 7 == False would
        # otherwise miss until day 8.
        days_since_success = _days_since(last_success_at)
        stale = last_success_at is None or _seconds_since(last_success_at) > STALENESS_DAYS * 86400

        if len(recent_errors) >= ERROR_STREAK and all(row[0] is not None for row in recent_errors):
            state = "red"
        elif stale:
            state = "amber"
        else:
            state = "green"

        return {
            "state": state,
            "last_success_at": last_success_at,
            "last_run_at": last_run_at,
            "consecutive_errors": consecutive_errors,
            "total_runs": total_runs,
            "outputs_24h": outputs_24h,
            "outputs_7d": outputs_7d,
            "patterns_7d": patterns_7d,
            "memories_7d": memories_7d,
            "days_since_success": days_since_success,
        }


def _elapsed_since(timestamp: str) -> timedelta:
    """Compute the elapsed time since a stored UTC timestamp.

    Args:
        timestamp: A timestamp string in "%Y-%m-%d %H:%M:%S" UTC format.

    Returns:
        The `timedelta` between now and `timestamp`.
    """
    then = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - then


def _days_since(timestamp: str | None) -> int | None:
    """Compute whole days elapsed since a stored UTC timestamp.

    Used only for the `days_since_success` display field - state
    thresholds must use `_seconds_since` so the amber boundary isn't
    subject to whole-day truncation.

    Args:
        timestamp: A timestamp string in "%Y-%m-%d %H:%M:%S" UTC format, or None.

    Returns:
        Whole days elapsed, or None if timestamp is None.
    """
    if timestamp is None:
        return None
    return _elapsed_since(timestamp).days


def _seconds_since(timestamp: str) -> float:
    """Compute exact seconds elapsed since a stored UTC timestamp.

    Args:
        timestamp: A timestamp string in "%Y-%m-%d %H:%M:%S" UTC format.

    Returns:
        Seconds elapsed as a float.
    """
    return _elapsed_since(timestamp).total_seconds()
