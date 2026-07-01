"""Tests for v0.8 loop observability: mining_runs, probe, health, decay."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from memory_mcp.migrations import SCHEMA_VERSION
from memory_mcp.mining import run_mining
from memory_mcp.storage import Storage
from memory_mcp.storage.mining_runs import PROBE_SESSION_ID


class TestMiningRunsMigration:
    def _columns(self, storage, table):
        with storage._connection() as conn:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}

    # Mirror the Storage import used by tests/test_storage.py if the package
    # __init__ does not re-export it.

    def test_fresh_db_has_mining_runs_table(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            cols = self._columns(storage, "mining_runs")
            assert cols == {
                "id",
                "started_at",
                "finished_at",
                "outputs_processed",
                "patterns_found",
                "memories_created",
                "error",
            }
            assert storage.get_schema_version() == SCHEMA_VERSION
        finally:
            storage.close()

    def test_v17_db_gains_table_on_reopen(self, temp_settings):
        storage = Storage(temp_settings)
        with storage.transaction() as conn:
            conn.execute("DROP TABLE mining_runs")
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (17)")
        storage.close()

        storage = Storage(temp_settings)
        try:
            assert "started_at" in self._columns(storage, "mining_runs")
            assert storage.get_schema_version() == SCHEMA_VERSION
        finally:
            storage.close()


def _ts(days_ago: float = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class TestMiningRunRecording:
    def test_records_successful_run(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            run_id = storage.record_mining_run(
                started_at=_ts(),
                finished_at=_ts(),
                stats={"outputs_processed": 4, "patterns_found": 2, "new_memories": 1},
            )
            with storage._connection() as conn:
                row = conn.execute(
                    "SELECT outputs_processed, patterns_found, memories_created, error"
                    " FROM mining_runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
            assert tuple(row) == (4, 2, 1, None)
        finally:
            storage.close()

    def test_records_failed_run(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            storage.record_mining_run(started_at=_ts(), finished_at=_ts(), stats={}, error="boom")
            with storage._connection() as conn:
                row = conn.execute("SELECT error FROM mining_runs").fetchone()
            assert row[0] == "boom"
        finally:
            storage.close()


class TestLoopHealth:
    def _seed_run(self, storage, days_ago, error=None):
        storage.record_mining_run(
            started_at=_ts(days_ago),
            finished_at=_ts(days_ago),
            stats={"outputs_processed": 1, "patterns_found": 1, "new_memories": 1},
            error=error,
        )

    def test_empty_db_is_amber(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            health = storage.get_loop_health()
            assert health["state"] == "amber"
            assert health["last_success_at"] is None
        finally:
            storage.close()

    def test_recent_success_is_green(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            self._seed_run(storage, days_ago=1)
            assert storage.get_loop_health()["state"] == "green"
        finally:
            storage.close()

    def test_stale_success_is_amber(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            self._seed_run(storage, days_ago=8)
            health = storage.get_loop_health()
            assert health["state"] == "amber"
            assert health["days_since_success"] >= 8
        finally:
            storage.close()

    def test_error_streak_is_red(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            self._seed_run(storage, days_ago=2)  # old success
            for _ in range(3):
                self._seed_run(storage, days_ago=0, error="boom")
            health = storage.get_loop_health()
            assert health["state"] == "red"
            assert health["consecutive_errors"] == 3
        finally:
            storage.close()

    def test_probe_outputs_excluded_from_counts(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            storage.log_output("real output content", session_id="real-session")
            storage.log_output("probe sentinel content", session_id=PROBE_SESSION_ID)
            health = storage.get_loop_health()
            assert health["outputs_24h"] == 1
            assert health["outputs_7d"] == 1
        finally:
            storage.close()


def _count_runs(storage) -> int:
    with storage._connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM mining_runs").fetchone()[0]


class TestRunMiningPersistence:
    def test_successful_run_is_recorded(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            storage.log_output("import numpy as np", session_id="s1")
            result = run_mining(storage, hours=1)
            with storage._connection() as conn:
                row = conn.execute(
                    "SELECT outputs_processed, patterns_found, memories_created, error"
                    " FROM mining_runs"
                ).fetchone()
            assert row[0] == result["outputs_processed"]
            assert row[1] == result["patterns_found"]
            assert row[2] == result["new_memories"]
            assert row[3] is None
        finally:
            storage.close()

    def test_failed_run_records_error_and_reraises(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            storage.log_output("import numpy as np", session_id="s1")
            with patch(
                "memory_mcp.mining.extract_patterns",
                side_effect=RuntimeError("extractor exploded"),
            ):
                with pytest.raises(RuntimeError):
                    run_mining(storage, hours=1)
            with storage._connection() as conn:
                row = conn.execute("SELECT error FROM mining_runs").fetchone()
            assert "extractor exploded" in row[0]
        finally:
            storage.close()

    def test_record_run_false_writes_nothing(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            run_mining(storage, hours=1, record_run=False)
            assert _count_runs(storage) == 0
        finally:
            storage.close()

    def test_session_id_scopes_outputs(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            storage.log_output("import alpha_only_lib", session_id="s-alpha")
            storage.log_output("import beta_only_lib", session_id="s-beta")
            result = run_mining(storage, hours=1, session_id="s-alpha", record_run=False)
            assert result["outputs_processed"] == 1
        finally:
            storage.close()
