"""Tests for v0.8 loop observability: mining_runs, probe, health, decay."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from memory_mcp.migrations import SCHEMA_VERSION
from memory_mcp.mining import run_mining
from memory_mcp.models import MemorySource, MemoryType
from memory_mcp.probe import run_probe
from memory_mcp.storage import Storage
from memory_mcp.storage.injection_tracking import distinctive_tokens
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


def _residue(storage) -> tuple[int, int, int]:
    with storage._connection() as conn:
        outputs = conn.execute(
            "SELECT COUNT(*) FROM output_log WHERE session_id = ?", (PROBE_SESSION_ID,)
        ).fetchone()[0]
        patterns = conn.execute(
            "SELECT COUNT(*) FROM mined_patterns WHERE pattern LIKE '%memory_probe_%'"
        ).fetchone()[0]
        memories = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE '%memory_probe_%'"
        ).fetchone()[0]
    return outputs, patterns, memories


class TestProbe:
    def test_round_trip_succeeds_on_clean_db(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            result = run_probe(storage)
            assert result.ok, f"failed at {result.stage}: {result.error}"
        finally:
            storage.close()

    def test_no_residue_after_success(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            run_probe(storage)
            assert _residue(storage) == (0, 0, 0)
            assert _count_runs(storage) == 0  # deviation 1: never records a run
        finally:
            storage.close()

    def test_failure_reports_stage(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            with patch("memory_mcp.probe.run_mining", side_effect=RuntimeError("wiring")):
                result = run_probe(storage)
            assert not result.ok
            assert result.stage == "mine"
            assert "wiring" in result.error
        finally:
            storage.close()

    def test_leftovers_excluded_from_health_and_swept_by_next_probe(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            with patch("memory_mcp.probe.run_mining", side_effect=RuntimeError("wiring")):
                run_probe(storage)  # leaves the sentinel output behind
            assert storage.get_loop_health()["outputs_24h"] == 0  # marker excluded
            result = run_probe(storage)  # pre-sweep + fresh round trip
            assert result.ok
            assert _residue(storage) == (0, 0, 0)
        finally:
            storage.close()


class TestDistinctiveTokens:
    def test_identifier_shapes_are_distinctive(self):
        tokens = distinctive_tokens("run deploy-staging via make_target or FooBarBaz v1.2.3")
        assert "deploy-staging" in tokens
        assert "make_target" in tokens
        assert "FooBarBaz" in tokens

    def test_common_words_are_not(self):
        assert distinctive_tokens("the quick brown foxes jumped over everything") == []


class TestUtilityDecay:
    def _mined_memory(self, storage, days_old=31, **overrides):
        memory_id, _ = storage.store_memory(
            content=f"mined fact {days_old} {overrides}",
            memory_type=MemoryType.PATTERN,
            source=MemorySource.MINED,
        )
        sets = {"created_at": _ts(days_ago=days_old), **overrides}
        assignments = ", ".join(f"{k} = ?" for k in sets)
        with storage.transaction() as conn:
            conn.execute(
                f"UPDATE memories SET {assignments} WHERE id = ?",
                (*sets.values(), memory_id),
            )
        return memory_id

    def _row(self, storage, memory_id):
        with storage._connection() as conn:
            return conn.execute(
                "SELECT is_hot, utility_score FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

    def test_demotes_and_floors_qualifying_memory(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            mid = self._mined_memory(storage, days_old=31, is_hot=1)
            result = storage.decay_unused_mined_memories()
            assert result["demoted"] == 1
            is_hot, utility = self._row(storage, mid)
            assert is_hot == 0 and utility == 0.0
            assert self._row(storage, mid) is not None  # nothing deleted
        finally:
            storage.close()

    def test_exemptions(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            pinned = self._mined_memory(storage, days_old=31, is_hot=1, is_pinned=1)
            young = self._mined_memory(storage, days_old=29, is_hot=1)
            retrieved = self._mined_memory(storage, days_old=31, is_hot=1, retrieved_count=2)
            used = self._mined_memory(storage, days_old=31, is_hot=1, used_count=1)
            storage.decay_unused_mined_memories()
            for mid in (pinned, young, retrieved, used):
                assert self._row(storage, mid)[0] == 1, f"memory {mid} wrongly demoted"
        finally:
            storage.close()

    def test_non_mined_sources_exempt(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            memory_id, _ = storage.store_memory(
                content="user seeded fact",
                memory_type=MemoryType.PATTERN,
                source=MemorySource.MANUAL,
            )
            with storage.transaction() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, is_hot = 1 WHERE id = ?",
                    (_ts(days_ago=40), memory_id),
                )
            storage.decay_unused_mined_memories()
            assert self._row(storage, memory_id)[0] == 1
        finally:
            storage.close()


class TestMarkUsedMemories:
    def _inject(self, storage, content):
        memory_id, _ = storage.store_memory(
            content=content, memory_type=MemoryType.PATTERN, source=MemorySource.MINED
        )
        storage.log_injection(memory_id, resource="hot-cache", session_id="s1")
        return memory_id

    def _used(self, storage, memory_id):
        with storage._connection() as conn:
            return conn.execute(
                "SELECT used_count, last_used_at FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

    def test_true_positive_bumps_used_count(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            mid = self._inject(storage, "use `make deploy-staging` for releases")
            n = storage.mark_used_memories("I ran make deploy-staging and it worked")
            assert n == 1
            used_count, last_used_at = self._used(storage, mid)
            assert used_count == 1 and last_used_at is not None
        finally:
            storage.close()

    def test_near_miss_common_words_do_not_match(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            mid = self._inject(storage, "always check the command output first")
            assert storage.mark_used_memories("the command failed with an error") == 0
            assert self._used(storage, mid)[0] == 0
        finally:
            storage.close()

    def test_injected_but_unused_untouched(self, temp_settings):
        storage = Storage(temp_settings)
        try:
            mid = self._inject(storage, "use `make deploy-staging` for releases")
            assert storage.mark_used_memories("unrelated response text entirely") == 0
            assert self._used(storage, mid)[0] == 0
        finally:
            storage.close()
