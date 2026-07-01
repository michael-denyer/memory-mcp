"""Tests for v0.8 loop observability: mining_runs, probe, health, decay."""

from memory_mcp.migrations import SCHEMA_VERSION
from memory_mcp.storage import Storage


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
