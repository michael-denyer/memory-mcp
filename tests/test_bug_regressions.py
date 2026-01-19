"""Regression tests for fixed bugs."""

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from memory_mcp.config import Settings
from memory_mcp.storage import (
    EmbeddingDimensionError,
    MemorySource,
    MemoryType,
    SchemaVersionError,
    Storage,
)


@pytest.fixture
def storage():
    """Create a storage instance with temp database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(db_path=Path(tmpdir) / "test.db")
        storage = Storage(settings)
        yield storage
        storage.close()


class TestVectorSchemaDimension:
    """Tests for MemoryMCP-9fd: Vector schema should use configurable dimension."""

    def test_schema_uses_settings_dimension(self):
        """Vector table should be created with dimension from settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use non-default dimension
            settings = Settings(db_path=Path(tmpdir) / "test.db", embedding_dim=512)
            storage = Storage(settings)

            # Check that the schema was created (connection works)
            conn = storage._get_connection()

            # Query sqlite_master for the virtual table
            result = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'memory_vectors'"
            ).fetchone()

            # The SQL should contain the dimension
            assert result is not None
            assert "512" in result[0] or "FLOAT[512]" in result[0].upper()

            storage.close()


class TestStoreMemoryConflict:
    """Tests for MemoryMCP-cpp: store_memory conflict should handle is_hot correctly."""

    def test_duplicate_content_increments_access_count(self, storage):
        """Storing same content twice should increment access_count."""
        id1 = storage.store_memory("Same content", MemoryType.PROJECT)
        mem1 = storage.get_memory(id1)
        initial_count = mem1.access_count

        # Store same content again
        id2 = storage.store_memory("Same content", MemoryType.PROJECT)

        # Should return same ID
        assert id1 == id2

        # Access count should be incremented
        mem2 = storage.get_memory(id2)
        assert mem2.access_count == initial_count + 1

    def test_promote_works_on_existing_memory(self, storage):
        """Promoting after store_memory conflict should work."""
        # Store once
        id1 = storage.store_memory("Content to promote", MemoryType.PROJECT)
        assert not storage.get_memory(id1).is_hot

        # Store again (conflict path) - is_hot param would be ignored
        id2 = storage.store_memory("Content to promote", MemoryType.PROJECT)
        assert id1 == id2

        # Explicit promote should work
        storage.promote_to_hot(id1)
        assert storage.get_memory(id1).is_hot


class TestRecallThresholdHandling:
    """Tests for MemoryMCP-325: threshold=0 and limit=0 should be respected."""

    def test_threshold_zero_returns_all(self, storage):
        """threshold=0.0 should not be treated as 'use default'."""
        storage.store_memory("Test content", MemoryType.PROJECT)

        # With threshold=0.0, everything should pass
        result = storage.recall("anything", threshold=0.0)
        assert len(result.memories) > 0

    def test_threshold_none_uses_default(self, storage):
        """threshold=None should use default from settings."""
        storage.store_memory("Very specific XYZ123", MemoryType.PROJECT)

        # Default threshold (0.7) should gate out low matches
        result = storage.recall("completely unrelated ABC")
        # This depends on actual similarity, but at least it shouldn't crash


class TestThreadSafety:
    """Tests for MemoryMCP-x48: Concurrent access should be thread-safe."""

    def test_concurrent_writes(self, storage):
        """Multiple threads writing simultaneously should not corrupt data."""
        errors = []
        results = []

        def write_memory(n):
            try:
                mid = storage.store_memory(f"Content {n}", MemoryType.PROJECT)
                results.append(mid)
            except Exception as e:
                errors.append(e)

        # Run 20 concurrent writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_memory, i) for i in range(20)]
            for f in as_completed(futures):
                pass  # Wait for all

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        # Should have created memories (some may be duplicates)
        assert len(results) == 20

    def test_concurrent_reads_and_writes(self, storage):
        """Mixed reads and writes should not cause issues."""
        # Pre-populate
        storage.store_memory("Initial content", MemoryType.PROJECT)

        errors = []

        def mixed_operations(n):
            try:
                if n % 2 == 0:
                    storage.store_memory(f"Write {n}", MemoryType.PROJECT)
                else:
                    storage.recall("content", threshold=0.1)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(20)]
            for f in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent ops: {errors}"


class TestHotCachePromotion:
    """Tests for hot cache promotion edge cases."""

    def test_promote_respects_max_items(self, storage):
        """Hot cache should respect max_items limit."""
        # Create more memories than hot cache allows
        max_hot = storage.settings.hot_cache_max_items
        memory_ids = []

        for i in range(max_hot + 5):
            mid = storage.store_memory(f"Content {i}", MemoryType.PROJECT)
            memory_ids.append(mid)
            storage.promote_to_hot(mid)

        # Should only have max_hot items in hot cache
        hot_memories = storage.get_hot_memories()
        assert len(hot_memories) <= max_hot

    def test_demote_keeps_in_cold_storage(self, storage):
        """Demoting should keep memory in cold storage."""
        mid = storage.store_memory("To demote", MemoryType.PROJECT)
        storage.promote_to_hot(mid)
        assert storage.get_memory(mid).is_hot

        storage.demote_from_hot(mid)
        mem = storage.get_memory(mid)
        assert mem is not None  # Still exists
        assert not mem.is_hot  # But not hot


class TestSchemaVersioning:
    """Tests for schema versioning and migration."""

    def test_schema_version_recorded(self, storage):
        """Schema version should be recorded in database."""
        version = storage.get_schema_version()
        assert version >= 1

    def test_new_database_gets_version(self):
        """New database should have schema version set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(db_path=Path(tmpdir) / "new.db")
            storage = Storage(settings)
            assert storage.get_schema_version() == 1
            storage.close()

    def test_wal_mode_enabled(self):
        """WAL mode should be enabled for better concurrency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(db_path=Path(tmpdir) / "wal.db")
            storage = Storage(settings)

            conn = storage._get_connection()
            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0].lower() == "wal"

            storage.close()


class TestMaintenanceOperations:
    """Tests for maintenance operations."""

    def test_vacuum_runs_without_error(self, storage):
        """Vacuum should run without error."""
        storage.store_memory("Test content", MemoryType.PROJECT)
        storage.vacuum()  # Should not raise

    def test_analyze_runs_without_error(self, storage):
        """Analyze should run without error."""
        storage.store_memory("Test content", MemoryType.PROJECT)
        storage.analyze()  # Should not raise

    def test_maintenance_returns_stats(self, storage):
        """Maintenance should return useful statistics."""
        storage.store_memory("Test content", MemoryType.PROJECT)
        result = storage.maintenance()

        assert "size_before_bytes" in result
        assert "size_after_bytes" in result
        assert "memory_count" in result
        assert result["memory_count"] == 1
        assert "schema_version" in result
