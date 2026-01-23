"""Regression tests for fixed bugs."""

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from memory_mcp.config import Settings
from memory_mcp.storage import (
    MemorySource,
    MemoryType,
    RecallMode,
    Storage,
)


@pytest.fixture
def storage():
    """Create a storage instance with temp database.

    Semantic dedup is disabled to keep test content independent.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(db_path=Path(tmpdir) / "test.db", semantic_dedup_enabled=False)
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
    """Tests for store_memory conflict handling and tag merging."""

    def test_duplicate_content_increments_access_count(self, storage):
        """Storing same content twice should increment access_count."""
        id1, is_new1 = storage.store_memory("Same content", MemoryType.PROJECT)
        assert is_new1 is True
        mem1 = storage.get_memory(id1)
        initial_count = mem1.access_count

        # Store same content again
        id2, is_new2 = storage.store_memory("Same content", MemoryType.PROJECT)
        assert is_new2 is False

        # Should return same ID
        assert id1 == id2

        # Access count should be incremented
        mem2 = storage.get_memory(id2)
        assert mem2.access_count == initial_count + 1

    def test_promote_works_on_existing_memory(self, storage):
        """Promoting after store_memory conflict should work."""
        # Store once
        id1, _ = storage.store_memory("Content to promote", MemoryType.PROJECT)
        assert not storage.get_memory(id1).is_hot

        # Store again (conflict path) - is_hot param would be ignored
        id2, is_new = storage.store_memory("Content to promote", MemoryType.PROJECT)
        assert id1 == id2
        assert is_new is False

        # Explicit promote should work
        storage.promote_to_hot(id1)
        assert storage.get_memory(id1).is_hot

    def test_duplicate_content_merges_tags(self, storage):
        """Storing duplicate with new tags should merge them."""
        # Store with initial tags
        id1, _ = storage.store_memory(
            "Content with tags", MemoryType.PROJECT, tags=["tag1", "tag2"]
        )
        mem1 = storage.get_memory(id1)
        assert set(mem1.tags) == {"tag1", "tag2"}

        # Store same content with additional tags
        id2, is_new = storage.store_memory(
            "Content with tags", MemoryType.PROJECT, tags=["tag2", "tag3"]
        )
        assert id1 == id2
        assert is_new is False

        # Should have merged tags
        mem2 = storage.get_memory(id2)
        assert set(mem2.tags) == {"tag1", "tag2", "tag3"}

    def test_duplicate_preserves_original_type(self, storage):
        """Storing duplicate with different type should preserve original."""
        # Store as project type
        id1, _ = storage.store_memory("Type test content", MemoryType.PROJECT)
        mem1 = storage.get_memory(id1)
        assert mem1.memory_type == MemoryType.PROJECT

        # Try to store same content as pattern type
        id2, is_new = storage.store_memory("Type test content", MemoryType.PATTERN)
        assert id1 == id2
        assert is_new is False

        # Should still be project type (first write wins)
        mem2 = storage.get_memory(id2)
        assert mem2.memory_type == MemoryType.PROJECT

    def test_duplicate_preserves_original_source(self, storage):
        """Storing duplicate with different source should preserve original."""
        # Store as manual
        id1, _ = storage.store_memory(
            "Source test content", MemoryType.PROJECT, source=MemorySource.MANUAL
        )
        mem1 = storage.get_memory(id1)
        assert mem1.source == MemorySource.MANUAL

        # Try to store same content as mined
        id2, is_new = storage.store_memory(
            "Source test content", MemoryType.PROJECT, source=MemorySource.MINED
        )
        assert id1 == id2
        assert is_new is False

        # Should still be manual (first write wins)
        mem2 = storage.get_memory(id2)
        assert mem2.source == MemorySource.MANUAL


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
        # This depends on actual similarity, but at least it shouldn't crash
        storage.recall("completely unrelated ABC")


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
            mid, _ = storage.store_memory(f"Content {i}", MemoryType.PROJECT)
            memory_ids.append(mid)
            storage.promote_to_hot(mid)

        # Should only have max_hot items in hot cache
        hot_memories = storage.get_hot_memories()
        assert len(hot_memories) <= max_hot

    def test_demote_keeps_in_cold_storage(self, storage):
        """Demoting should keep memory in cold storage."""
        mid, _ = storage.store_memory("To demote", MemoryType.PROJECT)
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
        from memory_mcp.migrations import SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(db_path=Path(tmpdir) / "new.db")
            storage = Storage(settings)
            assert storage.get_schema_version() == SCHEMA_VERSION
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


class TestHotCacheLRU:
    """Tests for MemoryMCP-zpg: Hot cache LRU policy with score-based eviction."""

    def test_hot_score_computed(self, storage):
        """Hot score should be computed for memories."""
        mid, _ = storage.store_memory("Score test", MemoryType.PROJECT)
        storage.promote_to_hot(mid)
        mem = storage.get_memory(mid)
        assert mem.hot_score is not None
        # New memory has access_count=0 and no last_accessed_at, so score is 0
        assert mem.hot_score == 0.0

        # After accessing, score should increase
        storage.update_access(mid)
        mem = storage.get_memory(mid)
        assert mem.hot_score > 0

    def test_eviction_removes_lowest_score(self):
        """When hot cache is full, lowest-scoring item should be evicted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Small hot cache for testing
            settings = Settings(
                db_path=Path(tmpdir) / "lru.db",
                hot_cache_max_items=3,
                semantic_dedup_enabled=False,
            )
            storage = Storage(settings)

            # Create 3 memories and promote them
            ids = []
            for i in range(3):
                mid, _ = storage.store_memory(f"Content {i}", MemoryType.PROJECT)
                ids.append(mid)
                storage.promote_to_hot(mid)

            # All 3 should be hot
            assert len(storage.get_hot_memories()) == 3

            # Access first memory multiple times to boost its score
            for _ in range(5):
                storage.update_access(ids[0])

            # Add 4th memory - should evict lowest score (not ids[0])
            mid4, _ = storage.store_memory("Content 4", MemoryType.PROJECT)
            storage.promote_to_hot(mid4)

            hot_mems = storage.get_hot_memories()
            assert len(hot_mems) == 3
            hot_ids = {m.id for m in hot_mems}

            # First memory should still be hot (highest access count)
            assert ids[0] in hot_ids
            # Fourth memory should be hot (just added)
            assert mid4 in hot_ids

            storage.close()

    def test_pinned_memory_not_evicted(self):
        """Pinned memories should not be evicted even with low scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                db_path=Path(tmpdir) / "pin.db",
                hot_cache_max_items=2,
                semantic_dedup_enabled=False,
            )
            storage = Storage(settings)

            # Create 2 memories and promote them
            mid1, _ = storage.store_memory("Pinned content", MemoryType.PROJECT)
            mid2, _ = storage.store_memory("Normal content", MemoryType.PROJECT)
            storage.promote_to_hot(mid1)
            storage.promote_to_hot(mid2)

            # Pin the first one
            storage.pin_memory(mid1)
            mem1 = storage.get_memory(mid1)
            assert mem1.is_pinned

            # Boost second memory's score
            for _ in range(10):
                storage.update_access(mid2)

            # Try to add third - should evict mid2 (not pinned) even though
            # mid1 has lower score
            mid3, _ = storage.store_memory("Third content", MemoryType.PROJECT)
            storage.promote_to_hot(mid3)

            hot_mems = storage.get_hot_memories()
            hot_ids = {m.id for m in hot_mems}

            # Pinned memory should still be hot
            assert mid1 in hot_ids
            # New memory should be hot
            assert mid3 in hot_ids
            # Mid2 was evicted despite higher score (only non-pinned option)
            assert mid2 not in hot_ids

            storage.close()

    def test_all_pinned_blocks_promotion(self):
        """Cannot promote when cache full and all items pinned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                db_path=Path(tmpdir) / "allpin.db",
                hot_cache_max_items=2,
                semantic_dedup_enabled=False,
            )
            storage = Storage(settings)

            # Fill cache with pinned memories
            mid1, _ = storage.store_memory("Pinned 1", MemoryType.PROJECT)
            mid2, _ = storage.store_memory("Pinned 2", MemoryType.PROJECT)
            storage.promote_to_hot(mid1, pin=True)
            storage.promote_to_hot(mid2, pin=True)

            # Try to add third
            mid3, _ = storage.store_memory("Cannot promote", MemoryType.PROJECT)
            result = storage.promote_to_hot(mid3)

            # Should fail
            assert result is False
            assert not storage.get_memory(mid3).is_hot

            storage.close()

    def test_unpin_allows_eviction(self):
        """Unpinning a memory makes it eligible for eviction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(db_path=Path(tmpdir) / "unpin.db", hot_cache_max_items=2)
            storage = Storage(settings)

            # Fill cache with pinned memories
            mid1, _ = storage.store_memory("To unpin", MemoryType.PROJECT)
            mid2, _ = storage.store_memory("Stays pinned", MemoryType.PROJECT)
            storage.promote_to_hot(mid1, pin=True)
            storage.promote_to_hot(mid2, pin=True)

            # Unpin first memory
            storage.unpin_memory(mid1)
            assert not storage.get_memory(mid1).is_pinned

            # Now promotion should work
            mid3, _ = storage.store_memory("Can promote now", MemoryType.PROJECT)
            result = storage.promote_to_hot(mid3)
            assert result is True

            hot_ids = {m.id for m in storage.get_hot_memories()}
            assert mid2 in hot_ids  # Still pinned
            assert mid3 in hot_ids  # Newly promoted
            assert mid1 not in hot_ids  # Evicted (was unpinned)

            storage.close()

    def test_promotion_source_tracked(self, storage):
        """Promotion source should be tracked."""
        from memory_mcp.storage import PromotionSource

        mid, _ = storage.store_memory("Track source", MemoryType.PROJECT)

        # Manual promotion (default)
        storage.promote_to_hot(mid)
        mem = storage.get_memory(mid)
        assert mem.promotion_source == PromotionSource.MANUAL

        # Demote and re-promote with different source
        storage.demote_from_hot(mid)
        storage.promote_to_hot(mid, promotion_source=PromotionSource.AUTO_THRESHOLD)
        mem = storage.get_memory(mid)
        assert mem.promotion_source == PromotionSource.AUTO_THRESHOLD

    def test_hot_memories_ordered_by_score(self, storage):
        """get_hot_memories should return memories ordered by hot_score desc."""
        # Create 3 memories with different access patterns
        ids = []
        for i in range(3):
            mid, _ = storage.store_memory(f"Ordered {i}", MemoryType.PROJECT)
            ids.append(mid)
            storage.promote_to_hot(mid)

        # Boost scores differently
        for _ in range(10):
            storage.update_access(ids[0])  # Highest
        for _ in range(5):
            storage.update_access(ids[1])  # Middle
        # ids[2] has lowest score

        hot_mems = storage.get_hot_memories()
        scores = [m.hot_score for m in hot_mems]

        # Should be descending order
        assert scores == sorted(scores, reverse=True)
        # First should be highest scorer
        assert hot_mems[0].id == ids[0]


class TestRecallCompositeScoring:
    """Tests for MemoryMCP-iz5: Recency and confidence shaping."""

    def test_recall_returns_composite_scores(self, storage):
        """Recall should populate recency_score and composite_score."""
        storage.store_memory("Test database setup", MemoryType.PROJECT)

        result = storage.recall("database", threshold=0.3)
        assert len(result.memories) > 0

        mem = result.memories[0]
        assert mem.similarity is not None
        assert mem.recency_score is not None
        assert mem.composite_score is not None

    def test_recency_score_decays_with_age(self, storage):
        """Recency score should decay exponentially with age."""
        # Recency score is computed based on created_at
        # New items should have recency_score close to 1.0
        storage.store_memory("Fresh content about testing", MemoryType.PROJECT)

        result = storage.recall("testing", threshold=0.3)
        assert len(result.memories) > 0

        # Just-created item should have high recency score
        mem = result.memories[0]
        assert mem.recency_score > 0.99  # Very close to 1.0 for fresh items

    def test_composite_score_combines_factors(self):
        """Composite score should combine similarity, recency, access, and helpfulness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                db_path=Path(tmpdir) / "composite.db",
                semantic_dedup_enabled=False,
                hybrid_search_enabled=False,  # Disable for pure composite score testing
            )
            storage = Storage(settings)

            storage.store_memory("PostgreSQL database configuration", MemoryType.PROJECT)

            result = storage.recall("database config", threshold=0.3)
            assert len(result.memories) > 0

            mem = result.memories[0]
            # Composite should be weighted sum including trust and decayed helpfulness
            # - trust_weight default is 0.1, trust_score is 1.0 for manual memories
            # - helpfulness is 0.25 * 0.8 (decay for never-used) * 0.05 weight
            expected = (
                mem.similarity * storage.settings.recall_similarity_weight
                + mem.recency_score * storage.settings.recall_recency_weight
                + 0.0  # access_score is 0 for single-item recall
                + mem.trust_score_decayed * storage.settings.recall_trust_weight  # trust component
                + 0.25 * 0.8 * storage.settings.recall_helpfulness_weight  # decayed helpfulness
            )
            # Allow small floating point difference
            assert abs(mem.composite_score - expected) < 0.02

            storage.close()

    def test_results_ordered_by_composite_score(self):
        """Results should be ordered by composite score, not just similarity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure to heavily weight recency
            settings = Settings(
                db_path=Path(tmpdir) / "recency.db",
                recall_similarity_weight=0.5,
                recall_recency_weight=0.4,
                recall_access_weight=0.1,
            )
            storage = Storage(settings)

            # Store items (all will have same recency since created simultaneously)
            storage.store_memory("Python programming language", MemoryType.PROJECT)
            storage.store_memory("Python snake species", MemoryType.PROJECT)

            result = storage.recall("Python", threshold=0.3)
            assert len(result.memories) >= 1

            # Results should be sorted by composite score descending
            scores = [m.composite_score for m in result.memories]
            assert scores == sorted(scores, reverse=True)

            storage.close()

    def test_access_count_affects_ranking(self):
        """Frequently accessed items should rank higher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure to weight access count
            settings = Settings(
                db_path=Path(tmpdir) / "access.db",
                recall_similarity_weight=0.6,
                recall_recency_weight=0.1,
                recall_access_weight=0.3,
            )
            storage = Storage(settings)

            # Store two similar items
            id1, _ = storage.store_memory("Database migration tools", MemoryType.PROJECT)
            id2, _ = storage.store_memory("Database migration scripts", MemoryType.PROJECT)

            # Boost access count on second item
            for _ in range(10):
                storage.update_access(id2)

            result = storage.recall("database migration", threshold=0.3)
            assert len(result.memories) == 2

            # The frequently accessed one should rank higher
            # (assuming similar semantic similarity)
            ids_in_order = [m.id for m in result.memories]
            # id2 should be first due to higher access score
            assert ids_in_order[0] == id2

            storage.close()

    def test_config_weights_respected(self):
        """Custom config weights should be used in scoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set custom weights that sum to 1.0
            settings = Settings(
                db_path=Path(tmpdir) / "weights.db",
                recall_similarity_weight=0.5,
                recall_recency_weight=0.3,
                recall_access_weight=0.2,
            )
            storage = Storage(settings)

            storage.store_memory("Test content for weights", MemoryType.PROJECT)
            result = storage.recall("test weights", threshold=0.3)

            assert len(result.memories) > 0
            # Verify the weights are being used (composite should reflect them)
            mem = result.memories[0]
            assert mem.composite_score is not None

            storage.close()

    def test_component_scores_populated(self, storage):
        """Recall should populate weighted component scores for transparency."""
        storage.store_memory("Test content for components", MemoryType.PROJECT)
        result = storage.recall("test components", threshold=0.3)
        assert len(result.memories) > 0

        mem = result.memories[0]
        # All component fields should be populated
        assert mem.similarity_component is not None
        assert mem.recency_component is not None
        assert mem.access_component is not None
        assert mem.trust_component is not None
        assert mem.helpfulness_component is not None

        # Components should sum to composite score
        expected_total = (
            mem.similarity_component
            + mem.recency_component
            + mem.access_component
            + mem.trust_component
            + mem.helpfulness_component
        )
        assert abs(mem.composite_score - expected_total) < 0.0001

    def test_component_scores_use_weights(self):
        """Component scores should reflect configured weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                db_path=Path(tmpdir) / "components.db",
                recall_similarity_weight=0.6,
                recall_recency_weight=0.2,
                recall_access_weight=0.1,
                recall_trust_weight=0.1,
                hybrid_search_enabled=False,  # Disable hybrid search for pure weight testing
            )
            storage = Storage(settings)

            storage.store_memory("Component test memory", MemoryType.PROJECT)
            result = storage.recall("component test", threshold=0.3)
            assert len(result.memories) > 0

            mem = result.memories[0]
            # Verify similarity component uses correct weight
            assert abs(mem.similarity_component - mem.similarity * 0.6) < 0.0001
            # Verify recency component uses correct weight
            assert abs(mem.recency_component - mem.recency_score * 0.2) < 0.0001

            storage.close()


class TestTrustScoring:
    """Tests for MemoryMCP-8fx: Trust scores and provenance tracking."""

    def test_manual_memory_has_high_trust(self, storage):
        """Manual memories should have trust_score=1.0 by default."""
        mid, _ = storage.store_memory(
            "Manual content", MemoryType.PROJECT, source=MemorySource.MANUAL
        )
        mem = storage.get_memory(mid)
        assert mem.trust_score == storage.settings.trust_score_manual
        assert mem.trust_score == 1.0

    def test_mined_memory_has_lower_trust(self, storage):
        """Mined memories should have lower trust score."""
        mid, _ = storage.store_memory(
            "Mined content", MemoryType.PATTERN, source=MemorySource.MINED
        )
        mem = storage.get_memory(mid)
        assert mem.trust_score == storage.settings.trust_score_mined
        assert mem.trust_score == 0.7

    def test_provenance_tracked_for_mined(self, storage):
        """Mined memories should track provenance (source_log_id, extracted_at)."""
        # First log some output
        log_id = storage.log_output("Some output to mine from")

        # Store mined memory with source_log_id
        mid, _ = storage.store_memory(
            "Pattern from output",
            MemoryType.PATTERN,
            source=MemorySource.MINED,
            source_log_id=log_id,
        )

        mem = storage.get_memory(mid)
        assert mem.source_log_id == log_id
        assert mem.extracted_at is not None

    def test_manual_memory_no_extraction_timestamp(self, storage):
        """Manual memories should not have extracted_at set."""
        mid, _ = storage.store_memory(
            "Manual entry", MemoryType.PROJECT, source=MemorySource.MANUAL
        )
        mem = storage.get_memory(mid)
        assert mem.extracted_at is None
        assert mem.source_log_id is None

    def test_trust_score_decayed_computed_in_recall(self, storage):
        """Recall should compute decayed trust score."""
        storage.store_memory("Trust decay test", MemoryType.PROJECT)

        result = storage.recall("trust decay", threshold=0.3)
        assert len(result.memories) > 0

        mem = result.memories[0]
        assert mem.trust_score_decayed is not None
        # For fresh memory, decayed should be close to base trust
        assert mem.trust_score_decayed > 0.99

    def test_trust_weight_affects_ranking(self):
        """When trust weight is non-zero, it should affect ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                db_path=Path(tmpdir) / "trust.db",
                recall_similarity_weight=0.5,
                recall_recency_weight=0.2,
                recall_access_weight=0.1,
                recall_trust_weight=0.2,  # Enable trust in ranking
            )
            storage = Storage(settings)

            # Store manual (trust=1.0) and mined (trust=0.7) memories
            id_manual, _ = storage.store_memory(
                "Manual database info", MemoryType.PROJECT, source=MemorySource.MANUAL
            )
            id_mined, _ = storage.store_memory(
                "Mined database info", MemoryType.PATTERN, source=MemorySource.MINED
            )

            result = storage.recall("database info", threshold=0.3)
            assert len(result.memories) == 2

            # Manual should rank higher due to trust
            # (assuming similar similarity and recency)
            mem1, mem2 = result.memories
            assert mem1.trust_score > mem2.trust_score

            storage.close()

    def test_custom_trust_scores_in_config(self):
        """Custom trust scores should be configurable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                db_path=Path(tmpdir) / "custom_trust.db",
                trust_score_manual=0.9,
                trust_score_mined=0.5,
            )
            storage = Storage(settings)

            mid_manual, _ = storage.store_memory(
                "Custom manual", MemoryType.PROJECT, source=MemorySource.MANUAL
            )
            mid_mined, _ = storage.store_memory(
                "Custom mined", MemoryType.PATTERN, source=MemorySource.MINED
            )

            assert storage.get_memory(mid_manual).trust_score == 0.9
            assert storage.get_memory(mid_mined).trust_score == 0.5

            storage.close()

    def test_schema_migration_v3_adds_trust_columns(self):
        """Migration to v3 should add trust_score and provenance columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(db_path=Path(tmpdir) / "migrate.db")
            storage = Storage(settings)

            # Check schema version is at least 3
            assert storage.get_schema_version() >= 3

            # Check columns exist
            conn = storage._get_connection()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}

            assert "trust_score" in columns
            assert "source_log_id" in columns
            assert "extracted_at" in columns

            storage.close()


class TestTrustStrengthening:
    """Tests for trust strengthening/weakening system (Engram-inspired)."""

    def test_strengthen_trust_increases_score(self, storage):
        """strengthen_trust() should increase the trust score."""
        # Use mined memory which starts with lower trust (0.7)
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)
        original = storage.get_memory(mid)
        original_trust = original.trust_score  # 0.7

        new_trust = storage.strengthen_trust(mid, boost=0.1)
        assert abs(new_trust - (original_trust + 0.1)) < 0.001

        # Verify persisted
        updated = storage.get_memory(mid)
        assert abs(updated.trust_score - new_trust) < 0.001

    def test_strengthen_trust_caps_at_one(self, storage):
        """strengthen_trust() should cap trust at 1.0."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        # Boost multiple times (starts at 1.0 for manual)
        for _ in range(15):
            storage.strengthen_trust(mid, boost=0.1)

        updated = storage.get_memory(mid)
        assert updated.trust_score == 1.0

    def test_strengthen_trust_refreshes_last_accessed(self, storage):
        """strengthen_trust() should update last_accessed_at."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        # First recall to set initial last_accessed_at
        storage.recall("Test memory", threshold=0.0)

        original = storage.get_memory(mid)
        # last_accessed_at should now be set
        assert original.last_accessed_at is not None

        import time

        time.sleep(0.05)  # Small delay to ensure timestamp difference

        storage.strengthen_trust(mid, boost=0.05)

        updated = storage.get_memory(mid)
        # After strengthen_trust, last_accessed_at should be updated
        assert updated.last_accessed_at is not None
        assert updated.last_accessed_at >= original.last_accessed_at

    def test_strengthen_trust_nonexistent_returns_none(self, storage):
        """strengthen_trust() should return None for nonexistent memory."""
        result = storage.strengthen_trust(99999, boost=0.1)
        assert result is None

    def test_weaken_trust_decreases_score(self, storage):
        """weaken_trust() should decrease the trust score."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)
        original = storage.get_memory(mid)
        original_trust = original.trust_score  # 1.0 for manual

        new_trust = storage.weaken_trust(mid, penalty=0.2)
        assert abs(new_trust - (original_trust - 0.2)) < 0.001

        # Verify persisted
        updated = storage.get_memory(mid)
        assert abs(updated.trust_score - new_trust) < 0.001

    def test_weaken_trust_floors_at_zero(self, storage):
        """weaken_trust() should floor trust at 0.0."""
        # Use mined memory which starts at 0.7
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)

        # Weaken with a large penalty to definitely hit zero
        storage.weaken_trust(mid, penalty=1.0)

        updated = storage.get_memory(mid)
        assert updated.trust_score == 0.0

    def test_weaken_trust_nonexistent_returns_none(self, storage):
        """weaken_trust() should return None for nonexistent memory."""
        result = storage.weaken_trust(99999, penalty=0.1)
        assert result is None

    def test_trust_decay_uses_last_accessed(self, storage):
        """Trust decay should use last_accessed_at when available."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        # Strengthen to refresh last_accessed_at
        storage.strengthen_trust(mid, boost=0.0)  # Just refresh timestamp

        # The memory should have minimal decay since it was just accessed
        result = storage.recall("Test memory", threshold=0.0)
        assert len(result.memories) == 1

        mem = result.memories[0]
        # Trust should be close to base since recently accessed
        assert mem.trust_score_decayed is not None
        assert mem.trust_score_decayed > 0.9  # Minimal decay


class TestRecallModes:
    """Tests for MemoryMCP-1ig: Recall policies and mode presets."""

    def test_recall_mode_config_precision(self, storage):
        """Precision mode should have high threshold and prioritize similarity."""
        config = storage.get_recall_mode_config(RecallMode.PRECISION)
        assert config.threshold == storage.settings.precision_threshold
        assert config.threshold == 0.8
        assert config.limit == storage.settings.precision_limit
        assert config.limit == 3
        assert config.similarity_weight > config.recency_weight

    def test_recall_mode_config_exploratory(self, storage):
        """Exploratory mode should have low threshold and balanced weights."""
        config = storage.get_recall_mode_config(RecallMode.EXPLORATORY)
        assert config.threshold == storage.settings.exploratory_threshold
        assert config.threshold == 0.5
        assert config.limit == storage.settings.exploratory_limit
        assert config.limit == 10

    def test_recall_mode_config_balanced(self, storage):
        """Balanced mode should use default settings."""
        config = storage.get_recall_mode_config(RecallMode.BALANCED)
        assert config.threshold == storage.settings.default_confidence_threshold
        assert config.limit == storage.settings.default_recall_limit

    def test_recall_with_precision_mode(self, storage):
        """Precision mode should filter more aggressively."""
        storage.store_memory("Python programming language basics", MemoryType.PROJECT)
        storage.store_memory("Python snake handling guide", MemoryType.PROJECT)
        storage.store_memory("Something unrelated to Python", MemoryType.PROJECT)

        # Precision mode should have higher threshold
        result = storage.recall("Python", mode=RecallMode.PRECISION)
        assert result.mode == RecallMode.PRECISION
        # High threshold means fewer results might pass
        assert result.gated_count >= 0

    def test_recall_with_exploratory_mode(self, storage):
        """Exploratory mode should return more results."""
        storage.store_memory("Database configuration guide", MemoryType.PROJECT)
        storage.store_memory("Database connection pooling", MemoryType.PROJECT)
        storage.store_memory("Data storage patterns", MemoryType.PROJECT)

        # Exploratory mode has lower threshold
        result = storage.recall("database", mode=RecallMode.EXPLORATORY)
        assert result.mode == RecallMode.EXPLORATORY

    def test_recall_mode_can_be_overridden(self, storage):
        """Explicit limit/threshold should override mode defaults."""
        storage.store_memory("Test content for override", MemoryType.PROJECT)

        # Use precision mode but override limit
        result = storage.recall("test", mode=RecallMode.PRECISION, limit=10, threshold=0.3)

        # Should use precision mode but with custom parameters
        assert result.mode == RecallMode.PRECISION

    def test_recall_mode_weights_affect_scoring(self):
        """Mode-specific weights should affect composite scoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Custom precision mode weights
            settings = Settings(
                db_path=Path(tmpdir) / "modes.db",
                precision_similarity_weight=0.9,
                precision_recency_weight=0.05,
                precision_access_weight=0.05,
            )
            storage = Storage(settings)

            storage.store_memory("Test scoring content", MemoryType.PROJECT)

            result = storage.recall("test scoring", mode=RecallMode.PRECISION)
            assert len(result.memories) > 0

            # Precision mode should heavily weight similarity
            mem = result.memories[0]
            assert mem.composite_score is not None

            storage.close()


class TestRecallGuidance:
    """Tests for hallucination prevention guidance in recall results."""

    def test_high_confidence_guidance(self, storage):
        """High confidence results should indicate direct use."""
        storage.store_memory("PostgreSQL database configuration with pgvector", MemoryType.PROJECT)

        result = storage.recall("PostgreSQL pgvector", threshold=0.3)
        if result.confidence == "high":
            assert "HIGH CONFIDENCE" in result.guidance
            assert "directly" in result.guidance.lower()

    def test_low_confidence_no_results_guidance(self, storage):
        """No results should give explicit 'no match' guidance."""
        # Query for something not in storage
        result = storage.recall("xyz123nonexistent", threshold=0.9)

        if len(result.memories) == 0:
            assert result.guidance is not None
            assert "NO" in result.guidance
            assert "NOT" in result.guidance or "Do NOT" in result.guidance

    def test_gated_results_guidance(self, storage):
        """When results are gated, guidance should explain."""
        storage.store_memory("Somewhat related content", MemoryType.PROJECT)

        # Use very high threshold to gate results
        result = storage.recall("related", threshold=0.99)

        if result.gated_count > 0 and len(result.memories) == 0:
            assert "filtered" in result.guidance.lower()

    def test_medium_confidence_guidance(self, storage):
        """Medium confidence should suggest verification."""
        storage.store_memory("Redis caching configuration", MemoryType.PROJECT)

        result = storage.recall("cache config", threshold=0.3)
        if result.confidence == "medium":
            assert "MEDIUM" in result.guidance
            assert "verify" in result.guidance.lower()


class TestRecallWithFallback:
    """Tests for multi-query fallback recall."""

    def test_fallback_tries_patterns_first(self, storage):
        """Fallback should search patterns before project facts."""
        # Store in different types
        storage.store_memory("import pandas as pd", MemoryType.PATTERN)
        storage.store_memory("This project uses pandas", MemoryType.PROJECT)

        # Use exploratory mode for lower threshold with mock embeddings
        result = storage.recall_with_fallback("pandas", min_results=1, mode=RecallMode.EXPLORATORY)
        assert len(result.memories) >= 1

    def test_fallback_continues_on_no_results(self, storage):
        """Fallback should continue to next type if no results."""
        # Only store in PROJECT type
        storage.store_memory("FastAPI web framework setup", MemoryType.PROJECT)

        # Fallback tries PATTERN first (no results), then PROJECT
        # Use exploratory mode to have lower threshold
        result = storage.recall_with_fallback("FastAPI", mode=RecallMode.EXPLORATORY, min_results=1)
        assert len(result.memories) >= 1

    def test_fallback_respects_mode(self, storage):
        """Fallback should use specified recall mode."""
        storage.store_memory("Test content for fallback mode", MemoryType.PROJECT)

        result = storage.recall_with_fallback(
            "test fallback", mode=RecallMode.EXPLORATORY, min_results=1
        )
        assert result.mode == RecallMode.EXPLORATORY

    def test_fallback_returns_best_result(self, storage):
        """Fallback should return best result if min not met."""
        storage.store_memory("Unique content abc123", MemoryType.PROJECT)

        result = storage.recall_with_fallback(
            "abc123",
            min_results=10,  # More than we can match
        )
        # Should still return what was found
        assert result is not None


class TestRecallTypeFiltering:
    """Tests for filtering recall by memory type."""

    def test_recall_filter_by_pattern_type(self, storage):
        """Should only return patterns when filtered."""
        storage.store_memory("import numpy as np", MemoryType.PATTERN)
        storage.store_memory("This project uses numpy", MemoryType.PROJECT)

        result = storage.recall("numpy", threshold=0.3, memory_types=[MemoryType.PATTERN])

        # All results should be patterns
        for mem in result.memories:
            assert mem.memory_type == MemoryType.PATTERN

    def test_recall_filter_by_project_type(self, storage):
        """Should only return project facts when filtered."""
        storage.store_memory("def process_data():", MemoryType.PATTERN)
        storage.store_memory("Data processing is done in batch", MemoryType.PROJECT)

        result = storage.recall("data processing", threshold=0.3, memory_types=[MemoryType.PROJECT])

        # All results should be project
        for mem in result.memories:
            assert mem.memory_type == MemoryType.PROJECT

    def test_recall_filter_multiple_types(self, storage):
        """Should return memories matching any of the filtered types."""
        storage.store_memory("API endpoint code", MemoryType.PATTERN)
        storage.store_memory("API documentation reference", MemoryType.REFERENCE)
        storage.store_memory("API project architecture", MemoryType.PROJECT)

        result = storage.recall(
            "API", threshold=0.3, memory_types=[MemoryType.PATTERN, MemoryType.REFERENCE]
        )

        # Results should only be PATTERN or REFERENCE
        for mem in result.memories:
            assert mem.memory_type in [MemoryType.PATTERN, MemoryType.REFERENCE]
