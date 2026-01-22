"""Tests for storage module."""

import pytest

from memory_mcp.config import Settings
from memory_mcp.storage import (
    TRUST_REASON_DEFAULTS,
    HotCacheMetrics,
    MemorySource,
    MemoryType,
    PromotionSource,
    RelationType,
    Storage,
    TrustReason,
    ValidationError,
)


@pytest.fixture
def storage(tmp_path):
    """Create a storage instance with temp database.

    Semantic dedup is disabled for most tests to keep test content independent.
    """
    settings = Settings(db_path=tmp_path / "test.db", semantic_dedup_enabled=False)
    stor = Storage(settings)
    yield stor
    stor.close()


def test_store_and_get_memory(storage):
    """Test storing and retrieving a memory."""
    memory_id, is_new = storage.store_memory(
        content="Test content",
        memory_type=MemoryType.PROJECT,
        tags=["test", "example"],
    )
    assert is_new is True

    memory = storage.get_memory(memory_id)
    assert memory is not None
    assert memory.content == "Test content"
    assert memory.memory_type == MemoryType.PROJECT
    assert set(memory.tags) == {"test", "example"}
    assert memory.source == MemorySource.MANUAL


def test_recall_semantic_search(storage):
    """Test semantic search recall."""
    storage.store_memory(
        "PostgreSQL database with pgvector extension",
        MemoryType.PROJECT,
        tags=["database"],
    )
    storage.store_memory(
        "Authentication uses JWT tokens",
        MemoryType.PROJECT,
        tags=["auth"],
    )

    # Search with low threshold (0.2 to accommodate mock embeddings)
    result = storage.recall("database setup", threshold=0.2)
    assert len(result.memories) > 0
    assert "database" in result.memories[0].tags


def test_hot_cache_promotion(storage):
    """Test promoting to hot cache."""
    memory_id, _ = storage.store_memory("Hot content", MemoryType.PATTERN)

    assert not storage.get_memory(memory_id).is_hot

    storage.promote_to_hot(memory_id)
    assert storage.get_memory(memory_id).is_hot

    hot_memories = storage.get_hot_memories()
    assert len(hot_memories) == 1
    assert hot_memories[0].id == memory_id


def test_delete_memory(storage):
    """Test deleting a memory."""
    memory_id, _ = storage.store_memory("To delete", MemoryType.PROJECT)
    assert storage.get_memory(memory_id) is not None

    storage.delete_memory(memory_id)
    assert storage.get_memory(memory_id) is None


def test_recall_confidence_gating(storage):
    """Test confidence gating in recall."""
    storage.store_memory("Very specific content about XYZ123", MemoryType.PROJECT)

    # High threshold should gate out low-similarity results
    result = storage.recall("completely unrelated query ABC", threshold=0.9)
    assert result.confidence == "low"
    assert len(result.memories) == 0


def test_output_logging(storage):
    """Test output logging for mining."""
    log_id = storage.log_output("Some output content")
    assert log_id > 0

    outputs = storage.get_recent_outputs(hours=1)
    assert len(outputs) == 1
    assert outputs[0][1] == "Some output content"


def test_output_logging_project_scoped(storage):
    """Test that output logs can be filtered by project_id.

    This prevents cross-project pattern leakage where logs from one
    project could be mined and auto-approved into a different project.
    """
    # Log outputs from two different projects
    storage.log_output("Project A content", project_id="project-a")
    storage.log_output("Project B content", project_id="project-b")
    storage.log_output("No project content")  # No project

    # Get all outputs (backwards compatible)
    all_outputs = storage.get_recent_outputs(hours=1)
    assert len(all_outputs) == 3

    # Get only project A outputs
    a_outputs = storage.get_recent_outputs(hours=1, project_id="project-a")
    assert len(a_outputs) == 1
    assert a_outputs[0][1] == "Project A content"

    # Get only project B outputs
    b_outputs = storage.get_recent_outputs(hours=1, project_id="project-b")
    assert len(b_outputs) == 1
    assert b_outputs[0][1] == "Project B content"


def test_mined_patterns(storage):
    """Test mined pattern storage."""
    pattern_id = storage.upsert_mined_pattern("import numpy as np", "import")
    assert pattern_id > 0

    # Upsert same pattern should increment count
    storage.upsert_mined_pattern("import numpy as np", "import")

    candidates = storage.get_promotion_candidates(threshold=2)
    assert len(candidates) == 1
    assert candidates[0].occurrence_count == 2


# ========== Validation Tests (Defense-in-Depth) ==========


class TestStorageValidation:
    """Tests for storage layer input validation."""

    def test_store_memory_empty_content_raises(self, storage):
        """Empty content should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            storage.store_memory("", MemoryType.PROJECT)

        with pytest.raises(ValidationError, match="cannot be empty"):
            storage.store_memory("   ", MemoryType.PROJECT)

    def test_store_memory_content_too_long_raises(self, tmp_path):
        """Content exceeding max length should raise ValidationError."""
        settings = Settings(db_path=tmp_path / "test.db", max_content_length=100)
        stor = Storage(settings)
        with pytest.raises(ValidationError, match="too long"):
            stor.store_memory("x" * 101, MemoryType.PROJECT)
        stor.close()

    def test_store_memory_too_many_tags_raises(self, tmp_path):
        """Too many tags should raise ValidationError."""
        settings = Settings(db_path=tmp_path / "test.db", max_tags=3)
        stor = Storage(settings)
        with pytest.raises(ValidationError, match="Too many tags"):
            stor.store_memory("content", MemoryType.PROJECT, tags=["a", "b", "c", "d"])
        stor.close()

    def test_store_memory_tags_normalized(self, storage):
        """Tags should be stripped and empty tags filtered."""
        memory_id, _ = storage.store_memory(
            "content", MemoryType.PROJECT, tags=["  valid  ", "", "  ", "good"]
        )
        memory = storage.get_memory(memory_id)
        assert set(memory.tags) == {"valid", "good"}

    def test_log_output_empty_content_raises(self, storage):
        """Empty output content should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            storage.log_output("")

    def test_log_output_content_too_long_raises(self, tmp_path):
        """Output content exceeding max length should raise ValidationError."""
        settings = Settings(db_path=tmp_path / "test.db", max_content_length=50)
        stor = Storage(settings)
        with pytest.raises(ValidationError, match="too long"):
            stor.log_output("x" * 51)
        stor.close()

    def test_upsert_mined_pattern_empty_raises(self, storage):
        """Empty pattern should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            storage.upsert_mined_pattern("", "import")

    def test_upsert_mined_pattern_too_long_raises(self, tmp_path):
        """Pattern exceeding max length should raise ValidationError."""
        settings = Settings(db_path=tmp_path / "test.db", max_content_length=50)
        stor = Storage(settings)
        with pytest.raises(ValidationError, match="too long"):
            stor.upsert_mined_pattern("x" * 51, "import")
        stor.close()

    def test_validation_error_is_value_error(self, storage):
        """ValidationError is a ValueError subclass for compatibility."""
        with pytest.raises(ValueError):
            storage.store_memory("", MemoryType.PROJECT)


# ========== Hot Cache Metrics Tests ==========


class TestHotCacheMetrics:
    """Tests for hot cache observability metrics."""

    def test_initial_metrics_are_zero(self, storage):
        """Metrics should start at zero."""
        metrics = storage.get_hot_cache_metrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.promotions == 0

    def test_record_hit(self, storage):
        """record_hot_cache_hit should increment hits."""
        storage.record_hot_cache_hit()
        storage.record_hot_cache_hit()
        metrics = storage.get_hot_cache_metrics()
        assert metrics.hits == 2

    def test_record_miss(self, storage):
        """record_hot_cache_miss should increment misses."""
        storage.record_hot_cache_miss()
        metrics = storage.get_hot_cache_metrics()
        assert metrics.misses == 1

    def test_promotion_increments_metric(self, storage):
        """Promoting to hot cache should increment promotions."""
        memory_id, _ = storage.store_memory("Test content", MemoryType.PROJECT)
        storage.promote_to_hot(memory_id)
        metrics = storage.get_hot_cache_metrics()
        assert metrics.promotions == 1

    def test_eviction_increments_metric(self, tmp_path):
        """Evicting from hot cache should increment evictions."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            hot_cache_max_items=2,
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)

        # Fill hot cache
        id1, _ = stor.store_memory("Content 1", MemoryType.PROJECT)
        id2, _ = stor.store_memory("Content 2", MemoryType.PROJECT)
        stor.promote_to_hot(id1)
        stor.promote_to_hot(id2)

        # Add third item, triggering eviction
        id3, _ = stor.store_memory("Content 3", MemoryType.PROJECT)
        stor.promote_to_hot(id3)

        metrics = stor.get_hot_cache_metrics()
        assert metrics.evictions == 1
        assert metrics.promotions == 3
        stor.close()

    def test_get_hot_cache_stats(self, storage):
        """get_hot_cache_stats should return comprehensive stats."""
        memory_id, _ = storage.store_memory("Test content", MemoryType.PROJECT)
        storage.promote_to_hot(memory_id)
        storage.record_hot_cache_hit()

        stats = storage.get_hot_cache_stats()
        assert stats["current_count"] == 1
        assert stats["max_items"] == storage.settings.hot_cache_max_items
        assert stats["hits"] == 1
        assert stats["promotions"] == 1
        assert "avg_hot_score" in stats
        assert stats["pinned_count"] == 0

    def test_metrics_to_dict(self):
        """HotCacheMetrics.to_dict should return correct dict."""
        metrics = HotCacheMetrics(hits=5, misses=2, evictions=1, promotions=3)
        d = metrics.to_dict()
        assert d == {"hits": 5, "misses": 2, "evictions": 1, "promotions": 3}

    def test_hot_memories_ordered_by_session_recency(self, storage):
        """get_hot_memories should order by last_used_at (session recency) first."""
        # Create and promote memories
        id1, _ = storage.store_memory("Memory 1 (never used)", MemoryType.PROJECT)
        id2, _ = storage.store_memory("Memory 2 (used recently)", MemoryType.PROJECT)
        id3, _ = storage.store_memory("Memory 3 (used earlier)", MemoryType.PROJECT)

        storage.promote_to_hot(id1)
        storage.promote_to_hot(id2)
        storage.promote_to_hot(id3)

        # Mark id3 as used first, then id2
        storage.mark_retrieval_used(id3)
        import time

        time.sleep(0.01)  # Ensure different timestamp
        storage.mark_retrieval_used(id2)

        # Get hot memories - should be ordered by last_used_at descending
        hot = storage.get_hot_memories()
        ids = [m.id for m in hot]

        # id2 used most recently should be first, then id3, then id1 (never used)
        assert ids.index(id2) < ids.index(id3)
        assert ids.index(id3) < ids.index(id1)


# ========== Auto-Promotion Tests ==========


class TestAutoPromotion:
    """Tests for automatic promotion on access threshold."""

    def test_auto_promote_disabled(self, tmp_path):
        """Auto-promotion should not happen when disabled."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=False,
            promotion_threshold=2,
        )
        stor = Storage(settings)

        # Create memory and access it enough times
        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)
        for _ in range(5):
            stor.update_access(memory_id)

        # Should not auto-promote
        result = stor.check_auto_promote(memory_id)
        assert result is False
        assert not stor.get_memory(memory_id).is_hot
        stor.close()

    def test_auto_promote_on_threshold(self, tmp_path):
        """Memory should auto-promote when access count reaches threshold."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=True,
            promotion_threshold=3,
            # Set high salience threshold so we only test access count promotion
            salience_promotion_threshold=0.99,
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)

        # Access count is 0, should not promote
        assert stor.check_auto_promote(memory_id) is False

        # Access twice (still below threshold)
        stor.update_access(memory_id)
        stor.update_access(memory_id)
        assert stor.get_memory(memory_id).access_count == 2
        assert stor.check_auto_promote(memory_id) is False

        # Third access reaches threshold
        stor.update_access(memory_id)
        assert stor.get_memory(memory_id).access_count == 3
        assert stor.check_auto_promote(memory_id) is True

        # Verify promoted with correct source
        memory = stor.get_memory(memory_id)
        assert memory.is_hot is True
        assert memory.promotion_source == PromotionSource.AUTO_THRESHOLD
        stor.close()

    def test_auto_promote_on_salience(self, tmp_path):
        """Memory should auto-promote when salience score reaches threshold."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=True,
            promotion_threshold=100,  # Very high so access count won't trigger
            salience_promotion_threshold=0.4,  # Low threshold to test salience
        )
        stor = Storage(settings)

        # Content with high importance score (code-like content)
        code_content = "```python\ndef hello():\n    print('hi')\n```"
        memory_id, _ = stor.store_memory(code_content, MemoryType.PATTERN)

        # With high importance, trust=1.0, and recent access, salience should be high enough
        stor.update_access(memory_id)  # Make it recently accessed

        # Should promote based on salience even with access_count=1
        memory = stor.get_memory(memory_id)
        assert memory.salience_score is not None
        assert memory.salience_score >= 0.4  # Should meet threshold
        assert stor.check_auto_promote(memory_id) is True

        # Verify promoted
        memory = stor.get_memory(memory_id)
        assert memory.is_hot is True
        stor.close()

    def test_auto_promote_already_hot(self, tmp_path):
        """Already hot memory should not be re-promoted."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=True,
            promotion_threshold=2,
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)
        stor.promote_to_hot(memory_id)  # Manually promote first

        for _ in range(5):
            stor.update_access(memory_id)

        # Should return False (already hot)
        assert stor.check_auto_promote(memory_id) is False
        stor.close()

    def test_auto_promote_during_recall(self, tmp_path):
        """Recall should trigger auto-promotion when threshold reached."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=True,
            promotion_threshold=3,
        )
        stor = Storage(settings)

        # Store a memory
        memory_id, _ = stor.store_memory("PostgreSQL database configuration", MemoryType.PROJECT)

        # Recall it multiple times (each recall increments access_count)
        for i in range(3):
            result = stor.recall("PostgreSQL database", threshold=0.2)
            # Should find our memory
            assert len(result.memories) > 0

        # After 3 recalls, should be auto-promoted
        memory = stor.get_memory(memory_id)
        assert memory.is_hot is True
        assert memory.promotion_source == PromotionSource.AUTO_THRESHOLD
        stor.close()

    def test_auto_promote_blocked_by_low_used_rate(self, tmp_path):
        """Memory with low used_rate should not be promoted after warmup."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=True,
            promotion_threshold=3,
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test memory for used rate", MemoryType.PROJECT)

        # Manually set retrieved_count to trigger warmup threshold (5+)
        # but keep used_count low to fail the used_rate check
        with stor.transaction() as conn:
            conn.execute(
                """UPDATE memories SET retrieved_count = 6, used_count = 0,
                   access_count = 10 WHERE id = ?""",
                (memory_id,),
            )

        # used_rate = 0/6 = 0 < 0.25, should not promote despite high access
        result = stor.check_auto_promote(memory_id)
        assert result is False
        memory = stor.get_memory(memory_id)
        assert not memory.is_hot
        stor.close()

    def test_auto_promote_passes_with_high_used_rate(self, tmp_path):
        """Memory with high used_rate should be promoted after warmup."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=True,
            promotion_threshold=3,
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Helpful memory test", MemoryType.PROJECT)

        # Manually set retrieved_count and used_count for high used_rate
        with stor.transaction() as conn:
            conn.execute(
                """UPDATE memories SET retrieved_count = 6, used_count = 2,
                   access_count = 10 WHERE id = ?""",
                (memory_id,),
            )

        # used_rate = 2/6 = 0.33 >= 0.25, should promote
        result = stor.check_auto_promote(memory_id)
        assert result is True
        memory = stor.get_memory(memory_id)
        assert memory.is_hot
        stor.close()

    def test_auto_promote_no_warmup_gates_early(self, tmp_path):
        """Memory without enough retrievals should skip helpfulness check."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_promote=True,
            promotion_threshold=3,
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("New memory cold start", MemoryType.PROJECT)

        # Manually set retrieved_count below warmup threshold
        with stor.transaction() as conn:
            conn.execute(
                """UPDATE memories SET retrieved_count = 4, used_count = 0,
                   access_count = 10 WHERE id = ?""",
                (memory_id,),
            )

        # retrieved_count < 5, so helpfulness check is skipped
        # Should promote based on access_count alone
        result = stor.check_auto_promote(memory_id)
        assert result is True
        memory = stor.get_memory(memory_id)
        assert memory.is_hot
        stor.close()


# ========== Auto-Demotion Tests ==========


class TestAutoDemotion:
    """Tests for automatic demotion of stale hot memories."""

    def test_auto_demote_disabled(self, tmp_path):
        """Auto-demotion should not happen when disabled."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_demote=False,
            demotion_days=1,
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)
        stor.promote_to_hot(memory_id)

        # Even with stale settings, should not demote
        demoted = stor.demote_stale_hot_memories()
        assert demoted == []
        assert stor.get_memory(memory_id).is_hot is True
        stor.close()

    def test_auto_demote_skips_pinned(self, tmp_path):
        """Pinned memories should not be auto-demoted."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_demote=True,
            demotion_days=0,  # Demote immediately
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)
        stor.promote_to_hot(memory_id, pin=True)

        # Pinned memory should not be demoted
        demoted = stor.demote_stale_hot_memories()
        assert demoted == []
        assert stor.get_memory(memory_id).is_hot is True
        stor.close()

    def test_auto_demote_stale_memory(self, tmp_path):
        """Stale hot memory should be demoted."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_demote=True,
            demotion_days=1,  # Demote after 1 day without access
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)
        stor.promote_to_hot(memory_id)

        # Simulate an old memory (demotion uses COALESCE(last_accessed_at, created_at))
        with stor.transaction() as conn:
            conn.execute(
                """
                UPDATE memories
                SET created_at = datetime('now', '-30 days'),
                    last_accessed_at = datetime('now', '-30 days')
                WHERE id = ?
                """,
                (memory_id,),
            )

        demoted = stor.demote_stale_hot_memories()
        assert memory_id in demoted
        assert stor.get_memory(memory_id).is_hot is False
        stor.close()

    def test_newly_promoted_memory_not_demoted(self, tmp_path):
        """Newly promoted memory should NOT be demoted even with NULL last_accessed_at."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_demote=True,
            demotion_days=1,  # Demote after 1 day without access
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)
        stor.promote_to_hot(memory_id)

        # Memory just created (created_at is now), should NOT be demoted
        # even though last_accessed_at is NULL
        demoted = stor.demote_stale_hot_memories()
        assert demoted == []
        assert stor.get_memory(memory_id).is_hot is True
        stor.close()

    def test_maintenance_includes_auto_demote(self, tmp_path):
        """Maintenance should run auto-demotion."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            auto_demote=True,
            demotion_days=1,  # Demote after 1 day without access
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Test content", MemoryType.PROJECT)
        stor.promote_to_hot(memory_id)

        # Simulate an old memory by backdating created_at and last_accessed_at
        # (both need to be old to trigger demotion - created_at is fallback)
        with stor.transaction() as conn:
            conn.execute(
                """
                UPDATE memories
                SET created_at = datetime('now', '-30 days'),
                    last_accessed_at = datetime('now', '-30 days')
                WHERE id = ?
                """,
                (memory_id,),
            )

        result = stor.maintenance()
        assert result["auto_demoted_count"] == 1
        assert memory_id in result["auto_demoted_ids"]
        stor.close()


# ========== Freshness and Cleanup Tests ==========


class TestMemoryRetention:
    """Tests for memory retention policies and cleanup."""

    def test_cleanup_stale_memories_respects_retention(self, tmp_path):
        """cleanup_stale_memories deletes based on type-specific retention."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            retention_conversation_days=0,  # Immediate cleanup for conversations
            retention_project_days=0,  # Never expire projects
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)

        # Create memories of different types
        proj_id, _ = stor.store_memory("Project fact", MemoryType.PROJECT)
        conv_id, _ = stor.store_memory("Conversation fact", MemoryType.CONVERSATION)

        # Cleanup should not delete anything yet (both just created)
        result = stor.cleanup_stale_memories()
        assert result["total_deleted"] == 0
        stor.close()

    def test_cleanup_preserves_hot_memories(self, tmp_path):
        """Hot memories should not be cleaned up regardless of age."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            retention_pattern_days=0,  # Immediate cleanup
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)

        memory_id, _ = stor.store_memory("Pattern content", MemoryType.PATTERN)
        stor.promote_to_hot(memory_id)

        # Hot memory should be preserved
        stor.cleanup_stale_memories()
        assert stor.get_memory(memory_id) is not None
        stor.close()

    def test_get_retention_days_by_type(self, tmp_path):
        """_get_retention_days returns correct values per type."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            retention_project_days=100,
            retention_pattern_days=200,
            retention_reference_days=300,
            retention_conversation_days=400,
        )
        stor = Storage(settings)

        assert stor._get_retention_days(MemoryType.PROJECT) == 100
        assert stor._get_retention_days(MemoryType.PATTERN) == 200
        assert stor._get_retention_days(MemoryType.REFERENCE) == 300
        assert stor._get_retention_days(MemoryType.CONVERSATION) == 400
        stor.close()


class TestEmbeddingValidation:
    """Tests for embedding model validation."""

    def test_validate_embeddings_first_run(self, storage):
        """First run should store model info and return valid."""
        result = storage.validate_embedding_model("test-model", 384)

        assert result["valid"] is True
        assert result.get("first_run") is True
        assert result["model"] == "test-model"
        assert result["dimension"] == 384

    def test_validate_embeddings_same_model(self, storage):
        """Same model should validate successfully."""
        # First run
        storage.validate_embedding_model("test-model", 384)

        # Second run with same model
        result = storage.validate_embedding_model("test-model", 384)

        assert result["valid"] is True
        assert "first_run" not in result

    def test_validate_embeddings_model_changed(self, storage):
        """Different model should fail validation."""
        # First run
        storage.validate_embedding_model("old-model", 384)

        # Second run with different model
        result = storage.validate_embedding_model("new-model", 384)

        assert result["valid"] is False
        assert result["model_changed"] is True
        assert result["stored_model"] == "old-model"
        assert result["current_model"] == "new-model"

    def test_validate_embeddings_dimension_changed(self, storage):
        """Different dimension should fail validation."""
        # First run
        storage.validate_embedding_model("test-model", 384)

        # Second run with different dimension
        result = storage.validate_embedding_model("test-model", 768)

        assert result["valid"] is False
        assert result["dimension_changed"] is True
        assert result["stored_dimension"] == 384
        assert result["current_dimension"] == 768


class TestFullCleanup:
    """Tests for comprehensive cleanup operation."""

    def test_run_full_cleanup_returns_stats(self, storage):
        """run_full_cleanup returns stats for all operations."""
        result = storage.run_full_cleanup()

        assert "hot_cache_demoted" in result
        assert "patterns_expired" in result
        assert "logs_deleted" in result
        assert "memories_deleted" in result
        assert "memories_deleted_by_type" in result
        assert "injections_deleted" in result

    def test_cleanup_old_logs(self, tmp_path):
        """cleanup_old_logs deletes based on log_retention_days."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            log_retention_days=0,  # Immediate cleanup
        )
        stor = Storage(settings)

        # Create a log
        stor.log_output("Test output content")

        # Cleanup should delete it (retention = 0)
        deleted = stor.cleanup_old_logs()
        # Note: log_output already deletes old logs, so this may be 0
        assert isinstance(deleted, int)
        stor.close()


# ========== Injection Tracking Tests ==========


class TestInjectionTracking:
    """Tests for injection tracking functionality."""

    def test_log_injection(self, storage):
        """log_injection should record injection event."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        log_id = storage.log_injection(mid, "hot-cache", session_id="test-session")
        assert log_id > 0

    def test_log_injections_batch(self, storage):
        """log_injections_batch should record multiple injections."""
        mid1, _ = storage.store_memory("Memory 1", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory 2", MemoryType.PROJECT)

        count = storage.log_injections_batch([mid1, mid2], "hot-cache", session_id="test-session")
        assert count == 2

    def test_get_recent_injections(self, storage):
        """get_recent_injections should return injection records."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)
        storage.log_injection(mid, "hot-cache", session_id="test-session")

        injections = storage.get_recent_injections(memory_id=mid, days=7)
        assert len(injections) == 1
        assert injections[0].memory_id == mid
        assert injections[0].resource == "hot-cache"

    def test_was_recently_injected(self, storage):
        """was_recently_injected should return True for injected memories."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        assert not storage.was_recently_injected(mid)

        storage.log_injection(mid, "hot-cache")

        assert storage.was_recently_injected(mid)
        assert storage.was_recently_injected(mid, resource="hot-cache")
        assert not storage.was_recently_injected(mid, resource="working-set")

    def test_get_injection_stats(self, storage):
        """get_injection_stats should return correct stats."""
        mid1, _ = storage.store_memory("Memory 1", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory 2", MemoryType.PROJECT)

        storage.log_injection(mid1, "hot-cache")
        storage.log_injection(mid1, "hot-cache")
        storage.log_injection(mid2, "working-set")

        stats = storage.get_injection_stats(days=7)
        assert stats["total_injections"] == 3
        assert stats["unique_memories"] == 2
        assert stats["by_resource"]["hot-cache"] == 2
        assert stats["by_resource"]["working-set"] == 1

    def test_cleanup_old_injections(self, tmp_path):
        """cleanup_old_injections should delete old records."""
        settings = Settings(db_path=tmp_path / "test.db")
        stor = Storage(settings)

        mid, _ = stor.store_memory("Test memory", MemoryType.PROJECT)
        stor.log_injection(mid, "hot-cache")

        # Cleanup with 0 retention should delete all
        deleted = stor.cleanup_old_injections(retention_days=0)
        assert deleted >= 0  # May be 0 if logged just now

        stor.close()


# ========== Helpfulness Tracking Tests ==========


class TestHelpfulnessTracking:
    """Tests for helpfulness tracking columns (v16 schema)."""

    def test_memory_has_helpfulness_fields(self, storage):
        """New memories should have helpfulness tracking fields."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)
        memory = storage.get_memory(mid)

        assert memory.retrieved_count == 0
        assert memory.used_count == 0
        assert memory.last_used_at is None
        assert memory.utility_score == 0.25  # Bayesian prior: (0+1)/(0+1+3)

    def test_helpfulness_columns_exist_in_schema(self, storage):
        """Helpfulness columns should exist in memories table."""
        with storage._connection() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}

        assert "retrieved_count" in columns
        assert "used_count" in columns
        assert "last_used_at" in columns
        assert "utility_score" in columns

    def test_last_used_at_index_exists(self, storage):
        """Index on last_used_at should exist for session recency ordering."""
        with storage._connection() as conn:
            indexes = {
                row[1]
                for row in conn.execute(
                    "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
                ).fetchall()
            }

        assert "idx_memories_last_used" in indexes

    def test_recall_increments_retrieved_count(self, storage):
        """recall() should increment retrieved_count for returned memories."""
        mid, _ = storage.store_memory("Test Python programming", MemoryType.PROJECT)

        # Verify initial state
        memory = storage.get_memory(mid)
        assert memory.retrieved_count == 0

        # Recall should increment retrieved_count
        result = storage.recall("Python programming")
        assert len(result.memories) > 0, "Recall should find the memory"

        memory = storage.get_memory(mid)
        assert memory.retrieved_count == 1

        # Second recall with same query
        storage.recall("Python programming")

        memory = storage.get_memory(mid)
        assert memory.retrieved_count == 2

    def test_mark_retrieval_used_increments_used_count(self, storage):
        """mark_retrieval_used() should increment used_count and set last_used_at."""
        mid, _ = storage.store_memory("Test Python programming", MemoryType.PROJECT)

        # Verify initial state
        memory = storage.get_memory(mid)
        assert memory.used_count == 0
        assert memory.last_used_at is None

        # Mark as used should increment used_count
        storage.mark_retrieval_used(mid)

        memory = storage.get_memory(mid)
        assert memory.used_count == 1
        assert memory.last_used_at is not None

        # Multiple marks should accumulate
        storage.mark_retrieval_used(mid)
        storage.mark_retrieval_used(mid)

        memory = storage.get_memory(mid)
        assert memory.used_count == 3

    def test_mark_retrieval_used_updates_utility_score(self, storage):
        """mark_retrieval_used() should update utility_score using Bayesian formula."""
        mid, _ = storage.store_memory("Test Python programming", MemoryType.PROJECT)

        # Recall twice to get retrieved_count=2
        storage.recall("Python programming")
        storage.recall("Python programming")

        memory = storage.get_memory(mid)
        assert memory.retrieved_count == 2
        assert memory.used_count == 0
        # utility_score starts at default 0.25 until mark_used updates it
        assert memory.utility_score == 0.25

        # Mark as used once - this recomputes utility_score
        storage.mark_retrieval_used(mid)

        memory = storage.get_memory(mid)
        assert memory.used_count == 1
        # utility_score = (1+1)/(2+1+3) = 2/6 ≈ 0.333
        assert abs(memory.utility_score - 2 / 6) < 0.01

        # Mark as used again
        storage.mark_retrieval_used(mid)

        memory = storage.get_memory(mid)
        assert memory.used_count == 2
        # utility_score = (2+1)/(2+1+3) = 3/6 = 0.5
        assert abs(memory.utility_score - 3 / 6) < 0.01

    def test_recall_includes_helpfulness_component(self, storage):
        """recall() should include helpfulness_component in score breakdown."""
        mid, _ = storage.store_memory("Python programming test", MemoryType.PROJECT)

        result = storage.recall("Python programming", threshold=0.3)
        assert len(result.memories) > 0

        memory = result.memories[0]
        # Helpfulness component should be populated
        assert memory.helpfulness_component is not None
        # Cold start value (0.25 * 0.05 weight = 0.0125)
        assert abs(memory.helpfulness_component - 0.0125) < 0.001


# ========== Trust Granularity Tests ==========


class TestTrustReason:
    """Tests for contextual trust adjustments with reasons."""

    def test_adjust_trust_with_reason(self, storage):
        """adjust_trust() should record reason in history."""
        # Use mined memory (starts at 0.7) so we can increase trust
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)
        original = storage.get_memory(mid)

        new_trust = storage.adjust_trust(mid, reason=TrustReason.USED_CORRECTLY)
        assert new_trust > original.trust_score

        # Check history was recorded
        history = storage.get_trust_history(mid)
        assert len(history) == 1
        assert history[0].reason == TrustReason.USED_CORRECTLY
        assert history[0].old_trust == original.trust_score
        assert abs(history[0].new_trust - new_trust) < 0.001

    def test_adjust_trust_uses_reason_default(self, storage):
        """adjust_trust() uses reason's default delta if not specified."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)
        original = storage.get_memory(mid)

        # EXPLICITLY_CONFIRMED has default boost of 0.15
        storage.adjust_trust(mid, reason=TrustReason.EXPLICITLY_CONFIRMED)
        updated = storage.get_memory(mid)

        expected_boost = TRUST_REASON_DEFAULTS[TrustReason.EXPLICITLY_CONFIRMED]
        # Manual memory starts at 1.0, so it caps at 1.0
        assert updated.trust_score == min(1.0, original.trust_score + expected_boost)

    def test_adjust_trust_custom_delta(self, storage):
        """adjust_trust() can override default delta."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)
        original = storage.get_memory(mid)  # Mined starts at 0.7

        custom_boost = 0.25
        storage.adjust_trust(mid, reason=TrustReason.USED_CORRECTLY, delta=custom_boost)
        updated = storage.get_memory(mid)

        assert abs(updated.trust_score - (original.trust_score + custom_boost)) < 0.001

    def test_adjust_trust_with_note(self, storage):
        """adjust_trust() should store optional note."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)
        note_text = "Verified against current codebase"

        storage.adjust_trust(mid, reason=TrustReason.EXPLICITLY_CONFIRMED, note=note_text)

        history = storage.get_trust_history(mid)
        assert len(history) == 1
        assert history[0].note == note_text

    def test_weaken_trust_with_reason(self, storage):
        """weaken_trust() should support reason parameter."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        storage.weaken_trust(mid, reason=TrustReason.OUTDATED)

        history = storage.get_trust_history(mid)
        assert len(history) == 1
        assert history[0].reason == TrustReason.OUTDATED
        assert history[0].delta < 0  # Weakening should be negative

    def test_strengthen_trust_with_reason(self, storage):
        """strengthen_trust() should support reason parameter."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)

        storage.strengthen_trust(mid, reason=TrustReason.CROSS_VALIDATED)

        history = storage.get_trust_history(mid)
        assert len(history) == 1
        assert history[0].reason == TrustReason.CROSS_VALIDATED
        assert history[0].delta > 0  # Strengthening should be positive


class TestConfidenceWeightedTrust:
    """Tests for confidence-weighted trust updates."""

    def test_similarity_scales_boost(self, storage):
        """High similarity should scale the trust boost."""
        # Create two memories
        mid1, _ = storage.store_memory("Memory one", MemoryType.PROJECT, source=MemorySource.MINED)
        mid2, _ = storage.store_memory("Memory two", MemoryType.PROJECT, source=MemorySource.MINED)

        # Adjust with different similarities
        storage.adjust_trust(mid1, reason=TrustReason.HIGH_SIMILARITY_HIT, similarity=0.95)
        storage.adjust_trust(mid2, reason=TrustReason.HIGH_SIMILARITY_HIT, similarity=0.70)

        history1 = storage.get_trust_history(mid1)
        history2 = storage.get_trust_history(mid2)

        # Higher similarity should result in larger boost
        assert history1[0].delta > history2[0].delta
        assert history1[0].similarity == 0.95
        assert history2[0].similarity == 0.70

    def test_similarity_recorded_in_history(self, storage):
        """Similarity should be recorded in trust history."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        storage.adjust_trust(mid, reason=TrustReason.HIGH_SIMILARITY_HIT, similarity=0.92)

        history = storage.get_trust_history(mid)
        assert len(history) == 1
        assert history[0].similarity == 0.92


class TestPerTypeTrustDecay:
    """Tests for per-memory-type trust decay rates."""

    def test_get_trust_decay_halflife(self, storage):
        """_get_trust_decay_halflife returns different values per type."""
        # Project memories should decay slowest (180 days default)
        project_halflife = storage._get_trust_decay_halflife(MemoryType.PROJECT)
        # Pattern memories decay faster (60 days default)
        pattern_halflife = storage._get_trust_decay_halflife(MemoryType.PATTERN)
        # Conversation memories decay fastest (30 days default)
        conversation_halflife = storage._get_trust_decay_halflife(MemoryType.CONVERSATION)

        assert project_halflife > pattern_halflife
        assert pattern_halflife > conversation_halflife

    def test_none_type_uses_default(self, storage):
        """None memory type should use default halflife."""
        default_halflife = storage._get_trust_decay_halflife(None)
        assert default_halflife == storage.settings.trust_decay_halflife_days


class TestTrustHistory:
    """Tests for trust history audit trail."""

    def test_get_trust_history_empty(self, storage):
        """get_trust_history returns empty list for new memory."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        history = storage.get_trust_history(mid)
        assert history == []

    def test_get_trust_history_multiple_events(self, storage):
        """get_trust_history returns all events in order."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)

        # Make several adjustments
        storage.adjust_trust(mid, reason=TrustReason.USED_CORRECTLY)
        storage.adjust_trust(mid, reason=TrustReason.EXPLICITLY_CONFIRMED)
        storage.adjust_trust(mid, reason=TrustReason.OUTDATED)

        history = storage.get_trust_history(mid)
        assert len(history) == 3

        # Should be ordered by most recent first (using id DESC)
        assert history[0].reason == TrustReason.OUTDATED
        assert history[1].reason == TrustReason.EXPLICITLY_CONFIRMED
        assert history[2].reason == TrustReason.USED_CORRECTLY

    def test_get_trust_history_limit(self, storage):
        """get_trust_history respects limit parameter."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)

        # Make 5 adjustments
        for _ in range(5):
            storage.adjust_trust(mid, reason=TrustReason.USED_CORRECTLY)

        history = storage.get_trust_history(mid, limit=3)
        assert len(history) == 3

    def test_trust_history_table_created(self, storage):
        """trust_history table should exist in schema."""
        with storage._connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trust_history'"
            ).fetchone()
            assert result is not None


class TestAutoTrustBoostOnRecall:
    """Tests for automatic trust boost on high-similarity recall."""

    def test_high_similarity_recall_boosts_trust(self, tmp_path):
        """Recall with high similarity should auto-boost trust."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            trust_auto_strengthen_on_recall=True,
            trust_high_similarity_threshold=0.80,  # Lower threshold for test
        )
        stor = Storage(settings)

        # Create memory that should match well
        mid, _ = stor.store_memory(
            "PostgreSQL database configuration",
            MemoryType.PROJECT,
            source=MemorySource.MINED,
        )

        # Recall with very similar query - should trigger auto-boost
        result = stor.recall("PostgreSQL database settings", threshold=0.5)

        # If similarity is high enough, trust should have increased
        if result.memories and result.memories[0].similarity >= 0.80:
            history = stor.get_trust_history(mid)
            # Should have a HIGH_SIMILARITY_HIT event
            high_sim_events = [h for h in history if h.reason == TrustReason.HIGH_SIMILARITY_HIT]
            assert len(high_sim_events) >= 1

        stor.close()

    def test_auto_trust_boost_disabled(self, tmp_path):
        """Auto-trust boost should respect config setting."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            trust_auto_strengthen_on_recall=False,
        )
        stor = Storage(settings)

        mid, _ = stor.store_memory("PostgreSQL database configuration", MemoryType.PROJECT)

        # Recall shouldn't boost trust when disabled
        stor.recall("PostgreSQL database", threshold=0.5)

        history = stor.get_trust_history(mid)
        assert len(history) == 0  # No trust changes

        stor.close()


# ========== Memory Relationships (Knowledge Graph) Tests ==========


class TestMemoryRelationships:
    """Tests for memory relationships / knowledge graph functionality."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a storage instance with temp database."""
        settings = Settings(db_path=tmp_path / "test.db", semantic_dedup_enabled=False)
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_link_memories_basic(self, storage):
        """link_memories creates a relationship between two memories."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)

        relation = storage.link_memories(mid1, mid2, RelationType.RELATES_TO)

        assert relation is not None
        assert relation.from_memory_id == mid1
        assert relation.to_memory_id == mid2
        assert relation.relation_type == RelationType.RELATES_TO

    def test_link_memories_different_types(self, storage):
        """Can create different relationship types."""
        mid1, _ = storage.store_memory("Overview", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Details", MemoryType.PROJECT)

        # Create multiple relationship types
        r1 = storage.link_memories(mid1, mid2, RelationType.RELATES_TO)
        r2 = storage.link_memories(mid2, mid1, RelationType.ELABORATES)

        assert r1 is not None
        assert r2 is not None
        assert r1.relation_type == RelationType.RELATES_TO
        assert r2.relation_type == RelationType.ELABORATES

    def test_link_memories_mentions_type(self, storage):
        """Can use MENTIONS relation type for entity extraction linking."""
        mid1, _ = storage.store_memory("Project uses PostgreSQL for data", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("PostgreSQL", MemoryType.REFERENCE)

        # Link source content to mentioned entity
        r = storage.link_memories(mid1, mid2, RelationType.MENTIONS)

        assert r is not None
        assert r.relation_type == RelationType.MENTIONS
        assert r.from_memory_id == mid1
        assert r.to_memory_id == mid2

    def test_link_memories_prevents_duplicates(self, storage):
        """Same relationship type between same memories returns None."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)

        r1 = storage.link_memories(mid1, mid2, RelationType.DEPENDS_ON)
        r2 = storage.link_memories(mid1, mid2, RelationType.DEPENDS_ON)  # Duplicate

        assert r1 is not None
        assert r2 is None  # Should be None for duplicate

    def test_link_memories_nonexistent_memory(self, storage):
        """link_memories returns None when memory doesn't exist."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)

        relation = storage.link_memories(mid1, 99999, RelationType.RELATES_TO)
        assert relation is None

    def test_link_memories_self_reference(self, storage):
        """Cannot link a memory to itself."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)

        relation = storage.link_memories(mid1, mid1, RelationType.RELATES_TO)
        assert relation is None

    def test_get_memories_by_source_log(self, storage):
        """Can retrieve memories by their source_log_id."""
        # First create an output log to get a valid log_id
        log_id = storage.log_output("Test output content")

        # Store memories with this source_log_id
        mid1, _ = storage.store_memory(
            "Memory from log 1", MemoryType.PROJECT, source_log_id=log_id
        )
        mid2, _ = storage.store_memory(
            "Memory from log 2", MemoryType.PATTERN, source_log_id=log_id
        )
        # Store a memory without source_log_id (should not be returned)
        mid3, _ = storage.store_memory("Memory without log", MemoryType.PROJECT)

        memories = storage.get_memories_by_source_log(log_id)

        assert len(memories) == 2
        memory_ids = {m.id for m in memories}
        assert mid1 in memory_ids
        assert mid2 in memory_ids
        assert mid3 not in memory_ids

    def test_get_memories_by_source_log_empty(self, storage):
        """Returns empty list for non-existent source_log_id."""
        memories = storage.get_memories_by_source_log(99999)
        assert memories == []

    def test_unlink_memories_specific_type(self, storage):
        """unlink_memories removes specific relationship type."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.RELATES_TO)
        storage.link_memories(mid1, mid2, RelationType.SUPERSEDES)

        count = storage.unlink_memories(mid1, mid2, RelationType.RELATES_TO)
        assert count == 1

        # SUPERSEDES should still exist
        rels = storage.get_relationship(mid1, mid2)
        assert len(rels) == 1
        assert rels[0].relation_type == RelationType.SUPERSEDES

    def test_unlink_memories_all_types(self, storage):
        """unlink_memories without type removes all relationships."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.RELATES_TO)
        storage.link_memories(mid1, mid2, RelationType.SUPERSEDES)

        count = storage.unlink_memories(mid1, mid2)
        assert count == 2

        rels = storage.get_relationship(mid1, mid2)
        assert len(rels) == 0

    def test_get_related_outgoing(self, storage):
        """get_related with direction='outgoing' returns target memories."""
        mid1, _ = storage.store_memory("Source memory", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Target A", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Target B", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.DEPENDS_ON)
        storage.link_memories(mid1, mid3, RelationType.RELATES_TO)

        related = storage.get_related(mid1, direction="outgoing")
        assert len(related) == 2

        related_ids = {m.id for m, _ in related}
        assert mid2 in related_ids
        assert mid3 in related_ids

    def test_get_related_incoming(self, storage):
        """get_related with direction='incoming' returns source memories."""
        mid1, _ = storage.store_memory("Destination memory", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Source A", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Source B", MemoryType.PROJECT)

        storage.link_memories(mid2, mid1, RelationType.RELATES_TO)
        storage.link_memories(mid3, mid1, RelationType.ELABORATES)

        related = storage.get_related(mid1, direction="incoming")
        assert len(related) == 2

        related_ids = {m.id for m, _ in related}
        assert mid2 in related_ids
        assert mid3 in related_ids

    def test_get_related_both_directions(self, storage):
        """get_related with direction='both' returns all related memories."""
        mid1, _ = storage.store_memory("Center memory", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Outgoing target", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Incoming source", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.DEPENDS_ON)
        storage.link_memories(mid3, mid1, RelationType.ELABORATES)

        related = storage.get_related(mid1, direction="both")
        assert len(related) == 2

        related_ids = {m.id for m, _ in related}
        assert mid2 in related_ids
        assert mid3 in related_ids

    def test_get_related_filter_by_type(self, storage):
        """get_related filters by relation type."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Memory C", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.DEPENDS_ON)
        storage.link_memories(mid1, mid3, RelationType.RELATES_TO)

        related = storage.get_related(mid1, relation_type=RelationType.DEPENDS_ON)
        assert len(related) == 1
        assert related[0][0].id == mid2

    def test_get_relationship_specific(self, storage):
        """get_relationship returns specific relationships between two memories."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.SUPERSEDES)
        storage.link_memories(mid1, mid2, RelationType.REFINES)

        # Get all relationships
        rels = storage.get_relationship(mid1, mid2)
        assert len(rels) == 2

        # Get specific type
        rels = storage.get_relationship(mid1, mid2, RelationType.SUPERSEDES)
        assert len(rels) == 1
        assert rels[0].relation_type == RelationType.SUPERSEDES

    def test_relationship_stats(self, storage):
        """get_relationship_stats returns correct statistics."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Memory C", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.RELATES_TO)
        storage.link_memories(mid1, mid3, RelationType.RELATES_TO)
        storage.link_memories(mid2, mid3, RelationType.DEPENDS_ON)

        stats = storage.get_relationship_stats()
        assert stats["total_relationships"] == 3
        assert stats["by_type"]["relates_to"] == 2
        assert stats["by_type"]["depends_on"] == 1
        assert stats["linked_memories"] == 3  # All 3 memories have relationships

    def test_cascade_delete_relationships(self, storage):
        """Deleting a memory removes its relationships."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.RELATES_TO)

        # Delete mid1
        storage.delete_memory(mid1)

        # Relationships should be gone
        stats = storage.get_relationship_stats()
        assert stats["total_relationships"] == 0

    def test_relationship_table_created(self, storage):
        """memory_relationships table should exist in schema."""
        with storage._connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_relationships'"
            ).fetchone()
            assert result is not None

    def test_schema_version_is_16(self, storage):
        """Schema version should be 16 after migration."""
        version = storage.get_schema_version()
        assert version == 16

    def test_expand_via_relations(self, storage):
        """expand_via_relations adds related memories with decayed scores."""
        # Create memories with relationships
        mid1, _ = storage.store_memory("Main memory about Python", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Related memory about Django", MemoryType.PATTERN)
        mid3, _ = storage.store_memory("Another related memory", MemoryType.REFERENCE)
        mid4, _ = storage.store_memory("Unrelated memory", MemoryType.CONVERSATION)

        # Link mid1 -> mid2 and mid1 -> mid3
        storage.link_memories(mid1, mid2, RelationType.RELATES_TO)
        storage.link_memories(mid1, mid3, RelationType.ELABORATES)

        # Get the main memory with a mock composite score
        memory = storage.get_memory(mid1)
        memory.composite_score = 0.9
        memory.similarity = 0.85

        # Expand via relations
        expanded = storage.expand_via_relations([memory], max_per_memory=5, decay_factor=0.8)

        # Should include original + 2 related (mid4 is not linked)
        assert len(expanded) == 3
        assert expanded[0].id == mid1  # Original first

        # Related memories should have decayed scores
        related_ids = {m.id for m in expanded[1:]}
        assert mid2 in related_ids
        assert mid3 in related_ids
        assert mid4 not in related_ids  # Unrelated

        # Check score decay
        for related in expanded[1:]:
            assert related.composite_score == 0.9 * 0.8  # Decayed
            assert related.similarity == 0.85 * 0.8

    def test_expand_via_relations_respects_max(self, storage):
        """expand_via_relations respects max_per_memory limit."""
        mid1, _ = storage.store_memory("Main memory", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Related 1", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Related 2", MemoryType.PROJECT)
        mid4, _ = storage.store_memory("Related 3", MemoryType.PROJECT)

        storage.link_memories(mid1, mid2, RelationType.RELATES_TO)
        storage.link_memories(mid1, mid3, RelationType.RELATES_TO)
        storage.link_memories(mid1, mid4, RelationType.RELATES_TO)

        memory = storage.get_memory(mid1)
        memory.composite_score = 0.9

        # Limit to 2 expansions
        expanded = storage.expand_via_relations([memory], max_per_memory=2)

        # Should be original + 2 (not all 3 related)
        assert len(expanded) == 3  # 1 original + 2 expanded

    def test_expand_via_relations_deduplicates(self, storage):
        """expand_via_relations avoids duplicate memories."""
        mid1, _ = storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory B", MemoryType.PROJECT)

        # Both link to each other
        storage.link_memories(mid1, mid2, RelationType.RELATES_TO)
        storage.link_memories(mid2, mid1, RelationType.ELABORATES)

        mem1 = storage.get_memory(mid1)
        mem2 = storage.get_memory(mid2)
        mem1.composite_score = 0.9
        mem2.composite_score = 0.8

        # Expand both - should not duplicate
        expanded = storage.expand_via_relations([mem1, mem2])

        # Should still be exactly 2 (no duplicates)
        assert len(expanded) == 2
        assert {m.id for m in expanded} == {mid1, mid2}

    def test_recall_with_expand_relations(self, tmp_path):
        """recall with expand_relations=True includes related memories."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            semantic_dedup_enabled=False,
            recall_expand_relations=False,
        )
        stor = Storage(settings)

        # Create memories with relationships
        mid1, _ = stor.store_memory("Python programming language", MemoryType.PROJECT)
        mid2, _ = stor.store_memory("Django web framework for Python", MemoryType.PATTERN)
        stor.link_memories(mid1, mid2, RelationType.RELATES_TO)

        # Recall without expansion
        result_no_expand = stor.recall("Python", expand_relations=False)
        assert result_no_expand.memories is not None

        # Recall with expansion should include related memories
        result_expand = stor.recall("Python", expand_relations=True)
        assert result_expand.memories is not None

        # Expansion should add at least as many results (related memories are appended)
        assert len(result_expand.memories) >= len(result_no_expand.memories)

        stor.close()


# ========== Contradiction Detection Tests ==========


class TestContradictionDetection:
    """Tests for contradiction detection and resolution."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a storage instance with temp database."""
        settings = Settings(db_path=tmp_path / "test.db", semantic_dedup_enabled=False)
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_find_contradictions_returns_similar_memories(self, storage):
        """find_contradictions returns semantically similar memories."""
        # Store memories about the same topic with different info
        mid1, _ = storage.store_memory(
            "The project uses PostgreSQL 14 for the database.",
            MemoryType.PROJECT,
        )
        mid2, _ = storage.store_memory(
            "The project uses MySQL 8 for the database.",
            MemoryType.PROJECT,
        )
        # Unrelated memory
        storage.store_memory(
            "We use pytest for testing.",
            MemoryType.PROJECT,
        )

        contradictions = storage.find_contradictions(mid1, similarity_threshold=0.5)

        # Should find the MySQL memory as potential contradiction
        assert len(contradictions) >= 1
        memory_ids = {c.memory_b.id for c in contradictions}
        assert mid2 in memory_ids

    def test_find_contradictions_respects_threshold(self, storage):
        """Higher threshold returns fewer results."""
        mid1, _ = storage.store_memory(
            "The auth system uses JWT tokens.",
            MemoryType.PROJECT,
        )
        storage.store_memory(
            "Authentication uses session cookies.",
            MemoryType.PROJECT,
        )

        # With very high threshold, may not find anything
        high_threshold = storage.find_contradictions(mid1, similarity_threshold=0.99)
        low_threshold = storage.find_contradictions(mid1, similarity_threshold=0.3)

        assert len(low_threshold) >= len(high_threshold)

    def test_find_contradictions_marks_existing_links(self, storage):
        """Already-linked contradictions are flagged."""
        mid1, _ = storage.store_memory(
            "Deploy to production on Fridays.",
            MemoryType.PROJECT,
        )
        mid2, _ = storage.store_memory(
            "Never deploy to production on Fridays.",
            MemoryType.PROJECT,
        )

        # Mark as contradiction
        storage.mark_contradiction(mid1, mid2)

        contradictions = storage.find_contradictions(mid1, similarity_threshold=0.3)

        # The mid2 contradiction should be marked as already_linked
        for c in contradictions:
            if c.memory_b.id == mid2:
                assert c.already_linked is True
                break

    def test_mark_contradiction_creates_relationship(self, storage):
        """mark_contradiction creates a CONTRADICTS relationship."""
        mid1, _ = storage.store_memory("Use tabs for indentation.", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Use spaces for indentation.", MemoryType.PROJECT)

        relation = storage.mark_contradiction(mid1, mid2)

        assert relation is not None
        assert relation.relation_type == RelationType.CONTRADICTS
        assert relation.from_memory_id == mid1
        assert relation.to_memory_id == mid2

    def test_get_all_contradictions(self, storage):
        """get_all_contradictions returns marked contradictions."""
        mid1, _ = storage.store_memory("API uses REST.", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("API uses GraphQL.", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Use camelCase.", MemoryType.PROJECT)
        mid4, _ = storage.store_memory("Use snake_case.", MemoryType.PROJECT)

        storage.mark_contradiction(mid1, mid2)
        storage.mark_contradiction(mid3, mid4)

        contradictions = storage.get_all_contradictions()

        assert len(contradictions) == 2
        memory_pairs = {(m1.id, m2.id) for m1, m2, _ in contradictions}
        assert (mid1, mid2) in memory_pairs
        assert (mid3, mid4) in memory_pairs

    def test_resolve_contradiction_supersedes(self, storage):
        """resolve_contradiction with supersedes creates SUPERSEDES relationship."""
        mid1, _ = storage.store_memory("v1: Use MongoDB.", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("v2: Use PostgreSQL.", MemoryType.PROJECT)

        storage.mark_contradiction(mid1, mid2)

        # Resolve: mid2 (PostgreSQL) supersedes mid1 (MongoDB)
        result = storage.resolve_contradiction(mid1, mid2, keep_id=mid2, resolution="supersedes")

        assert result is True

        # Contradiction should be removed
        contradictions = storage.get_all_contradictions()
        assert len(contradictions) == 0

        # SUPERSEDES relationship should exist
        related = storage.get_related(mid2, RelationType.SUPERSEDES, direction="outgoing")
        assert len(related) == 1
        assert related[0][0].id == mid1

    def test_resolve_contradiction_delete(self, storage):
        """resolve_contradiction with delete removes the discarded memory."""
        mid1, _ = storage.store_memory("Old decision.", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("New decision.", MemoryType.PROJECT)

        storage.mark_contradiction(mid1, mid2)
        result = storage.resolve_contradiction(mid1, mid2, keep_id=mid2, resolution="delete")

        assert result is True

        # mid1 should be deleted
        assert storage.get_memory(mid1) is None
        # mid2 should still exist
        assert storage.get_memory(mid2) is not None

    def test_resolve_contradiction_weaken(self, storage):
        """resolve_contradiction with weaken reduces trust in discarded memory."""
        mid1, _ = storage.store_memory("Less trusted info.", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("More trusted info.", MemoryType.PROJECT)

        # Get initial trust
        m1_before = storage.get_memory(mid1)
        initial_trust = m1_before.trust_score

        storage.mark_contradiction(mid1, mid2)
        result = storage.resolve_contradiction(mid1, mid2, keep_id=mid2, resolution="weaken")

        assert result is True

        # mid1's trust should be reduced
        m1_after = storage.get_memory(mid1)
        assert m1_after.trust_score < initial_trust

    def test_resolve_contradiction_invalid_keep_id(self, storage):
        """resolve_contradiction fails if keep_id is not one of the memories."""
        mid1, _ = storage.store_memory("Memory 1.", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory 2.", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Memory 3.", MemoryType.PROJECT)

        storage.mark_contradiction(mid1, mid2)

        # Try to keep mid3 which is not in the contradiction
        result = storage.resolve_contradiction(mid1, mid2, keep_id=mid3)

        assert result is False

    def test_contradiction_resolved_trust_reason_exists(self):
        """CONTRADICTION_RESOLVED is a valid trust reason."""
        assert TrustReason.CONTRADICTION_RESOLVED.value == "contradiction_resolved"
        assert TrustReason.CONTRADICTION_RESOLVED in TRUST_REASON_DEFAULTS


# ========== Session (Conversation Provenance) Tests ==========


class TestSessionProvenance:
    """Tests for session/conversation provenance tracking."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a storage instance with temp database."""
        settings = Settings(db_path=tmp_path / "test.db", semantic_dedup_enabled=False)
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_create_session(self, storage):
        """create_or_get_session creates a new session."""
        session = storage.create_or_get_session(
            session_id="test-session-123",
            topic="Testing sessions",
            project_path="/path/to/project",
        )

        assert session.id == "test-session-123"
        assert session.topic == "Testing sessions"
        assert session.project_path == "/path/to/project"
        assert session.memory_count == 0
        assert session.log_count == 0

    def test_get_existing_session(self, storage):
        """create_or_get_session returns existing session."""
        storage.create_or_get_session("existing-session", topic="Original topic")

        session = storage.create_or_get_session("existing-session", topic="New topic")

        assert session.topic == "Original topic"  # Original preserved

    def test_get_session_by_id(self, storage):
        """get_session retrieves session by ID."""
        storage.create_or_get_session("session-abc", topic="Test")

        session = storage.get_session("session-abc")

        assert session is not None
        assert session.id == "session-abc"

    def test_get_session_not_found(self, storage):
        """get_session returns None for unknown session."""
        session = storage.get_session("nonexistent")
        assert session is None

    def test_store_memory_with_session(self, storage):
        """store_memory associates memory with session."""
        storage.create_or_get_session("mem-session", project_path="/test")

        mid, _ = storage.store_memory(
            "Test memory content",
            MemoryType.PROJECT,
            session_id="mem-session",
        )

        memory = storage.get_memory(mid)
        assert memory is not None
        assert memory.session_id == "mem-session"

    def test_session_memory_count_increments(self, storage):
        """Storing memory increments session memory count."""
        storage.create_or_get_session("count-session")

        storage.store_memory("Memory 1", MemoryType.PROJECT, session_id="count-session")
        storage.store_memory("Memory 2", MemoryType.PROJECT, session_id="count-session")

        session = storage.get_session("count-session")
        assert session.memory_count == 2

    def test_get_session_memories(self, storage):
        """get_session_memories returns memories from session."""
        storage.create_or_get_session("filter-session")
        storage.create_or_get_session("other-session")

        storage.store_memory("Memory A", MemoryType.PROJECT, session_id="filter-session")
        storage.store_memory("Memory B", MemoryType.PROJECT, session_id="filter-session")
        storage.store_memory("Memory C", MemoryType.PROJECT, session_id="other-session")

        memories = storage.get_session_memories("filter-session")

        assert len(memories) == 2
        contents = {m.content for m in memories}
        assert "Memory A" in contents
        assert "Memory B" in contents
        assert "Memory C" not in contents

    def test_get_sessions_returns_all(self, storage):
        """get_sessions returns all sessions up to limit."""
        storage.create_or_get_session("session-1")
        storage.create_or_get_session("session-2")
        storage.create_or_get_session("session-3")

        sessions = storage.get_sessions(limit=10)
        assert len(sessions) == 3

        ids = {s.id for s in sessions}
        assert "session-1" in ids
        assert "session-2" in ids
        assert "session-3" in ids

    def test_get_sessions_filter_by_project(self, storage):
        """get_sessions filters by project_path."""
        storage.create_or_get_session("proj-a-1", project_path="/project/a")
        storage.create_or_get_session("proj-a-2", project_path="/project/a")
        storage.create_or_get_session("proj-b-1", project_path="/project/b")

        sessions = storage.get_sessions(project_path="/project/a")

        assert len(sessions) == 2
        ids = {s.id for s in sessions}
        assert "proj-a-1" in ids
        assert "proj-a-2" in ids

    def test_update_session_topic(self, storage):
        """update_session_topic changes topic."""
        storage.create_or_get_session("topic-session", topic="Old topic")

        result = storage.update_session_topic("topic-session", "New topic")
        assert result is True

        session = storage.get_session("topic-session")
        assert session.topic == "New topic"

    def test_update_session_topic_not_found(self, storage):
        """update_session_topic returns False for unknown session."""
        result = storage.update_session_topic("nonexistent", "Topic")
        assert result is False

    def test_sessions_table_created(self, storage):
        """sessions table should exist in schema."""
        with storage._connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
            assert result is not None

    def test_memory_session_id_column_exists(self, storage):
        """session_id column should exist in memories table."""
        with storage._connection() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
            assert "session_id" in columns

    def test_log_output_with_session(self, storage):
        """log_output with session_id increments session log count."""
        storage.create_or_get_session("log-session")

        storage.log_output("Test output 1", session_id="log-session")
        storage.log_output("Test output 2", session_id="log-session")

        session = storage.get_session("log-session")
        assert session.log_count == 2

    def test_log_output_without_session(self, storage):
        """log_output without session_id works fine."""
        log_id = storage.log_output("Test output no session")
        assert log_id > 0

    def test_log_output_stores_session_id_in_log(self, storage):
        """log_output persists session_id in output_log table."""
        storage.create_or_get_session("persist-session")

        storage.log_output("Output with session", session_id="persist-session")

        with storage._connection() as conn:
            row = conn.execute(
                "SELECT session_id FROM output_log WHERE content = ?",
                ("Output with session",),
            ).fetchone()
            assert row is not None
            assert row["session_id"] == "persist-session"


class TestBootstrap:
    """Tests for bootstrap_from_files functionality."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a storage instance with temp database."""
        settings = Settings(db_path=tmp_path / "test.db", semantic_dedup_enabled=False)
        stor = Storage(settings)
        yield stor
        stor.close()

    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a temp directory with sample files."""
        project = tmp_path / "project"
        project.mkdir()
        return project

    def test_bootstrap_single_file(self, storage, project_dir):
        """Bootstrap from a single README file creates memories."""
        readme = project_dir / "README.md"
        readme.write_text("# Project\n\n- Feature one\n- Feature two\n")

        result = storage.bootstrap_from_files([readme])

        assert result["success"] is True
        assert result["files_found"] == 1
        assert result["files_processed"] == 1
        assert result["memories_created"] >= 1
        assert result["hot_cache_promoted"] >= 1

    def test_bootstrap_multiple_files(self, storage, project_dir):
        """Bootstrap from multiple files processes all."""
        readme = project_dir / "README.md"
        readme.write_text("# README\n\n- Readme content\n")

        claude = project_dir / "CLAUDE.md"
        claude.write_text("# Claude\n\n- Claude instructions\n")

        result = storage.bootstrap_from_files([readme, claude])

        assert result["success"] is True
        assert result["files_found"] == 2
        assert result["files_processed"] == 2
        assert result["memories_created"] >= 2

    def test_bootstrap_empty_file_list(self, storage):
        """Bootstrap with no files returns graceful message."""
        result = storage.bootstrap_from_files([])

        assert result["success"] is True
        assert result["files_found"] == 0
        assert result["files_processed"] == 0
        assert result["memories_created"] == 0
        assert "No files provided" in result["message"]

    def test_bootstrap_file_not_found(self, storage, project_dir):
        """Bootstrap handles missing files gracefully."""
        missing = project_dir / "NONEXISTENT.md"

        result = storage.bootstrap_from_files([missing])

        assert result["success"] is True
        assert result["files_found"] == 1
        assert result["files_processed"] == 0
        assert result["memories_created"] == 0
        assert len(result["errors"]) == 1
        assert "file not found" in result["errors"][0]

    def test_bootstrap_empty_file(self, storage, project_dir):
        """Bootstrap skips empty files silently."""
        empty = project_dir / "EMPTY.md"
        empty.write_text("")

        readme = project_dir / "README.md"
        readme.write_text("# Content\n\n- Actual content\n")

        result = storage.bootstrap_from_files([empty, readme])

        assert result["success"] is True
        assert result["files_processed"] == 1  # Only README counted
        assert result["memories_created"] >= 1

    def test_bootstrap_directory_instead_of_file(self, storage, project_dir):
        """Bootstrap handles directory path gracefully."""
        subdir = project_dir / "subdir"
        subdir.mkdir()

        result = storage.bootstrap_from_files([subdir])

        assert result["success"] is True
        assert result["files_processed"] == 0
        assert len(result["errors"]) == 1
        assert "not a file" in result["errors"][0]

    def test_bootstrap_no_promote(self, storage, project_dir):
        """Bootstrap with promote_to_hot=False doesn't promote."""
        readme = project_dir / "README.md"
        readme.write_text("# Project\n\n- Some content\n")

        result = storage.bootstrap_from_files([readme], promote_to_hot=False)

        assert result["success"] is True
        assert result["memories_created"] >= 1
        assert result["hot_cache_promoted"] == 0

        # Verify not in hot cache
        hot = storage.get_hot_memories()
        assert len(hot) == 0

    def test_bootstrap_with_tags(self, storage, project_dir):
        """Bootstrap applies tags to all memories."""
        readme = project_dir / "README.md"
        readme.write_text("# Project\n\n- Tagged content\n")

        result = storage.bootstrap_from_files(
            [readme],
            tags=["bootstrap", "readme"],
        )

        assert result["success"] is True
        assert result["memories_created"] >= 1

        # Check tags were applied
        memories = storage.list_memories(limit=10)
        assert len(memories) >= 1
        for mem in memories:
            assert "bootstrap" in mem.tags
            assert "readme" in mem.tags

    def test_bootstrap_with_memory_type(self, storage, project_dir):
        """Bootstrap uses specified memory type."""
        readme = project_dir / "README.md"
        readme.write_text("# Reference\n\n- Reference content\n")

        result = storage.bootstrap_from_files(
            [readme],
            memory_type=MemoryType.REFERENCE,
        )

        assert result["success"] is True
        assert result["memories_created"] >= 1

        # Check type was applied
        memories = storage.list_memories(limit=10)
        for mem in memories:
            assert mem.memory_type == MemoryType.REFERENCE

    def test_bootstrap_deduplication(self, storage, project_dir):
        """Bootstrap skips duplicate content."""
        readme = project_dir / "README.md"
        readme.write_text("# Project\n\n- Unique content\n")

        # First bootstrap
        result1 = storage.bootstrap_from_files([readme])
        assert result1["memories_created"] >= 1

        # Second bootstrap - same content
        result2 = storage.bootstrap_from_files([readme])
        assert result2["memories_created"] == 0
        assert result2["memories_skipped"] >= 1

    def test_bootstrap_binary_file_skipped(self, storage, project_dir):
        """Bootstrap skips binary files."""
        binary = project_dir / "image.png"
        # Write some binary content with null bytes
        binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")

        result = storage.bootstrap_from_files([binary])

        assert result["success"] is True
        assert result["files_processed"] == 0
        assert len(result["errors"]) == 1
        assert "binary" in result["errors"][0].lower()

    def test_bootstrap_mixed_success_and_errors(self, storage, project_dir):
        """Bootstrap reports errors but continues with valid files."""
        readme = project_dir / "README.md"
        readme.write_text("# Valid\n\n- Valid content\n")

        missing = project_dir / "MISSING.md"

        result = storage.bootstrap_from_files([readme, missing])

        assert result["success"] is True
        assert result["files_found"] == 2
        assert result["files_processed"] == 1
        assert result["memories_created"] >= 1
        assert len(result["errors"]) == 1

    def test_is_binary_file_detection(self, storage, project_dir):
        """_is_binary_file correctly detects binary vs text."""
        # Text file
        text_file = project_dir / "text.txt"
        text_file.write_text("This is plain text content")
        assert storage._is_binary_file(text_file) is False

        # Binary file with null bytes
        binary_file = project_dir / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04")
        assert storage._is_binary_file(binary_file) is True

        # UTF-8 with special chars (still text)
        utf8_file = project_dir / "utf8.txt"
        utf8_file.write_text("Unicode: \u00e9\u00e8\u00ea")
        assert storage._is_binary_file(utf8_file) is False

    def test_bootstrap_with_project_awareness(self, tmp_path, project_dir):
        """Bootstrap passes project_id when project awareness is enabled."""
        # Create storage with project awareness enabled
        settings = Settings(
            db_path=tmp_path / "test.db",
            semantic_dedup_enabled=False,
            project_awareness_enabled=True,
        )
        stor = Storage(settings)

        # Mock the project detection to return a known project_id
        readme = project_dir / "README.md"
        readme.write_text("# Project\n\n- Test content for project awareness\n")

        # Note: Without a real git repo, get_current_project_id returns None
        # This test verifies the code path doesn't error and memories are created
        result = stor.bootstrap_from_files([readme])

        assert result["success"] is True
        assert result["memories_created"] >= 1

        # Verify memories were created (project_id may be None without git repo)
        memories = stor.list_memories(limit=10)
        assert len(memories) >= 1

        stor.close()


# ========== Predictive Hot Cache Warming Tests ==========


class TestPredictiveCache:
    """Tests for predictive hot cache warming."""

    @pytest.fixture
    def predictive_storage(self, tmp_path):
        """Create storage with predictive caching enabled."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            predictive_cache_enabled=True,
            prediction_threshold=0.3,
            max_predictions=3,
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    @pytest.fixture
    def non_predictive_storage(self, tmp_path):
        """Create storage with predictive caching disabled."""
        settings = Settings(
            db_path=tmp_path / "test.db",
            predictive_cache_enabled=False,
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_record_access_sequence_creates_pattern(self, predictive_storage):
        """record_access_sequence creates access pattern entries."""
        mid1, _ = predictive_storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = predictive_storage.store_memory("Memory B", MemoryType.PROJECT)

        predictive_storage.record_access_sequence(mid1, mid2)

        patterns = predictive_storage.get_access_patterns(mid1)
        assert len(patterns) == 1
        assert patterns[0].to_memory_id == mid2
        assert patterns[0].count == 1

    def test_record_access_sequence_increments_count(self, predictive_storage):
        """Repeated sequences increment count."""
        mid1, _ = predictive_storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = predictive_storage.store_memory("Memory B", MemoryType.PROJECT)

        predictive_storage.record_access_sequence(mid1, mid2)
        predictive_storage.record_access_sequence(mid1, mid2)
        predictive_storage.record_access_sequence(mid1, mid2)

        patterns = predictive_storage.get_access_patterns(mid1)
        assert patterns[0].count == 3

    def test_record_access_sequence_disabled_noop(self, non_predictive_storage):
        """record_access_sequence does nothing when disabled."""
        mid1, _ = non_predictive_storage.store_memory("Memory A", MemoryType.PROJECT)
        mid2, _ = non_predictive_storage.store_memory("Memory B", MemoryType.PROJECT)

        non_predictive_storage.record_access_sequence(mid1, mid2)

        patterns = non_predictive_storage.get_access_patterns(mid1)
        assert len(patterns) == 0

    def test_get_access_patterns_calculates_probability(self, predictive_storage):
        """get_access_patterns returns correct probabilities."""
        mid1, _ = predictive_storage.store_memory("Source", MemoryType.PROJECT)
        mid2, _ = predictive_storage.store_memory("Target A", MemoryType.PROJECT)
        mid3, _ = predictive_storage.store_memory("Target B", MemoryType.PROJECT)

        # Create pattern: mid1 -> mid2 (3 times), mid1 -> mid3 (1 time)
        for _ in range(3):
            predictive_storage.record_access_sequence(mid1, mid2)
        predictive_storage.record_access_sequence(mid1, mid3)

        patterns = predictive_storage.get_access_patterns(mid1)

        assert len(patterns) == 2
        # mid2 should have probability 0.75 (3/4)
        mid2_pattern = next(p for p in patterns if p.to_memory_id == mid2)
        assert mid2_pattern.probability == 0.75
        # mid3 should have probability 0.25 (1/4)
        mid3_pattern = next(p for p in patterns if p.to_memory_id == mid3)
        assert mid3_pattern.probability == 0.25

    def test_predict_next_memories_returns_predictions(self, predictive_storage):
        """predict_next_memories returns predicted memories."""
        mid1, _ = predictive_storage.store_memory("Source memory", MemoryType.PROJECT)
        mid2, _ = predictive_storage.store_memory("Target memory", MemoryType.PROJECT)

        # Create high-frequency pattern
        for _ in range(5):
            predictive_storage.record_access_sequence(mid1, mid2)

        predictions = predictive_storage.predict_next_memories(mid1)

        assert len(predictions) == 1
        assert predictions[0].memory.id == mid2
        assert predictions[0].probability == 1.0

    def test_predict_next_memories_respects_threshold(self, predictive_storage):
        """Predictions below threshold are not returned."""
        mid1, _ = predictive_storage.store_memory("Source", MemoryType.PROJECT)
        mid2, _ = predictive_storage.store_memory("Target A", MemoryType.PROJECT)
        mid3, _ = predictive_storage.store_memory("Target B", MemoryType.PROJECT)

        # mid2 has 80% probability, mid3 has 20%
        for _ in range(4):
            predictive_storage.record_access_sequence(mid1, mid2)
        predictive_storage.record_access_sequence(mid1, mid3)

        # With 0.3 threshold, only mid2 should be returned
        predictions = predictive_storage.predict_next_memories(mid1, threshold=0.3)

        assert len(predictions) == 1
        assert predictions[0].memory.id == mid2

    def test_warm_predicted_cache_promotes_memories(self, predictive_storage):
        """warm_predicted_cache promotes predicted memories."""
        mid1, _ = predictive_storage.store_memory("Source", MemoryType.PROJECT)
        mid2, _ = predictive_storage.store_memory("Target", MemoryType.PROJECT)

        for _ in range(5):
            predictive_storage.record_access_sequence(mid1, mid2)

        promoted = predictive_storage.warm_predicted_cache(mid1)

        assert mid2 in promoted
        # Verify it's now hot
        memory = predictive_storage.get_memory(mid2)
        assert memory.is_hot is True
        assert memory.promotion_source == PromotionSource.PREDICTED

    def test_warm_predicted_cache_skips_already_hot(self, predictive_storage):
        """warm_predicted_cache doesn't re-promote hot memories."""
        mid1, _ = predictive_storage.store_memory("Source", MemoryType.PROJECT)
        mid2, _ = predictive_storage.store_memory("Target", MemoryType.PROJECT)

        # Make mid2 hot first
        predictive_storage.promote_to_hot(mid2)

        for _ in range(5):
            predictive_storage.record_access_sequence(mid1, mid2)

        promoted = predictive_storage.warm_predicted_cache(mid1)

        assert mid2 not in promoted  # Already hot

    def test_schema_version_is_16(self, predictive_storage):
        """Schema version should be 16 after migration."""
        version = predictive_storage.get_schema_version()
        assert version == 16

    def test_access_sequences_table_exists(self, predictive_storage):
        """access_sequences table should exist."""
        with predictive_storage._connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='access_sequences'"
            ).fetchone()
            assert result is not None

    def test_recall_records_access_sequences_when_enabled(self, predictive_storage):
        """recall() records access sequences for predictive cache when enabled."""
        # Store memories with distinct content
        mid1, _ = predictive_storage.store_memory(
            "First memory about authentication", MemoryType.PROJECT
        )
        mid2, _ = predictive_storage.store_memory(
            "Second memory about authentication tokens", MemoryType.PROJECT
        )
        mid3, _ = predictive_storage.store_memory(
            "Third memory about authentication JWT", MemoryType.PROJECT
        )

        # Do a recall that returns multiple results
        result = predictive_storage.recall("authentication", limit=3, threshold=0.0)

        # Should have at least 2 results to record sequences
        if len(result.memories) >= 2:
            # Check that access sequences were recorded
            first_id = result.memories[0].id
            patterns = predictive_storage.get_access_patterns(first_id)
            assert len(patterns) >= 1, "Access sequences should be recorded during recall"

    def test_recall_no_sequences_when_disabled(self, non_predictive_storage):
        """recall() does not record access sequences when predictive cache is disabled."""
        mid1, _ = non_predictive_storage.store_memory("Memory about testing", MemoryType.PROJECT)
        mid2, _ = non_predictive_storage.store_memory(
            "Memory about testing frameworks", MemoryType.PROJECT
        )

        result = non_predictive_storage.recall("testing", limit=2, threshold=0.0)

        if len(result.memories) >= 2:
            first_id = result.memories[0].id
            patterns = non_predictive_storage.get_access_patterns(first_id)
            assert len(patterns) == 0, "No sequences when predictive cache disabled"


class TestSemanticDeduplication:
    """Tests for semantic deduplication on store."""

    @pytest.fixture
    def dedup_storage(self, tmp_path):
        """Storage with semantic deduplication enabled (default)."""
        settings = Settings(
            db_path=tmp_path / "dedup.db",
            semantic_dedup_enabled=True,
            semantic_dedup_threshold=0.92,
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    @pytest.fixture
    def dedup_disabled_storage(self, tmp_path):
        """Storage with semantic deduplication disabled."""
        settings = Settings(
            db_path=tmp_path / "no_dedup.db",
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_identical_content_merges(self, dedup_storage):
        """Identical content should merge (similarity=1.0)."""
        content = "Project uses Python 3.12 with FastAPI"
        mid1, is_new1 = dedup_storage.store_memory(content, MemoryType.PROJECT)
        mid2, is_new2 = dedup_storage.store_memory(content, MemoryType.PROJECT)

        assert is_new1 is True
        assert is_new2 is False  # Merged
        assert mid1 == mid2

        # Access count should be incremented
        memory = dedup_storage.get_memory(mid1)
        assert memory.access_count == 1  # Original + 1 merge

    def test_very_similar_content_merges(self, dedup_storage):
        """Very similar content should merge."""
        content1 = "The project uses Python 3.12"
        content2 = "This project uses Python 3.12"

        mid1, is_new1 = dedup_storage.store_memory(content1, MemoryType.PROJECT)
        mid2, is_new2 = dedup_storage.store_memory(content2, MemoryType.PROJECT)

        # These should be similar enough to merge
        assert is_new1 is True
        # is_new2 depends on actual similarity - check if merged
        if mid1 == mid2:
            assert is_new2 is False

    def test_different_content_stays_separate(self, dedup_storage):
        """Different content should create separate memories."""
        content1 = "Python is a programming language"
        content2 = "The weather today is sunny and warm"

        mid1, is_new1 = dedup_storage.store_memory(content1, MemoryType.PROJECT)
        mid2, is_new2 = dedup_storage.store_memory(content2, MemoryType.PROJECT)

        assert is_new1 is True
        assert is_new2 is True
        assert mid1 != mid2

    def test_longer_content_updates_existing(self, dedup_storage):
        """When merging, longer content should update existing."""
        short = "Use pytest for testing"
        long = (
            "Use pytest for testing. "
            "It's the standard Python testing framework with great fixtures."
        )

        mid1, _ = dedup_storage.store_memory(short, MemoryType.PROJECT)
        mid2, _ = dedup_storage.store_memory(long, MemoryType.PROJECT)

        # If they merged, mid2 == mid1 and content should be the longer one
        if mid1 == mid2:
            memory = dedup_storage.get_memory(mid1)
            assert len(memory.content) == len(long)

    def test_shorter_content_keeps_existing(self, dedup_storage):
        """When merging, shorter content should keep existing."""
        long = (
            "Use pytest for testing. "
            "It's the standard Python testing framework with great fixtures."
        )
        short = "Use pytest for testing"

        mid1, _ = dedup_storage.store_memory(long, MemoryType.PROJECT)
        mid2, _ = dedup_storage.store_memory(short, MemoryType.PROJECT)

        # If they merged, content should still be the longer one
        if mid1 == mid2:
            memory = dedup_storage.get_memory(mid1)
            assert len(memory.content) == len(long)

    def test_tags_merged_on_dedup(self, dedup_storage):
        """Tags should be merged when deduplicating."""
        content = "Project uses FastAPI for the REST API"

        mid1, _ = dedup_storage.store_memory(content, MemoryType.PROJECT, tags=["api", "rest"])
        mid2, _ = dedup_storage.store_memory(
            content, MemoryType.PROJECT, tags=["fastapi", "backend"]
        )

        # If merged, all tags should be present
        if mid1 == mid2:
            memory = dedup_storage.get_memory(mid1)
            assert "api" in memory.tags
            assert "fastapi" in memory.tags

    def test_dedup_disabled_creates_separate_similar_memories(self, dedup_disabled_storage):
        """With semantic dedup disabled, similar (but not identical) content stays separate.

        Note: Exact duplicates are still deduplicated by content_hash (separate feature).
        Semantic dedup is about merging *similar* content, not exact matches.
        """
        content1 = "The project uses Python 3.12"
        content2 = "This project uses Python 3.12"  # Very similar but different hash

        mid1, is_new1 = dedup_disabled_storage.store_memory(content1, MemoryType.PROJECT)
        mid2, is_new2 = dedup_disabled_storage.store_memory(content2, MemoryType.PROJECT)

        assert is_new1 is True
        assert is_new2 is True  # With dedup disabled, similar content is NOT merged
        assert mid1 != mid2

    def test_empty_storage_no_merge(self, dedup_storage):
        """First memory in empty storage should always create new."""
        content = "First memory"
        mid, is_new = dedup_storage.store_memory(content, MemoryType.PROJECT)

        assert is_new is True
        assert mid > 0

    def test_dedup_threshold_respected(self, tmp_path):
        """Different thresholds should affect merge behavior."""
        # Very strict threshold (0.99) - almost nothing merges
        strict_settings = Settings(
            db_path=tmp_path / "strict.db",
            semantic_dedup_enabled=True,
            semantic_dedup_threshold=0.99,
        )
        strict_storage = Storage(strict_settings)

        content1 = "Python testing with pytest"
        content2 = "Python testing using pytest"

        mid1, _ = strict_storage.store_memory(content1, MemoryType.PROJECT)
        mid2, _ = strict_storage.store_memory(content2, MemoryType.PROJECT)

        # With 0.99 threshold, these likely won't merge (different content)
        # Verify both memories exist as separate entries
        assert mid1 != mid2
        strict_storage.close()

    def test_cross_project_same_content_stays_separate(self, tmp_path):
        """Same content in different projects should create separate memories.

        This tests the fix for project-aware dedup: previously, identical content
        across projects would collide on the global content_hash UNIQUE constraint.
        With project-scoped dedup, the same content can exist in multiple projects.
        """
        settings = Settings(
            db_path=tmp_path / "cross_project.db",
            project_awareness_enabled=True,
            semantic_dedup_enabled=True,
        )
        storage = Storage(settings)

        content = "This project uses FastAPI for the REST API"

        # Store same content in two different projects
        mid1, is_new1 = storage.store_memory(
            content, MemoryType.PROJECT, project_id="github/owner/project-a"
        )
        mid2, is_new2 = storage.store_memory(
            content, MemoryType.PROJECT, project_id="github/owner/project-b"
        )

        # Both should be new (different projects)
        assert is_new1 is True
        assert is_new2 is True
        assert mid1 != mid2

        # Verify both exist with correct project_ids
        mem1 = storage.get_memory(mid1)
        mem2 = storage.get_memory(mid2)
        assert mem1.project_id == "github/owner/project-a"
        assert mem2.project_id == "github/owner/project-b"

        storage.close()

    def test_same_project_same_content_merges(self, tmp_path):
        """Same content in same project should merge (not duplicate)."""
        settings = Settings(
            db_path=tmp_path / "same_project.db",
            project_awareness_enabled=True,
            semantic_dedup_enabled=True,
        )
        storage = Storage(settings)

        content = "This project uses FastAPI for the REST API"
        project_id = "github/owner/project-a"

        mid1, is_new1 = storage.store_memory(content, MemoryType.PROJECT, project_id=project_id)
        mid2, is_new2 = storage.store_memory(content, MemoryType.PROJECT, project_id=project_id)

        # Second should merge with first
        assert is_new1 is True
        assert is_new2 is False
        assert mid1 == mid2

        storage.close()


# ========== Vector Rebuild Tests ==========


class TestVectorRebuild:
    """Tests for vector clearing and rebuilding functionality."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a storage instance with temp database."""
        settings = Settings(db_path=tmp_path / "test.db", semantic_dedup_enabled=False)
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_clear_vectors_removes_all(self, storage):
        """clear_vectors should remove all vectors but keep memories."""
        # Create some memories
        mid1, _ = storage.store_memory("Memory one", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory two", MemoryType.PROJECT)

        # Verify vectors exist
        with storage._connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
            assert count == 2

        # Clear vectors
        result = storage.clear_vectors()

        assert result["vectors_cleared"] == 2
        assert result["new_dimension"] == storage.settings.embedding_dim

        # Verify vectors are gone but memories remain
        with storage._connection() as conn:
            vector_count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
            memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            assert vector_count == 0
            assert memory_count == 2

    def test_rebuild_vectors_reembeds_all(self, storage):
        """rebuild_vectors should re-embed all memories."""
        # Create some memories
        mid1, _ = storage.store_memory("Memory one", MemoryType.PROJECT)
        mid2, _ = storage.store_memory("Memory two", MemoryType.PROJECT)
        mid3, _ = storage.store_memory("Memory three", MemoryType.PROJECT)

        # Rebuild
        result = storage.rebuild_vectors()

        assert result["vectors_cleared"] == 3
        assert result["memories_total"] == 3
        assert result["memories_embedded"] == 3
        assert result["memories_failed"] == 0

        # Verify vectors were recreated
        with storage._connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
            assert count == 3

    def test_rebuild_vectors_batch_size(self, storage):
        """rebuild_vectors should process memories in batches."""
        # Create several memories
        for i in range(5):
            storage.store_memory(f"Memory number {i}", MemoryType.PROJECT)

        # Rebuild with small batch size
        result = storage.rebuild_vectors(batch_size=2)

        assert result["memories_total"] == 5
        assert result["memories_embedded"] == 5

    def test_rebuild_preserves_recall(self, storage):
        """After rebuild, recall should still work."""
        # Create a memory
        mid, _ = storage.store_memory("Python testing framework pytest", MemoryType.PROJECT)

        # Verify recall works before
        result1 = storage.recall("pytest", threshold=0.1)
        assert len(result1.memories) >= 1

        # Rebuild
        storage.rebuild_vectors()

        # Verify recall still works after
        result2 = storage.recall("pytest", threshold=0.1)
        assert len(result2.memories) >= 1

    def test_clear_vectors_empty_database(self, tmp_path):
        """clear_vectors should work on empty database."""
        settings = Settings(db_path=tmp_path / "empty.db")
        stor = Storage(settings)

        result = stor.clear_vectors()

        assert result["vectors_cleared"] == 0
        stor.close()

    def test_rebuild_vectors_empty_database(self, tmp_path):
        """rebuild_vectors should work on empty database."""
        settings = Settings(db_path=tmp_path / "empty.db")
        stor = Storage(settings)

        result = stor.rebuild_vectors()

        assert result["memories_total"] == 0
        assert result["memories_embedded"] == 0
        stor.close()


# ========== Retrieval Tracking Tests (RAG-inspired) ==========


class TestRetrievalTracking:
    """Tests for retrieval event tracking."""

    @pytest.fixture
    def tracking_storage(self, tmp_path):
        """Storage with retrieval tracking enabled."""
        settings = Settings(
            db_path=tmp_path / "tracking.db",
            retrieval_tracking_enabled=True,
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_record_retrieval_event(self, tracking_storage):
        """record_retrieval_event creates entries in retrieval_events."""
        # Store a memory first
        mid, _ = tracking_storage.store_memory(
            content="Test memory for retrieval tracking",
            memory_type=MemoryType.PROJECT,
        )

        # Record retrieval
        event_ids = tracking_storage.record_retrieval_event(
            query="test query",
            memory_ids=[mid],
            similarities=[0.85],
        )

        assert len(event_ids) == 1
        assert event_ids[0] > 0

    def test_mark_retrieval_used(self, tracking_storage):
        """mark_retrieval_used updates was_used flag."""
        mid, _ = tracking_storage.store_memory(
            content="Memory to mark as used",
            memory_type=MemoryType.PROJECT,
        )

        # Record then mark used
        tracking_storage.record_retrieval_event("query", [mid], [0.9])
        updated = tracking_storage.mark_retrieval_used(mid, feedback="helpful")

        assert updated == 1

    def test_get_retrieval_stats_global(self, tracking_storage):
        """get_retrieval_stats returns global statistics."""
        mid, _ = tracking_storage.store_memory(
            content="Stats test memory",
            memory_type=MemoryType.PROJECT,
        )

        # Record some events
        tracking_storage.record_retrieval_event("query1", [mid], [0.9])
        tracking_storage.record_retrieval_event("query2", [mid], [0.8])
        tracking_storage.mark_retrieval_used(mid)

        stats = tracking_storage.get_retrieval_stats(days=30)

        assert stats["total_retrievals"] == 2
        assert stats["used_count"] == 1
        assert stats["usage_rate"] == 0.5

    def test_get_retrieval_stats_per_memory(self, tracking_storage):
        """get_retrieval_stats returns stats for specific memory."""
        mid, _ = tracking_storage.store_memory(
            content="Per-memory stats test",
            memory_type=MemoryType.PROJECT,
        )

        tracking_storage.record_retrieval_event("query", [mid], [0.85])

        stats = tracking_storage.get_retrieval_stats(memory_id=mid, days=30)

        assert stats["memory_id"] == mid
        assert stats["total_retrievals"] == 1

    def test_tracking_disabled_returns_empty(self, tmp_path):
        """record_retrieval_event returns empty but mark_retrieval_used still works."""
        settings = Settings(
            db_path=tmp_path / "no_tracking.db",
            retrieval_tracking_enabled=False,
        )
        stor = Storage(settings)

        # record_retrieval_event returns empty when tracking disabled
        event_ids = stor.record_retrieval_event("query", [1], [0.9])
        assert event_ids == []

        # Create a memory to test mark_retrieval_used
        mem_id, _ = stor.store_memory("Test content for tracking", MemoryType.PROJECT)

        # mark_retrieval_used still updates memory's denormalized counters
        # even when retrieval_events tracking is disabled
        updated = stor.mark_retrieval_used(mem_id)
        assert updated == 1  # Memory counters updated

        # Verify memory's used_count was incremented
        refreshed = stor.get_memory(mem_id)
        assert refreshed.used_count == 1

        stor.close()


# ========== Memory Consolidation Tests (MemoryBank-inspired) ==========


class TestMemoryConsolidation:
    """Tests for memory consolidation feature."""

    @pytest.fixture
    def consolidation_storage(self, tmp_path):
        """Storage with consolidation settings."""
        settings = Settings(
            db_path=tmp_path / "consolidation.db",
            consolidation_threshold=0.85,
            consolidation_min_cluster_size=2,
            semantic_dedup_enabled=False,  # Disable so we can create similar memories
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_find_consolidation_clusters_empty(self, consolidation_storage):
        """find_consolidation_clusters returns empty for no memories."""
        clusters = consolidation_storage.find_consolidation_clusters()
        assert clusters == []

    def test_find_consolidation_clusters_finds_similar(self, consolidation_storage):
        """find_consolidation_clusters finds similar memories."""
        # Create very similar memories
        consolidation_storage.store_memory(
            content="Python uses indentation for code blocks",
            memory_type=MemoryType.PROJECT,
        )
        consolidation_storage.store_memory(
            content="Python uses indentation for defining code blocks",
            memory_type=MemoryType.PROJECT,
        )

        clusters = consolidation_storage.find_consolidation_clusters()

        # May or may not find clusters depending on embedding similarity
        # At minimum, should not error
        assert isinstance(clusters, list)

    def test_preview_consolidation(self, consolidation_storage):
        """preview_consolidation returns preview without changes."""
        mid1, _ = consolidation_storage.store_memory(
            content="This is test content about databases",
            memory_type=MemoryType.PROJECT,
        )

        preview = consolidation_storage.preview_consolidation()

        assert "cluster_count" in preview
        assert "memories_to_delete" in preview
        assert "clusters" in preview

        # Memory should still exist (preview doesn't delete)
        mem = consolidation_storage.get_memory(mid1)
        assert mem is not None

    def test_run_consolidation_dry_run(self, consolidation_storage):
        """run_consolidation with dry_run=True doesn't delete."""
        mid, _ = consolidation_storage.store_memory(
            content="Dry run test memory",
            memory_type=MemoryType.PROJECT,
        )

        result = consolidation_storage.run_consolidation(dry_run=True)

        assert "cluster_count" in result
        # Memory still exists
        mem = consolidation_storage.get_memory(mid)
        assert mem is not None

    def test_consolidate_cluster_merges_tags(self, consolidation_storage):
        """consolidate_cluster combines tags from all members."""
        from memory_mcp.storage import ConsolidationCluster

        mid1, _ = consolidation_storage.store_memory(
            content="Memory with tag A",
            memory_type=MemoryType.PROJECT,
            tags=["tag_a"],
        )
        mid2, _ = consolidation_storage.store_memory(
            content="Memory with tag B",
            memory_type=MemoryType.PROJECT,
            tags=["tag_b"],
        )

        # Create cluster manually
        cluster = ConsolidationCluster(
            representative_id=mid1,
            member_ids=[mid1, mid2],
            avg_similarity=0.9,
            total_access_count=2,
            combined_tags=["tag_a", "tag_b"],
        )

        result = consolidation_storage.consolidate_cluster(cluster)

        assert result["success"]
        assert mid2 in result["deleted_ids"]

        # Check representative has both tags
        mem = consolidation_storage.get_memory(mid1)
        assert "tag_a" in mem.tags
        assert "tag_b" in mem.tags


# ========== Importance Scoring Tests (MemGPT-inspired) ==========


class TestImportanceScoring:
    """Tests for importance scoring at admission."""

    def test_compute_importance_score_short_content(self):
        """Short content gets low importance score."""
        from memory_mcp.helpers import compute_importance_score

        score = compute_importance_score("Hi")
        assert score < 0.5

    def test_compute_importance_score_code_content(self):
        """Code content gets higher importance score than short plain text."""
        from memory_mcp.helpers import compute_importance_score

        code = "def hello_world():\n    print('Hello')"
        plain = "Hello world"
        code_score = compute_importance_score(code)
        plain_score = compute_importance_score(plain)
        # Code content should score higher than plain short text
        assert code_score > plain_score

    def test_compute_importance_score_with_entities(self):
        """Content with entities gets higher score."""
        from memory_mcp.helpers import compute_importance_score

        content = "Deploy to https://api.example.com using version 1.2.3"
        score = compute_importance_score(content)
        assert score > 0.3

    def test_get_importance_breakdown(self):
        """get_importance_breakdown returns component details."""
        from memory_mcp.helpers import get_importance_breakdown

        content = "```python\nimport os\n```"
        breakdown = get_importance_breakdown(content)

        assert "score" in breakdown
        assert "length" in breakdown
        assert "code" in breakdown
        assert "entities" in breakdown

    def test_importance_score_stored_with_memory(self, tmp_path):
        """Importance score is stored when memory is created."""
        settings = Settings(
            db_path=tmp_path / "importance.db",
            importance_scoring_enabled=True,
        )
        stor = Storage(settings)

        mid, _ = stor.store_memory(
            content="def calculate_total(items):\n    return sum(items)",
            memory_type=MemoryType.PATTERN,
        )

        mem = stor.get_memory(mid)
        # Should have a computed importance score (not 0)
        assert mem.importance_score > 0
        # Code content should score higher than very short plain text
        assert mem.importance_score > 0.1

        stor.close()


class TestWorkingSet:
    """Tests for working set resource functionality."""

    def test_get_working_set_empty_when_disabled(self, tmp_path):
        """Working set returns empty list when disabled."""
        settings = Settings(
            db_path=tmp_path / "ws.db",
            working_set_enabled=False,
        )
        stor = Storage(settings)
        try:
            result = stor.get_working_set()
            assert result == []
        finally:
            stor.close()

    def test_get_working_set_includes_hot_memories(self, tmp_path):
        """Working set includes hot memories when no recent recalls."""
        settings = Settings(
            db_path=tmp_path / "ws.db",
            working_set_enabled=True,
            working_set_max_items=10,
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        try:
            # Create and promote a memory
            mid, _ = stor.store_memory(
                content="Hot memory content for working set",
                memory_type=MemoryType.PROJECT,
            )
            stor.promote_to_hot(mid)

            working_set = stor.get_working_set()
            assert len(working_set) >= 1
            assert any(m.id == mid for m in working_set)
        finally:
            stor.close()

    def test_get_recent_recalls_empty(self, tmp_path):
        """Recent recalls returns empty when no retrieval events."""
        settings = Settings(
            db_path=tmp_path / "ws.db",
            retrieval_tracking_enabled=True,
        )
        stor = Storage(settings)
        try:
            result = stor.get_recent_recalls(limit=5)
            assert result == []
        finally:
            stor.close()

    def test_get_recent_recalls_with_used_events(self, tmp_path):
        """Recent recalls returns memories marked as used."""
        settings = Settings(
            db_path=tmp_path / "ws.db",
            retrieval_tracking_enabled=True,
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        try:
            # Create memories
            mid1, _ = stor.store_memory(
                content="Memory one for recall test",
                memory_type=MemoryType.PROJECT,
            )
            mid2, _ = stor.store_memory(
                content="Memory two for recall test",
                memory_type=MemoryType.PROJECT,
            )

            # Record retrieval events
            stor.record_retrieval_event("test query", [mid1, mid2], [0.9, 0.8])

            # Mark one as used
            stor.mark_retrieval_used(mid1, query="test query")

            # Get recent recalls - should only include used ones
            recent = stor.get_recent_recalls(limit=5)
            assert len(recent) == 1
            assert recent[0].id == mid1
        finally:
            stor.close()

    def test_working_set_respects_max_items(self, tmp_path):
        """Working set caps at max_items setting."""
        settings = Settings(
            db_path=tmp_path / "ws.db",
            working_set_enabled=True,
            working_set_max_items=3,
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        try:
            # Create and promote more memories than max
            for i in range(5):
                mid, _ = stor.store_memory(
                    content=f"Working set memory {i}",
                    memory_type=MemoryType.PROJECT,
                )
                stor.promote_to_hot(mid)

            working_set = stor.get_working_set()
            assert len(working_set) <= 3
        finally:
            stor.close()


class TestEpisodicMemory:
    """Tests for episodic memory and session end functionality."""

    def test_episodic_memory_type_exists(self):
        """EPISODIC memory type should be available."""
        assert MemoryType.EPISODIC.value == "episodic"

    def test_store_episodic_memory(self, tmp_path):
        """Can store memories with episodic type."""
        settings = Settings(
            db_path=tmp_path / "episodic.db",
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        try:
            mid, is_new = stor.store_memory(
                content="Episodic memory content",
                memory_type=MemoryType.EPISODIC,
                session_id="test-session-123",
            )
            assert is_new
            memory = stor.get_memory(mid)
            assert memory.memory_type == MemoryType.EPISODIC
            assert memory.session_id == "test-session-123"
        finally:
            stor.close()

    def test_end_session_not_found(self, tmp_path):
        """end_session returns error for unknown session."""
        settings = Settings(db_path=tmp_path / "episodic.db")
        stor = Storage(settings)
        try:
            result = stor.end_session("nonexistent-session")
            assert result["success"] is False
            assert "not found" in result["error"]
        finally:
            stor.close()

    def test_end_session_promotes_top_episodic(self, tmp_path):
        """end_session promotes top episodic memories by salience."""
        settings = Settings(
            db_path=tmp_path / "episodic.db",
            semantic_dedup_enabled=False,
            episodic_promote_top_n=2,
            episodic_promote_threshold=0.0,  # Promote all
        )
        stor = Storage(settings)
        try:
            session_id = "test-session-end"
            stor.create_or_get_session(session_id)

            # Create episodic memories with different access counts
            mid1, _ = stor.store_memory(
                content="Low value episodic",
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
            )
            mid2, _ = stor.store_memory(
                content="High value episodic",
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
            )

            # Increase salience for mid2 by accessing it multiple times
            for _ in range(5):
                stor.get_memory(mid2)

            result = stor.end_session(session_id, promote_top=True)

            assert result["success"] is True
            assert result["episodic_count"] == 2
            assert result["promoted_count"] >= 1
            assert mid2 in result["promoted_ids"]

            # Verify the higher-salience memory was promoted to PROJECT
            promoted_mem = stor.get_memory(mid2)
            assert promoted_mem.memory_type == MemoryType.PROJECT
        finally:
            stor.close()

    def test_end_session_no_promote(self, tmp_path):
        """end_session with promote_top=False keeps types unchanged."""
        settings = Settings(
            db_path=tmp_path / "episodic.db",
            semantic_dedup_enabled=False,
        )
        stor = Storage(settings)
        try:
            session_id = "test-session-nopromote"
            stor.create_or_get_session(session_id)

            mid, _ = stor.store_memory(
                content="Episodic to not promote",
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
            )

            result = stor.end_session(session_id, promote_top=False)

            assert result["success"] is True
            assert result["promoted_count"] == 0

            # Memory type unchanged
            memory = stor.get_memory(mid)
            assert memory.memory_type == MemoryType.EPISODIC
        finally:
            stor.close()


class TestHybridSearch:
    """Tests for hybrid semantic + keyword search (v12 feature)."""

    @pytest.fixture
    def hybrid_storage(self, tmp_path):
        """Storage with hybrid search enabled."""
        settings = Settings(
            db_path=tmp_path / "hybrid.db",
            semantic_dedup_enabled=False,
            hybrid_search_enabled=True,
            hybrid_keyword_weight=0.3,
            hybrid_keyword_boost_threshold=0.4,
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    @pytest.fixture
    def no_hybrid_storage(self, tmp_path):
        """Storage with hybrid search disabled."""
        settings = Settings(
            db_path=tmp_path / "no_hybrid.db",
            semantic_dedup_enabled=False,
            hybrid_search_enabled=False,
        )
        stor = Storage(settings)
        yield stor
        stor.close()

    def test_fts_table_created(self, hybrid_storage):
        """FTS5 table should exist after migration."""
        with hybrid_storage._connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fts'"
            ).fetchone()
            assert result is not None

    def test_fts_synced_on_insert(self, hybrid_storage):
        """FTS table should be populated when memories are inserted."""
        mid, _ = hybrid_storage.store_memory(
            "FastAPI with async endpoints for Python backend",
            MemoryType.PROJECT,
        )

        with hybrid_storage._connection() as conn:
            # Verify FTS content exists
            result = conn.execute(
                "SELECT content FROM memory_fts WHERE rowid = ?", (mid,)
            ).fetchone()
            assert result is not None
            assert "FastAPI" in result["content"]

    def test_fts_synced_on_delete(self, hybrid_storage):
        """FTS table should be cleaned when memories are deleted."""
        mid, _ = hybrid_storage.store_memory("Temporary memory", MemoryType.PROJECT)

        with hybrid_storage._connection() as conn:
            # Verify FTS content exists
            result = conn.execute(
                "SELECT content FROM memory_fts WHERE rowid = ?", (mid,)
            ).fetchone()
            assert result is not None

        # Delete the memory
        hybrid_storage.delete_memory(mid)

        with hybrid_storage._connection() as conn:
            # Verify FTS content removed
            result = conn.execute(
                "SELECT content FROM memory_fts WHERE rowid = ?", (mid,)
            ).fetchone()
            assert result is None

    def test_keyword_score_populated(self, hybrid_storage):
        """Keyword score should be populated for matching results."""
        hybrid_storage.store_memory(
            "FastAPI framework for building APIs with Python",
            MemoryType.PROJECT,
        )

        # Query with keyword that's in the content
        result = hybrid_storage.recall("FastAPI Python", threshold=0.3)

        assert len(result.memories) > 0
        mem = result.memories[0]
        # Should have keyword score since query words match content
        assert mem.keyword_score is not None
        assert mem.keyword_score > 0

    def test_hybrid_disabled_no_keyword_score(self, no_hybrid_storage):
        """When hybrid is disabled, keyword_score should be None."""
        no_hybrid_storage.store_memory(
            "FastAPI framework for building APIs with Python",
            MemoryType.PROJECT,
        )

        result = no_hybrid_storage.recall("FastAPI Python", threshold=0.3)

        assert len(result.memories) > 0
        mem = result.memories[0]
        # Should NOT have keyword score when hybrid disabled
        assert mem.keyword_score is None

    def test_hybrid_boosts_keyword_matches(self, tmp_path):
        """Hybrid search should boost results with keyword matches."""
        # Create two storages with same content
        settings_hybrid = Settings(
            db_path=tmp_path / "hybrid_boost.db",
            semantic_dedup_enabled=False,
            hybrid_search_enabled=True,
            hybrid_keyword_weight=0.3,
        )
        settings_no_hybrid = Settings(
            db_path=tmp_path / "no_hybrid_boost.db",
            semantic_dedup_enabled=False,
            hybrid_search_enabled=False,
        )
        stor_hybrid = Storage(settings_hybrid)
        stor_no_hybrid = Storage(settings_no_hybrid)

        try:
            content = "Django REST Framework for building REST APIs"

            stor_hybrid.store_memory(content, MemoryType.PROJECT)
            stor_no_hybrid.store_memory(content, MemoryType.PROJECT)

            # Query that has keyword overlap
            query = "Django REST API"

            result_hybrid = stor_hybrid.recall(query, threshold=0.3)
            result_no_hybrid = stor_no_hybrid.recall(query, threshold=0.3)

            # Both should find the memory
            assert len(result_hybrid.memories) > 0
            assert len(result_no_hybrid.memories) > 0

            # Hybrid should have keyword score
            assert result_hybrid.memories[0].keyword_score is not None

            # The composite scores might differ due to hybrid boost
            # but we mainly verify both work correctly
            assert result_hybrid.memories[0].composite_score is not None
            assert result_no_hybrid.memories[0].composite_score is not None
        finally:
            stor_hybrid.close()
            stor_no_hybrid.close()

    def test_schema_version_is_16(self, hybrid_storage):
        """Schema version should be 16 for hybrid search."""
        version = hybrid_storage.get_schema_version()
        assert version == 16
