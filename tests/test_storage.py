"""Tests for storage module."""

import pytest

from memory_mcp.config import Settings
from memory_mcp.storage import (
    HotCacheMetrics,
    MemorySource,
    MemoryType,
    Storage,
    ValidationError,
)


@pytest.fixture
def storage(tmp_path):
    """Create a storage instance with temp database."""
    settings = Settings(db_path=tmp_path / "test.db")
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
        settings = Settings(db_path=tmp_path / "test.db", hot_cache_max_items=2)
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
