"""Tests for server module - tools and resources."""

import pytest

from memory_mcp.config import Settings
from memory_mcp.storage import MemoryType, Storage


@pytest.fixture
def storage(tmp_path):
    """Create a storage instance with temp database."""
    settings = Settings(db_path=tmp_path / "test.db", promotion_threshold=3)
    stor = Storage(settings)
    yield stor
    stor.close()


# ========== Promotion Suggestions Tests ==========


class TestPromotionSuggestions:
    """Tests for promotion suggestions in recall responses."""

    def test_no_suggestions_when_all_hot(self, storage):
        """No suggestions when all recalled memories are already hot."""
        from memory_mcp.server import get_promotion_suggestions

        # Create and promote memories
        id1, _ = storage.store_memory("Hot memory 1", MemoryType.PROJECT)
        storage.promote_to_hot(id1)
        memory = storage.get_memory(id1)

        suggestions = get_promotion_suggestions([memory])
        assert suggestions == []

    def test_no_suggestions_when_low_access(self, storage):
        """No suggestions when access count is below threshold."""
        from memory_mcp.server import get_promotion_suggestions

        # Create memory with low access count
        id1, _ = storage.store_memory("Cold memory", MemoryType.PROJECT)
        memory = storage.get_memory(id1)

        suggestions = get_promotion_suggestions([memory])
        assert suggestions == []

    def test_suggests_high_access_cold_memory(self, storage):
        """Suggests promoting cold memories with high access count."""
        from memory_mcp.server import get_promotion_suggestions

        # Create memory and access it multiple times
        id1, _ = storage.store_memory("Frequently accessed", MemoryType.PROJECT)
        for _ in range(5):
            storage.update_access(id1)
        memory = storage.get_memory(id1)

        suggestions = get_promotion_suggestions([memory])
        assert len(suggestions) == 1
        assert suggestions[0]["memory_id"] == id1
        assert "access_count" in suggestions[0]
        assert "reason" in suggestions[0]

    def test_max_suggestions_limit(self, storage):
        """Respects max_suggestions limit."""
        from memory_mcp.server import get_promotion_suggestions

        # Create multiple high-access memories
        memories = []
        for i in range(5):
            mid, _ = storage.store_memory(f"Memory {i}", MemoryType.PROJECT)
            for _ in range(10):
                storage.update_access(mid)
            memories.append(storage.get_memory(mid))

        suggestions = get_promotion_suggestions(memories, max_suggestions=2)
        assert len(suggestions) == 2


# ========== Hot Cache Effectiveness Tests ==========


class TestHotCacheEffectiveness:
    """Tests for hot cache effectiveness metrics."""

    def test_empty_hot_cache_effectiveness(self, storage):
        """Effectiveness metrics with empty hot cache."""
        hot_memories = storage.get_hot_memories()
        metrics = storage.get_hot_cache_metrics()

        total_accesses = sum(m.access_count for m in hot_memories)
        total_reads = metrics.hits + metrics.misses
        hit_rate = (metrics.hits / total_reads * 100) if total_reads > 0 else 0.0

        assert total_accesses == 0
        assert hit_rate == 0.0

    def test_effectiveness_with_hot_memories(self, storage):
        """Effectiveness metrics with hot memories."""
        # Create and promote memories with varying access counts
        id1, _ = storage.store_memory("Memory 1", MemoryType.PROJECT)
        id2, _ = storage.store_memory("Memory 2", MemoryType.PROJECT)

        for _ in range(5):
            storage.update_access(id1)
        for _ in range(2):
            storage.update_access(id2)

        storage.promote_to_hot(id1)
        storage.promote_to_hot(id2)

        hot_memories = storage.get_hot_memories()
        total_accesses = sum(m.access_count for m in hot_memories)

        # 5 + 2 accesses
        assert total_accesses == 7

    def test_hit_rate_calculation(self, storage):
        """Hit rate calculation."""
        metrics = storage.get_hot_cache_metrics()

        # Simulate some hits and misses
        storage.record_hot_cache_hit()
        storage.record_hot_cache_hit()
        storage.record_hot_cache_hit()
        storage.record_hot_cache_miss()

        metrics = storage.get_hot_cache_metrics()
        total_reads = metrics.hits + metrics.misses
        hit_rate = (metrics.hits / total_reads * 100) if total_reads > 0 else 0.0

        assert total_reads == 4
        assert hit_rate == 75.0

    def test_most_least_accessed_identification(self, storage):
        """Identifies most and least accessed hot memories."""
        # Create memories with different access counts
        id_low, _ = storage.store_memory("Low access", MemoryType.PROJECT)
        id_high, _ = storage.store_memory("High access", MemoryType.PROJECT)

        for _ in range(10):
            storage.update_access(id_high)

        storage.promote_to_hot(id_low)
        storage.promote_to_hot(id_high)

        hot_memories = storage.get_hot_memories()
        most_accessed = max(hot_memories, key=lambda m: m.access_count)
        unpinned = [m for m in hot_memories if not m.is_pinned]
        least_accessed = min(unpinned, key=lambda m: m.access_count)

        assert most_accessed.id == id_high
        assert least_accessed.id == id_low

    def test_least_accessed_excludes_pinned(self, storage):
        """Least accessed excludes pinned memories."""
        # Create memories
        id_low_pinned, _ = storage.store_memory("Low pinned", MemoryType.PROJECT)
        id_medium, _ = storage.store_memory("Medium access", MemoryType.PROJECT)

        for _ in range(3):
            storage.update_access(id_medium)

        storage.promote_to_hot(id_low_pinned, pin=True)
        storage.promote_to_hot(id_medium)

        hot_memories = storage.get_hot_memories()
        unpinned = [m for m in hot_memories if not m.is_pinned]
        least_accessed = min(unpinned, key=lambda m: m.access_count)

        # Should be medium, not the pinned low one
        assert least_accessed.id == id_medium
