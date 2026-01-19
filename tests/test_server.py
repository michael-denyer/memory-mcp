"""Tests for server module - tools and resources."""

import pytest

from memory_mcp.config import Settings
from memory_mcp.storage import MemoryType, Storage


@pytest.fixture
def storage(tmp_path):
    """Create a storage instance with temp database.

    Semantic dedup is disabled to keep test content independent.
    """
    settings = Settings(
        db_path=tmp_path / "test.db",
        promotion_threshold=3,
        semantic_dedup_enabled=False,
    )
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


# ========== Trust Management Tools Tests ==========


class TestTrustManagementTools:
    """Tests for trust strengthening/weakening via storage layer.

    Note: MCP tools are decorated with @mcp.tool and can't be called directly.
    These tests verify the underlying storage methods that the tools use.
    """

    def test_strengthen_trust_increases_score(self, storage):
        """strengthen_trust() should increase the trust score."""
        from memory_mcp.storage import MemorySource

        # Use mined memory which starts at 0.7
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)
        original = storage.get_memory(mid)

        new_trust = storage.strengthen_trust(mid, boost=0.15)

        assert abs(new_trust - (original.trust_score + 0.15)) < 0.001
        updated = storage.get_memory(mid)
        assert abs(updated.trust_score - new_trust) < 0.001

    def test_strengthen_trust_caps_at_one(self, storage):
        """strengthen_trust() should cap trust at 1.0."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)

        # Boost multiple times (manual starts at 1.0)
        for _ in range(15):
            result = storage.strengthen_trust(mid, boost=0.1)

        assert result == 1.0

    def test_strengthen_trust_not_found(self, storage):
        """strengthen_trust() should return None for nonexistent memory."""
        result = storage.strengthen_trust(99999, boost=0.1)
        assert result is None

    def test_weaken_trust_decreases_score(self, storage):
        """weaken_trust() should decrease the trust score."""
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT)
        original = storage.get_memory(mid)

        new_trust = storage.weaken_trust(mid, penalty=0.2)

        assert abs(new_trust - (original.trust_score - 0.2)) < 0.001
        updated = storage.get_memory(mid)
        assert abs(updated.trust_score - new_trust) < 0.001

    def test_weaken_trust_floors_at_zero(self, storage):
        """weaken_trust() should floor trust at 0.0."""
        from memory_mcp.storage import MemorySource

        # Use mined memory which starts at 0.7
        mid, _ = storage.store_memory("Test memory", MemoryType.PROJECT, source=MemorySource.MINED)

        # Weaken with a large penalty to definitely hit zero
        result = storage.weaken_trust(mid, penalty=1.0)

        assert result == 0.0

    def test_weaken_trust_not_found(self, storage):
        """weaken_trust() should return None for nonexistent memory."""
        result = storage.weaken_trust(99999, penalty=0.1)
        assert result is None

    def test_trust_affects_recall_ranking(self, storage):
        """Low trust memories should have lower decayed trust in recall results."""
        from memory_mcp.storage import MemorySource

        # Create two similar memories
        mid1, _ = storage.store_memory("Database configuration settings", MemoryType.PROJECT)
        mid2, _ = storage.store_memory(
            "Database configuration options", MemoryType.PROJECT, source=MemorySource.MINED
        )

        # Weaken trust on second memory significantly
        storage.weaken_trust(mid2, penalty=0.5)

        # Recall should return both memories
        result = storage.recall("database configuration", threshold=0.3)
        assert len(result.memories) == 2

        # Find each memory in results
        mem1 = next(m for m in result.memories if m.id == mid1)
        mem2 = next(m for m in result.memories if m.id == mid2)

        # The manual one (mid1) should have higher trust than weakened one (mid2)
        assert mem1.trust_score > mem2.trust_score
        assert mem1.trust_score == 1.0  # Manual memory, never weakened
        assert mem2.trust_score < 0.3  # Started at 0.7, weakened by 0.5


# ========== Auto-Bootstrap Tests ==========


class TestAutoBootstrap:
    """Tests for auto-bootstrap functionality."""

    def test_try_auto_bootstrap_with_files(self, tmp_path, monkeypatch):
        """Auto-bootstrap creates memories when documentation files exist."""
        from memory_mcp import server
        from memory_mcp.config import Settings

        # Set up temp directory with README
        readme = tmp_path / "README.md"
        readme.write_text("# Test Project\n\n- Feature one\n- Feature two\n")

        # Create fresh storage for this test
        settings = Settings(db_path=tmp_path / "test.db")
        test_storage = Storage(settings)

        # Monkeypatch the server's storage and cwd
        monkeypatch.setattr(server, "storage", test_storage)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server._auto_bootstrap_attempted.clear()

        # Trigger auto-bootstrap
        result = server._try_auto_bootstrap()

        assert result is True

        # Verify memories were created
        hot_memories = test_storage.get_hot_memories()
        assert len(hot_memories) >= 1

        # Verify tag was applied
        assert any("auto-bootstrap" in m.tags for m in hot_memories)

        test_storage.close()

    def test_try_auto_bootstrap_no_files(self, tmp_path, monkeypatch):
        """Auto-bootstrap returns False when no documentation files exist."""
        from memory_mcp import server
        from memory_mcp.config import Settings

        # Create fresh storage for this test
        settings = Settings(db_path=tmp_path / "test.db")
        test_storage = Storage(settings)

        # Monkeypatch the server's storage and cwd (empty directory)
        monkeypatch.setattr(server, "storage", test_storage)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server._auto_bootstrap_attempted.clear()

        # Trigger auto-bootstrap
        result = server._try_auto_bootstrap()

        assert result is False

        # Verify no memories were created
        hot_memories = test_storage.get_hot_memories()
        assert len(hot_memories) == 0

        test_storage.close()

    def test_try_auto_bootstrap_only_once_per_directory(self, tmp_path, monkeypatch):
        """Auto-bootstrap only runs once per directory per session."""
        from memory_mcp import server
        from memory_mcp.config import Settings

        # Set up temp directory with README
        readme = tmp_path / "README.md"
        readme.write_text("# Test Project\n\n- New content\n")

        # Create fresh storage for this test
        settings = Settings(db_path=tmp_path / "test.db")
        test_storage = Storage(settings)

        # Monkeypatch the server's storage and cwd
        monkeypatch.setattr(server, "storage", test_storage)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server._auto_bootstrap_attempted.clear()

        # First call should bootstrap
        result1 = server._try_auto_bootstrap()
        assert result1 is True

        # Second call should return False (already attempted)
        result2 = server._try_auto_bootstrap()
        assert result2 is False

        test_storage.close()

    def test_hot_cache_resource_triggers_auto_bootstrap(self, tmp_path, monkeypatch):
        """Hot cache resource triggers auto-bootstrap when empty."""
        from memory_mcp import server
        from memory_mcp.config import Settings

        # Set up temp directory with README
        readme = tmp_path / "README.md"
        readme.write_text("# Auto Bootstrap Test\n\n- Content line\n")

        # Create fresh storage for this test
        settings = Settings(db_path=tmp_path / "test.db")
        test_storage = Storage(settings)

        # Monkeypatch the server's storage and cwd
        monkeypatch.setattr(server, "storage", test_storage)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server._auto_bootstrap_attempted.clear()

        # Call the underlying function (not the FastMCP FunctionResource wrapper)
        # The actual function is stored in hot_cache_resource.fn
        content = server.hot_cache_resource.fn()

        # Should have bootstrapped and returned content
        assert "[MEMORY: Hot Cache" in content
        assert "empty" not in content.lower()

        # Verify memories exist
        hot_memories = test_storage.get_hot_memories()
        assert len(hot_memories) >= 1

        test_storage.close()
