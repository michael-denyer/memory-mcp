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

        # Create fresh storage for this test (enable auto_bootstrap)
        settings = Settings(db_path=tmp_path / "test.db", auto_bootstrap=True)
        test_storage = Storage(settings)

        # Monkeypatch the server.app's storage, settings, and cwd
        monkeypatch.setattr(server.app, "storage", test_storage)
        monkeypatch.setattr(server.app, "settings", settings)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server.app._auto_bootstrap_attempted.clear()

        # Trigger auto-bootstrap
        result = server.app._try_auto_bootstrap()

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

        # Create fresh storage for this test (enable auto_bootstrap)
        settings = Settings(db_path=tmp_path / "test.db", auto_bootstrap=True)
        test_storage = Storage(settings)

        # Monkeypatch the server.app's storage, settings, and cwd (empty directory)
        monkeypatch.setattr(server.app, "storage", test_storage)
        monkeypatch.setattr(server.app, "settings", settings)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server.app._auto_bootstrap_attempted.clear()

        # Trigger auto-bootstrap
        result = server.app._try_auto_bootstrap()

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

        # Create fresh storage for this test (enable auto_bootstrap)
        settings = Settings(db_path=tmp_path / "test.db", auto_bootstrap=True)
        test_storage = Storage(settings)

        # Monkeypatch the server's storage, settings, and cwd
        monkeypatch.setattr(server.app, "storage", test_storage)
        monkeypatch.setattr(server.app, "settings", settings)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server.app._auto_bootstrap_attempted.clear()

        # First call should bootstrap
        result1 = server.app._try_auto_bootstrap()
        assert result1 is True

        # Second call should return False (already attempted)
        result2 = server.app._try_auto_bootstrap()
        assert result2 is False

        test_storage.close()

    def test_hot_cache_resource_triggers_auto_bootstrap(self, tmp_path, monkeypatch):
        """Hot cache resource triggers auto-bootstrap when empty."""
        from memory_mcp import server
        from memory_mcp.config import Settings

        # Set up temp directory with README
        readme = tmp_path / "README.md"
        readme.write_text("# Auto Bootstrap Test\n\n- Content line\n")

        # Create fresh storage for this test (enable auto_bootstrap)
        settings = Settings(db_path=tmp_path / "test.db", auto_bootstrap=True)
        test_storage = Storage(settings)

        # Monkeypatch the server's storage, settings, and cwd
        monkeypatch.setattr(server.app, "storage", test_storage)
        monkeypatch.setattr(server.app, "settings", settings)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server.app._auto_bootstrap_attempted.clear()

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

    def test_try_auto_bootstrap_disabled_by_default(self, tmp_path, monkeypatch):
        """Auto-bootstrap returns False when disabled (default behavior)."""
        from memory_mcp import server
        from memory_mcp.config import Settings

        # Set up temp directory with README
        readme = tmp_path / "README.md"
        readme.write_text("# Test Project\n\n- Feature one\n- Feature two\n")

        # Create fresh storage with default settings (auto_bootstrap=False)
        settings = Settings(db_path=tmp_path / "test.db")
        assert settings.auto_bootstrap is False  # Verify default is False
        test_storage = Storage(settings)

        # Monkeypatch the server.app's storage, settings, and cwd
        monkeypatch.setattr(server.app, "storage", test_storage)
        monkeypatch.setattr(server.app, "settings", settings)
        monkeypatch.chdir(tmp_path)

        # Clear the bootstrap tracking set
        server.app._auto_bootstrap_attempted.clear()

        # Trigger auto-bootstrap - should return False since disabled
        result = server.app._try_auto_bootstrap()

        assert result is False

        # Verify no memories were created
        hot_memories = test_storage.get_hot_memories()
        assert len(hot_memories) == 0

        test_storage.close()


# ========== Metrics and Logging Tests ==========


class TestMetrics:
    """Tests for observability metrics tracking."""

    def test_metrics_snapshot_has_uptime(self):
        """Metrics snapshot includes uptime."""
        from memory_mcp.logging import metrics

        metrics.reset()
        snapshot = metrics.snapshot()
        assert "uptime_seconds" in snapshot
        assert snapshot["uptime_seconds"] >= 0

    def test_metrics_increment_counter(self):
        """Incrementing a counter tracks the value."""
        from memory_mcp.logging import metrics

        metrics.reset()
        metrics.increment("test.counter")
        metrics.increment("test.counter")
        assert metrics.get_counter("test.counter") == 2

    def test_metrics_increment_by_value(self):
        """Incrementing by a specific value works."""
        from memory_mcp.logging import metrics

        metrics.reset()
        metrics.increment("test.counter", 5)
        assert metrics.get_counter("test.counter") == 5

    def test_metrics_set_gauge(self):
        """Setting a gauge stores the value."""
        from memory_mcp.logging import metrics

        metrics.reset()
        metrics.set_gauge("test.gauge", 42.5)
        assert metrics.get_gauge("test.gauge") == 42.5

    def test_metrics_snapshot_includes_all(self):
        """Snapshot includes all counters and gauges."""
        from memory_mcp.logging import metrics

        metrics.reset()
        metrics.increment("recall.total", 10)
        metrics.set_gauge("hot_cache.size", 5.0)

        snapshot = metrics.snapshot()
        assert snapshot["counters"]["recall.total"] == 10
        assert snapshot["gauges"]["hot_cache.size"] == 5.0

    def test_record_recall_increments_counters(self):
        """record_recall() updates appropriate counters."""
        from memory_mcp.logging import metrics, record_recall

        metrics.reset()
        record_recall(query_length=20, results_count=3, gated_count=1, hot_hit=True, threshold=0.7)

        assert metrics.get_counter("recall.total") == 1
        assert metrics.get_counter("recall.results_returned") == 3
        assert metrics.get_counter("recall.results_gated") == 1
        assert metrics.get_counter("recall.hot_hits") == 1

    def test_record_recall_empty_result(self):
        """record_recall() tracks empty results."""
        from memory_mcp.logging import metrics, record_recall

        metrics.reset()
        record_recall(query_length=20, results_count=0, gated_count=5, hot_hit=False, threshold=0.7)

        assert metrics.get_counter("recall.empty") == 1
        assert metrics.get_counter("recall.hot_hits") == 0

    def test_record_store_by_type(self):
        """record_store() tracks stores by type."""
        from memory_mcp.logging import metrics, record_store

        metrics.reset()
        record_store(memory_type="project", merged=False, contradictions=0)
        record_store(memory_type="pattern", merged=True, contradictions=1)

        assert metrics.get_counter("store.total") == 2
        assert metrics.get_counter("store.type.project") == 1
        assert metrics.get_counter("store.type.pattern") == 1
        assert metrics.get_counter("store.merged") == 1
        assert metrics.get_counter("store.contradictions_found") == 1

    def test_record_mining_counters(self):
        """record_mining() updates mining counters."""
        from memory_mcp.logging import metrics, record_mining

        metrics.reset()
        record_mining(patterns_found=10, patterns_new=7, patterns_updated=3)

        assert metrics.get_counter("mining.runs") == 1
        assert metrics.get_counter("mining.patterns_found") == 10
        assert metrics.get_counter("mining.patterns_new") == 7
        assert metrics.get_counter("mining.patterns_updated") == 3

    def test_record_hot_cache_change(self):
        """record_hot_cache_change() tracks cache mutations."""
        from memory_mcp.logging import metrics, record_hot_cache_change

        metrics.reset()
        record_hot_cache_change(promoted=True)
        record_hot_cache_change(demoted=True)
        record_hot_cache_change(evicted=True)

        assert metrics.get_counter("hot_cache.promotions") == 1
        assert metrics.get_counter("hot_cache.demotions") == 1
        assert metrics.get_counter("hot_cache.evictions") == 1

    def test_update_hot_cache_stats_gauges(self):
        """update_hot_cache_stats() sets gauge values."""
        from memory_mcp.logging import metrics, update_hot_cache_stats

        metrics.reset()
        update_hot_cache_stats(size=5, max_size=20, pinned=2)

        assert metrics.get_gauge("hot_cache.size") == 5.0
        assert metrics.get_gauge("hot_cache.max_size") == 20.0
        assert metrics.get_gauge("hot_cache.pinned") == 2.0
        assert metrics.get_gauge("hot_cache.utilization") == 0.25


class TestLoggingConfiguration:
    """Tests for logging configuration."""

    def test_configure_logging_default(self):
        """Default logging configuration works."""
        from memory_mcp.logging import configure_logging

        # Should not raise
        configure_logging()

    def test_configure_logging_json_format(self):
        """JSON logging format can be configured."""
        from memory_mcp.logging import configure_logging

        # Should not raise
        configure_logging(level="DEBUG", log_format="json")

        # Reset to default
        configure_logging()

    def test_configure_logging_levels(self):
        """Various log levels can be configured."""
        from memory_mcp.logging import configure_logging

        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            configure_logging(level=level)

        # Reset to default
        configure_logging()

    def test_json_serializer_handles_datetime(self):
        """JSON serializer handles datetime objects."""
        import json
        from datetime import datetime, timezone

        from memory_mcp.logging import json_serializer

        record = {
            "time": datetime.now(timezone.utc),
            "level": type("Level", (), {"name": "INFO"})(),
            "extra": {"name": "test"},
            "name": "test_module",
            "function": "test_func",
            "message": "Test message",
        }
        result = json_serializer(record)
        parsed = json.loads(result)

        assert "timestamp" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"


# ========== Empty Content Validation Tests ==========


class TestEmptyContentValidation:
    """Tests for empty content validation in server tools.

    Tests call the underlying functions directly since FastMCP wraps
    decorated functions in FunctionTool objects.
    """

    def test_remember_empty_content_returns_error(self, storage, monkeypatch):
        """remember with empty content returns error, not exception."""
        # Import the module-level function (not the FastMCP-wrapped tool)
        import memory_mcp.server as server_module

        monkeypatch.setattr(server_module, "storage", storage)

        # Get the actual function from the tool wrapper
        remember_fn = server_module.remember.fn
        result = remember_fn(content="")
        assert result.get("success") is False
        assert "empty" in result.get("error", "").lower()

    def test_remember_whitespace_content_returns_error(self, storage, monkeypatch):
        """remember with whitespace-only content returns error."""
        import memory_mcp.server as server_module

        monkeypatch.setattr(server_module, "storage", storage)

        remember_fn = server_module.remember.fn
        result = remember_fn(content="   \n\t  ")
        assert result.get("success") is False
        assert "empty" in result.get("error", "").lower()

    def test_log_output_empty_content_returns_error(self, storage, monkeypatch):
        """log_output with empty content returns error, not exception."""
        import memory_mcp.server as server_module

        monkeypatch.setattr(server_module, "storage", storage)
        monkeypatch.setattr(server_module, "settings", storage.settings)

        log_output_fn = server_module.log_output.fn
        result = log_output_fn(content="")
        assert result.get("success") is False
        assert "empty" in result.get("error", "").lower()

    def test_log_output_whitespace_content_returns_error(self, storage, monkeypatch):
        """log_output with whitespace-only content returns error."""
        import memory_mcp.server as server_module

        monkeypatch.setattr(server_module, "storage", storage)
        monkeypatch.setattr(server_module, "settings", storage.settings)

        log_output_fn = server_module.log_output.fn
        result = log_output_fn(content="  \t\n  ")
        assert result.get("success") is False
        assert "empty" in result.get("error", "").lower()


# ========== Context Shaping Tests ==========


class TestContextShaping:
    """Tests for LLM-friendly context shaping in recall responses."""

    def test_format_age_just_now(self):
        """format_age returns 'just now' for very recent times."""
        from datetime import datetime, timedelta, timezone

        from memory_mcp.helpers import format_age

        now = datetime.now(timezone.utc)
        recent = now - timedelta(seconds=30)
        assert format_age(recent) == "just now"

    def test_format_age_hours(self):
        """format_age returns hours for recent times."""
        from datetime import datetime, timedelta, timezone

        from memory_mcp.helpers import format_age

        now = datetime.now(timezone.utc)
        hours_ago = now - timedelta(hours=5)
        assert format_age(hours_ago) == "5 hours"

    def test_format_age_days(self):
        """format_age returns days for multi-day times."""
        from datetime import datetime, timedelta, timezone

        from memory_mcp.helpers import format_age

        now = datetime.now(timezone.utc)
        days_ago = now - timedelta(days=3)
        assert format_age(days_ago) == "3 days"

    def test_format_age_weeks(self):
        """format_age returns weeks for multi-week times."""
        from datetime import datetime, timedelta, timezone

        from memory_mcp.helpers import format_age

        now = datetime.now(timezone.utc)
        weeks_ago = now - timedelta(days=14)
        assert format_age(weeks_ago) == "2 weeks"

    def test_format_age_months(self):
        """format_age returns months for multi-month times."""
        from datetime import datetime, timedelta, timezone

        from memory_mcp.helpers import format_age

        now = datetime.now(timezone.utc)
        months_ago = now - timedelta(days=60)
        assert format_age(months_ago) == "2 months"

    def test_format_age_years(self):
        """format_age returns years for multi-year times."""
        from datetime import datetime, timedelta, timezone

        from memory_mcp.helpers import format_age

        now = datetime.now(timezone.utc)
        years_ago = now - timedelta(days=400)
        assert format_age(years_ago) == "1 year"

    def test_bayesian_helpfulness_cold_start(self):
        """Bayesian helpfulness gives benefit of doubt for cold start."""
        from memory_mcp.helpers import get_bayesian_helpfulness

        # 0 used, 0 retrieved → 0.25 (benefit of doubt)
        rate = get_bayesian_helpfulness(0, 0)
        assert rate == 0.25

    def test_bayesian_helpfulness_low_utility(self):
        """Bayesian helpfulness detects low utility memories."""
        from memory_mcp.helpers import get_bayesian_helpfulness

        # 0 used, 5 retrieved → (0+1)/(5+1+3) = 1/9 ≈ 0.111 (evidence of low utility)
        rate = get_bayesian_helpfulness(0, 5)
        assert abs(rate - 0.111) < 0.01

    def test_bayesian_helpfulness_decent_signal(self):
        """Bayesian helpfulness recognizes decent usage signal."""
        from memory_mcp.helpers import get_bayesian_helpfulness

        # 2 used, 5 retrieved → 0.33 (decent signal)
        rate = get_bayesian_helpfulness(2, 5)
        assert abs(rate - 0.333) < 0.01

    def test_bayesian_helpfulness_strong_signal(self):
        """Bayesian helpfulness recognizes strong usage signal."""
        from memory_mcp.helpers import get_bayesian_helpfulness

        # 5 used, 10 retrieved → 0.43 (strong signal)
        rate = get_bayesian_helpfulness(5, 10)
        assert abs(rate - 0.429) < 0.01

    def test_summarize_content_short_text(self):
        """summarize_content returns short text unchanged."""
        from memory_mcp.helpers import summarize_content

        content = "This is a short fact about the project."
        assert summarize_content(content) == content

    def test_summarize_content_long_text_truncates(self):
        """summarize_content truncates long text with ellipsis."""
        from memory_mcp.helpers import summarize_content

        content = "x" * 200
        result = summarize_content(content, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_summarize_content_multiline_takes_first(self):
        """summarize_content takes first line of multiline text."""
        from memory_mcp.helpers import summarize_content

        content = "First line is important.\nSecond line is detail.\nThird is more."
        assert summarize_content(content) == "First line is important."

    def test_summarize_content_code_block_extracts_code(self):
        """summarize_content extracts meaningful content from code blocks."""
        from memory_mcp.helpers import summarize_content

        content = "```python\ndef hello_world():\n    print('hello')\n```"
        result = summarize_content(content)
        assert "def hello_world" in result

    def test_get_similarity_confidence_high(self):
        """get_similarity_confidence returns 'high' for high scores."""
        from memory_mcp.server import get_similarity_confidence

        assert get_similarity_confidence(0.90) == "high"
        assert get_similarity_confidence(0.95) == "high"

    def test_get_similarity_confidence_medium(self):
        """get_similarity_confidence returns 'medium' for medium scores."""
        from memory_mcp.server import get_similarity_confidence

        assert get_similarity_confidence(0.75) == "medium"
        assert get_similarity_confidence(0.80) == "medium"

    def test_get_similarity_confidence_low(self):
        """get_similarity_confidence returns 'low' for low scores."""
        from memory_mcp.server import get_similarity_confidence

        assert get_similarity_confidence(0.50) == "low"
        assert get_similarity_confidence(0.65) == "low"

    def test_get_similarity_confidence_none(self):
        """get_similarity_confidence returns 'unknown' for None."""
        from memory_mcp.server import get_similarity_confidence

        assert get_similarity_confidence(None) == "unknown"

    def test_format_memories_for_llm_empty(self):
        """format_memories_for_llm handles empty list."""
        from memory_mcp.server import format_memories_for_llm

        formatted, summary = format_memories_for_llm([])
        assert formatted == []
        assert summary == "No matching memories found"

    def test_format_memories_for_llm_basic(self, storage):
        """format_memories_for_llm creates formatted memories with annotations."""
        from memory_mcp.server import format_memories_for_llm

        # Store a memory
        mem_id, _ = storage.store_memory(
            "This project uses SQLite for persistence.",
            MemoryType.PROJECT,
            tags=["database", "architecture"],
        )
        memory = storage.get_memory(mem_id)
        memory.similarity = 0.80  # Simulate recall score (below high_confidence_threshold of 0.85)

        formatted, summary = format_memories_for_llm([memory])

        assert len(formatted) == 1
        assert formatted[0].memory_type == "project"
        assert "database" in formatted[0].tags
        assert formatted[0].age == "just now"
        assert formatted[0].confidence == "medium"  # 0.80 is medium (below 0.85 high threshold)
        assert formatted[0].source_hint == "cold storage"  # Not promoted

    def test_format_memories_for_llm_hot_cache_hint(self, storage):
        """format_memories_for_llm shows hot cache hint for promoted memories."""
        from memory_mcp.server import format_memories_for_llm

        mem_id, _ = storage.store_memory("Hot memory content", MemoryType.PROJECT)
        storage.promote_to_hot(mem_id)
        memory = storage.get_memory(mem_id)
        memory.similarity = 0.90

        formatted, summary = format_memories_for_llm([memory])

        assert formatted[0].source_hint == "hot cache"
        assert formatted[0].confidence == "high"  # 0.90 is high

    def test_format_memories_for_llm_summary(self, storage):
        """format_memories_for_llm generates context summary."""
        from memory_mcp.server import format_memories_for_llm

        # Store different memory types
        id1, _ = storage.store_memory("Project fact 1", MemoryType.PROJECT)
        id2, _ = storage.store_memory("Pattern code 1", MemoryType.PATTERN)
        id3, _ = storage.store_memory("Pattern code 2", MemoryType.PATTERN)

        memories = [storage.get_memory(id1), storage.get_memory(id2), storage.get_memory(id3)]
        for m in memories:
            m.similarity = 0.80

        formatted, summary = format_memories_for_llm(memories)

        assert "Found 3 memories" in summary
        assert "1 project" in summary
        assert "2 pattern" in summary

    def test_recall_includes_formatted_context(self, storage, monkeypatch):
        """recall response includes formatted_context and context_summary."""
        import memory_mcp.server as server_module
        from memory_mcp.server import app as server_app
        from memory_mcp.server.tools import cold_storage

        monkeypatch.setattr(server_module, "storage", storage)
        monkeypatch.setattr(server_module, "settings", storage.settings)
        # Also patch the app module where recall() gets storage from
        monkeypatch.setattr(server_app, "storage", storage)
        monkeypatch.setattr(server_app, "settings", storage.settings)
        # And patch cold_storage which imports storage at module load time
        monkeypatch.setattr(cold_storage, "storage", storage)
        monkeypatch.setattr(cold_storage, "settings", storage.settings)

        # Store a memory
        storage.store_memory(
            "The API uses JWT tokens for authentication.",
            MemoryType.PROJECT,
            tags=["auth", "api"],
        )

        recall_fn = server_module.recall.fn
        result = recall_fn(query="authentication tokens")

        # Response should have formatted_context
        assert hasattr(result, "formatted_context")
        assert hasattr(result, "context_summary")
        if result.memories:  # If we got results
            assert result.formatted_context is not None
            assert len(result.formatted_context) == len(result.memories)
            assert "Found" in result.context_summary


# ========== Input Validation Tests ==========


class TestInputValidation:
    """Tests for input validation in tool functions."""

    def test_list_memories_rejects_negative_offset(self, storage, monkeypatch):
        """list_memories returns error for negative offset."""
        from memory_mcp.server.tools import cold_storage

        monkeypatch.setattr(cold_storage, "storage", storage)
        monkeypatch.setattr(cold_storage, "settings", storage.settings)

        list_fn = cold_storage.list_memories.fn
        result = list_fn(offset=-1)

        # Should return error dict, not results
        assert isinstance(result, dict)
        assert "error" in result
        assert "offset" in result["error"]

    def test_list_memories_rejects_zero_limit(self, storage, monkeypatch):
        """list_memories returns error for limit < 1."""
        from memory_mcp.server.tools import cold_storage

        monkeypatch.setattr(cold_storage, "storage", storage)
        monkeypatch.setattr(cold_storage, "settings", storage.settings)

        list_fn = cold_storage.list_memories.fn
        result = list_fn(limit=0)

        # Should return error dict, not results
        assert isinstance(result, dict)
        assert "error" in result
        assert "limit" in result["error"]

    def test_list_memories_accepts_valid_params(self, storage, monkeypatch):
        """list_memories works with valid offset and limit."""
        from memory_mcp.server.tools import cold_storage

        monkeypatch.setattr(cold_storage, "storage", storage)
        monkeypatch.setattr(cold_storage, "settings", storage.settings)

        # Store a memory
        storage.store_memory("Test memory", MemoryType.PROJECT)

        list_fn = cold_storage.list_memories.fn
        result = list_fn(limit=5, offset=0)

        # Should return list, not error
        assert isinstance(result, list)
        assert len(result) >= 1
