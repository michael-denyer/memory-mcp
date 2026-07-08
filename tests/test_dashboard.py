"""Tests for the dashboard FastAPI application."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from memory_mcp.config import Settings
from memory_mcp.dashboard.app import app, get_projects
from memory_mcp.storage import MemoryType, Storage
from memory_mcp.storage.mining_runs import PROBE_SESSION_ID


def _ts(days_ago: float = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def dashboard_storage(tmp_path):
    """Create a storage instance for dashboard testing."""
    import memory_mcp.dashboard.app as app_module

    settings = Settings(db_path=tmp_path / "test.db", semantic_dedup_enabled=False)
    app_module.storage = Storage(settings)
    yield app_module.storage
    app_module.storage.close()
    app_module.storage = None


@pytest.fixture
def client(dashboard_storage):
    """Create a test client with initialized storage."""
    return TestClient(app)


def test_memories_page_loads(client, dashboard_storage):
    """Test that the memories page loads successfully."""
    response = client.get("/memories")
    assert response.status_code == 200
    assert b"Memory Browser" in response.content


def test_memories_page_includes_project_filter(client, dashboard_storage):
    """Test that the memories page includes project filter dropdown."""
    # Create memories with different projects
    dashboard_storage.store_memory("Project A memory", MemoryType.PROJECT, project_id="project-a")
    dashboard_storage.store_memory("Project B memory", MemoryType.PROJECT, project_id="project-b")

    response = client.get("/memories")
    assert response.status_code == 200
    # Check for project filter select element
    assert b'name="project_filter"' in response.content
    assert b"All projects" in response.content


def test_memories_api_search_with_project_filter(client, dashboard_storage):
    """Test API search endpoint with project filter."""
    # Create memories with different projects
    dashboard_storage.store_memory("Project A memory", MemoryType.PROJECT, project_id="project-a")
    dashboard_storage.store_memory("Project B memory", MemoryType.PROJECT, project_id="project-b")
    dashboard_storage.store_memory("Global memory", MemoryType.PROJECT)

    # Search without filter - should return all
    response = client.get("/api/memories/search")
    assert response.status_code == 200

    # Search with project filter
    response = client.get("/api/memories/search?project_filter=project-a")
    assert response.status_code == 200
    # Should include Project A and Global, but not Project B
    assert b"Project A memory" in response.content
    assert b"Global memory" in response.content
    assert b"Project B memory" not in response.content


def test_get_projects_returns_distinct_projects(dashboard_storage):
    """Test get_projects helper returns distinct project ids and paths."""
    # Create memories with different projects
    dashboard_storage.store_memory("Memory 1", MemoryType.PROJECT, project_id="github/owner/repo")
    dashboard_storage.store_memory("Memory 2", MemoryType.PROJECT, project_id="/Users/test/project")

    # Create session with project_path
    dashboard_storage.create_or_get_session("test-session", project_path="/another/path")

    projects = get_projects()

    # Should have all three projects
    project_ids = {p["id"] for p in projects}
    assert "github/owner/repo" in project_ids
    assert "/Users/test/project" in project_ids
    assert "/another/path" in project_ids

    # Check labels are shortened appropriately
    labels = {p["label"]: p["id"] for p in projects}
    assert "owner/repo" in labels  # GitHub format shortened
    assert "project" in labels  # Absolute path shows last component


def test_pagination_preserves_project_filter(client, dashboard_storage):
    """Pagination buttons must carry the project filter in hx-include."""
    for i in range(3):
        dashboard_storage.store_memory(
            f"Project A memory {i}", MemoryType.PROJECT, project_id="project-a"
        )

    # limit=2 over 3 rows -> 2 pages, so the Next button renders.
    response = client.get("/api/memories/search?project_filter=project-a&limit=2")
    assert response.status_code == 200
    assert "Page 1 of 2" in response.text
    assert "[name='project_filter']" in response.text


class TestInjectionResources:
    """Injection page reflects the resource values the server actually logs."""

    def _seed(self, storage, resource, n=1):
        for i in range(n):
            mid, _ = storage.store_memory(f"{resource} mem {i}", MemoryType.PROJECT)
            storage.log_injection(mid, resource)

    def test_stats_group_by_actual_resource(self, dashboard_storage):
        from memory_mcp.dashboard.app import _get_injection_stats

        self._seed(dashboard_storage, "hot-cache", 2)
        self._seed(dashboard_storage, "recall", 3)
        self._seed(dashboard_storage, "working-set", 1)  # legacy rows stay counted

        stats = _get_injection_stats(dashboard_storage)
        assert stats["by_resource"] == {"hot-cache": 2, "recall": 3, "working-set": 1}

    def test_recall_badge_renders_recall_not_working_set(self, client, dashboard_storage):
        mid, _ = dashboard_storage.store_memory("badge content", MemoryType.PROJECT)
        dashboard_storage.log_injection(mid, "recall")

        html = client.get("/api/injections").text
        assert "recall" in html
        assert "working-set" not in html

    def test_pagination_total_pages_reflects_real_count(self, client, dashboard_storage):
        # Exactly `limit` rows -> a real COUNT yields 1 page; the old guess (limit*2)
        # produced a phantom second page.
        self._seed(dashboard_storage, "recall", 2)
        assert "Page 1 of" not in client.get("/api/injections?limit=2").text

        # One more row tips it to a genuine second page.
        self._seed(dashboard_storage, "recall", 1)
        assert "Page 1 of 2" in client.get("/api/injections?limit=2").text


class TestCrossSessionPatterns:
    """Cross-session patterns are derived from injection history."""

    def test_storage_uses_injection_history(self, dashboard_storage):
        s = dashboard_storage
        shared, _ = s.store_memory("shared across sessions", MemoryType.PROJECT)
        solo, _ = s.store_memory("only one session", MemoryType.PROJECT)
        s.create_or_get_session("sess-1")
        s.create_or_get_session("sess-2")
        s.log_injections_batch([shared], "hot-cache", session_id="sess-1")
        s.log_injections_batch([shared], "hot-cache", session_id="sess-2")
        s.log_injections_batch([solo], "hot-cache", session_id="sess-1")

        patterns = s.get_cross_session_patterns(min_sessions=2)
        by_id = {p["id"]: p for p in patterns}
        assert by_id[shared]["session_count"] == 2
        assert solo not in by_id  # single-session memory is excluded

    def test_sessions_page_renders_cross_patterns(self, client, dashboard_storage):
        mid, _ = dashboard_storage.store_memory("cross session content xyz", MemoryType.PROJECT)
        dashboard_storage.log_injections_batch([mid], "hot-cache", session_id="sess-a")
        dashboard_storage.log_injections_batch([mid], "hot-cache", session_id="sess-b")

        html = client.get("/sessions").text
        assert "Cross-Session Patterns" in html
        assert "cross session content xyz" in html


class TestHotCacheRefresh:
    """Both hot-cache lists expose refreshable partial endpoints."""

    def test_promoted_partial_contains_promoted_memory(self, client, dashboard_storage):
        mid, _ = dashboard_storage.store_memory("promoted partial content", MemoryType.PROJECT)
        dashboard_storage.promote_to_hot(mid)
        resp = client.get("/api/promoted")
        assert resp.status_code == 200
        assert "promoted partial content" in resp.text

    def test_hot_cache_items_partial_contains_memory(self, client, dashboard_storage):
        mid, _ = dashboard_storage.store_memory("session aware content", MemoryType.PROJECT)
        dashboard_storage.promote_to_hot(mid)
        resp = client.get("/api/hot-cache/items")
        assert resp.status_code == 200
        assert "session aware content" in resp.text

    def test_promoted_partial_shows_pinned_badge(self, client, dashboard_storage):
        mid, _ = dashboard_storage.store_memory("pinned memory content", MemoryType.PROJECT)
        dashboard_storage.promote_to_hot(mid, pin=True)
        resp = client.get("/api/promoted")
        assert resp.status_code == 200
        assert "pinned" in resp.text

    def test_hot_cache_page_renders_pinned_badge_on_initial_load(self, client, dashboard_storage):
        # Initial render must use memory.is_pinned (not the empty pinned_ids),
        # so the pin badge survives a page load, matching the refresh partial.
        mid, _ = dashboard_storage.store_memory("initial pinned content", MemoryType.PROJECT)
        dashboard_storage.promote_to_hot(mid, pin=True)
        resp = client.get("/hot-cache")
        assert resp.status_code == 200
        assert "initial pinned content" in resp.text
        assert "pinned" in resp.text


class TestMemoryDetail:
    """Expandable memory-detail row endpoint."""

    def test_detail_shows_full_content_and_relationship(self, client, dashboard_storage):
        from memory_mcp.models import RelationType

        long_content = "detailed memory body " + "x" * 300
        a, _ = dashboard_storage.store_memory(long_content, MemoryType.PROJECT)
        b, _ = dashboard_storage.store_memory("the linked neighbour memory", MemoryType.REFERENCE)
        dashboard_storage.link_memories(a, b, RelationType.RELATES_TO)

        resp = client.get(f"/api/memories/{a}/detail")
        assert resp.status_code == 200
        assert "x" * 300 in resp.text  # full content, not the row's truncated view
        assert "the linked neighbour memory" in resp.text  # outgoing relationship snippet
        assert "relates_to" in resp.text

    def test_detail_unknown_id_returns_empty_200(self, client, dashboard_storage):
        resp = client.get("/api/memories/999999/detail")
        assert resp.status_code == 200
        assert resp.text == ""


class TestKnowledgeGraph:
    """Graph page legend and graph data API."""

    def test_graph_page_has_full_node_and_edge_legend(self, client, dashboard_storage):
        resp = client.get("/graph")
        assert resp.status_code == 200
        assert "conversation" in resp.text  # legend now covers all five node types
        assert "episodic" in resp.text
        assert 'id="edge-legend"' in resp.text

    def test_api_graph_returns_nodes_and_edges(self, client, dashboard_storage):
        from memory_mcp.models import RelationType

        a, _ = dashboard_storage.store_memory("graph node a", MemoryType.PROJECT)
        b, _ = dashboard_storage.store_memory("graph node b", MemoryType.REFERENCE)
        dashboard_storage.link_memories(a, b, RelationType.DEPENDS_ON)

        data = client.get("/api/graph").json()
        ids = {n["id"] for n in data["nodes"]}
        assert {a, b} <= ids
        assert any(
            e["from"] == a and e["to"] == b and e["type"] == "depends_on" for e in data["edges"]
        )


class TestStaticAssets:
    """The dashboard serves its own vendored CSS/JS, no CDN."""

    def test_tailwind_css_served(self, client):
        resp = client.get("/static/tailwind.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    def test_htmx_served(self, client):
        resp = client.get("/static/htmx.min.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]

    def test_base_page_uses_vendored_assets_not_cdn(self, client):
        html = client.get("/").text
        assert "/static/tailwind.css" in html
        assert "/static/htmx.min.js" in html
        assert "cdn.tailwindcss.com" not in html


class TestMiningLoopBanner:
    def test_empty_db_shows_amber(self, client):
        html = client.get("/mining").text
        assert "loop-banner-amber" in html

    def test_recent_success_shows_green(self, client, dashboard_storage):
        dashboard_storage.record_mining_run(
            started_at=_ts(),
            finished_at=_ts(),
            stats={"outputs_processed": 1, "patterns_found": 1, "new_memories": 1},
        )
        assert "loop-banner-green" in client.get("/mining").text

    def test_error_streak_shows_red(self, client, dashboard_storage):
        for _ in range(3):
            dashboard_storage.record_mining_run(
                started_at=_ts(), finished_at=_ts(), stats={}, error="boom"
            )
        assert "loop-banner-red" in client.get("/mining").text

    def test_output_stat_excludes_probe_rows(self, client, dashboard_storage):
        dashboard_storage.log_output("probe leftover", session_id=PROBE_SESSION_ID)
        html = client.get("/mining").text
        # The Output Logs stat card renders as:
        #   <p class="text-xs text-gray-400">Output Logs</p>
        #   <p class="text-2xl font-bold text-white">{{ mining_stats.output_count }}</p>
        # Anchor on the label through to its value so this can't coincidentally
        # match the "Mined Patterns" card, which shares the same value markup
        # and would also render 0.
        assert (
            '<p class="text-xs text-gray-400">Output Logs</p>\n'
            '            <p class="text-2xl font-bold text-white">0</p>'
        ) in html


class TestInjectionsPage:
    # Storage opens its SQLite connection lazily, so on a fresh instance
    # s._conn is None until the first query runs. These requests must be the
    # first thing to touch the database.

    def test_page_loads_on_cold_storage(self, client):
        response = client.get("/injections")
        assert response.status_code == 200
        assert "Injection History" in response.text

    def test_api_partial_loads_on_cold_storage(self, client):
        # Hits _get_injections without _get_injection_stats running first,
        # so it can't rely on an earlier call having opened the connection.
        response = client.get("/api/injections")
        assert response.status_code == 200
