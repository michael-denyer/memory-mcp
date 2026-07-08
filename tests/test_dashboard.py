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
