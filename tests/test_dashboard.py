"""Tests for the dashboard FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from memory_mcp.config import Settings
from memory_mcp.dashboard.app import app, get_projects
from memory_mcp.storage import MemoryType, Storage


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
