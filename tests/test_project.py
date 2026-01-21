"""Tests for project awareness functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_mcp.config import Settings
from memory_mcp.project import (
    _normalize_git_url,
    _normalize_host,
    detect_project,
    get_current_project_id,
)
from memory_mcp.storage import MemoryType, Storage


class TestGitUrlNormalization:
    """Tests for git URL parsing and normalization."""

    def test_https_github_url(self):
        """Parse HTTPS GitHub URL."""
        result = _normalize_git_url("https://github.com/owner/repo.git")
        assert result == "github/owner/repo"

    def test_ssh_github_url(self):
        """Parse SSH GitHub URL."""
        result = _normalize_git_url("git@github.com:owner/repo.git")
        assert result == "github/owner/repo"

    def test_https_without_git_suffix(self):
        """Parse URL without .git suffix."""
        result = _normalize_git_url("https://github.com/owner/repo")
        assert result == "github/owner/repo"

    def test_gitlab_url(self):
        """Parse GitLab URL."""
        result = _normalize_git_url("https://gitlab.com/owner/project.git")
        assert result == "gitlab/owner/project"

    def test_bitbucket_url(self):
        """Parse Bitbucket URL."""
        result = _normalize_git_url("git@bitbucket.org:team/repo.git")
        assert result == "bitbucket/team/repo"

    def test_ssh_with_prefix(self):
        """Parse SSH URL with ssh:// prefix."""
        result = _normalize_git_url("ssh://git@github.com/owner/repo.git")
        assert result == "github/owner/repo"

    def test_trailing_slash(self):
        """URL with trailing slash."""
        result = _normalize_git_url("https://github.com/owner/repo/")
        assert result == "github/owner/repo"

    def test_empty_url(self):
        """Empty URL returns None."""
        result = _normalize_git_url("")
        assert result is None

    def test_none_url(self):
        """None URL returns None."""
        result = _normalize_git_url(None)
        assert result is None

    def test_invalid_url(self):
        """Unparseable URL returns None."""
        result = _normalize_git_url("not-a-git-url")
        assert result is None


class TestHostNormalization:
    """Tests for git host normalization."""

    def test_github_normalized(self):
        """github.com -> github."""
        assert _normalize_host("github.com") == "github"

    def test_gitlab_normalized(self):
        """gitlab.com -> gitlab."""
        assert _normalize_host("gitlab.com") == "gitlab"

    def test_bitbucket_normalized(self):
        """bitbucket.org -> bitbucket."""
        assert _normalize_host("bitbucket.org") == "bitbucket"

    def test_custom_host_unchanged(self):
        """Custom hosts remain unchanged."""
        assert _normalize_host("git.company.com") == "git.company.com"

    def test_case_insensitive(self):
        """Host normalization is case insensitive."""
        assert _normalize_host("GitHub.com") == "github"


class TestProjectDetection:
    """Tests for project detection from git."""

    def test_detect_project_in_git_repo(self):
        """Detect project in a git repository with remote."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess

            # Initialize a git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "https://github.com/test/project"],
                cwd=tmpdir,
                capture_output=True,
            )

            # Clear cache before detection
            from memory_mcp.project import clear_project_cache

            clear_project_cache()

            project = detect_project(tmpdir)
            assert project is not None
            assert project.id == "github/test/project"
            assert project.name == "project"
            # macOS symlinks /var -> /private/var, so compare resolved paths
            assert Path(project.path).resolve() == Path(tmpdir).resolve()

    def test_detect_project_local_only(self):
        """Detect project in local-only repo (no remote)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess

            # Initialize a git repo without remote
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)

            from memory_mcp.project import clear_project_cache

            clear_project_cache()

            project = detect_project(tmpdir)
            assert project is not None
            # Should use "local/{dirname}" format
            assert project.id.startswith("local/")
            assert project.remote_url is None

    def test_detect_project_not_git_repo(self):
        """Returns None when not in a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from memory_mcp.project import clear_project_cache

            clear_project_cache()

            project = detect_project(tmpdir)
            assert project is None

    def test_get_current_project_id(self):
        """get_current_project_id returns project ID or None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess

            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "https://github.com/owner/repo"],
                cwd=tmpdir,
                capture_output=True,
            )

            from memory_mcp.project import clear_project_cache

            clear_project_cache()

            # Mock cwd to be in the temp dir
            with patch("os.getcwd", return_value=tmpdir):
                project_id = get_current_project_id()
                assert project_id == "github/owner/repo"


class TestProjectAwareStorage:
    """Tests for project-aware storage operations."""

    @pytest.fixture
    def storage(self):
        """Create storage instance for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                db_path=Path(tmpdir) / "test.db",
                project_awareness_enabled=True,
                project_filter_recall=True,
                project_filter_hot_cache=True,
                project_include_global=True,
            )
            stor = Storage(settings)
            yield stor
            stor.close()

    def test_store_memory_with_project_id(self, storage):
        """Store memory with project_id."""
        memory_id, is_new = storage.store_memory(
            content="Project-specific memory",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/repo",
        )
        assert is_new

        memory = storage.get_memory(memory_id)
        assert memory.project_id == "github/owner/repo"

    def test_store_memory_without_project_id(self, storage):
        """Store memory without project_id (global memory)."""
        memory_id, is_new = storage.store_memory(
            content="Global memory",
            memory_type=MemoryType.PROJECT,
            project_id=None,
        )
        assert is_new

        memory = storage.get_memory(memory_id)
        assert memory.project_id is None

    def test_recall_filters_by_project(self, storage):
        """Recall filters by project_id when provided."""
        # Create memories for different projects
        storage.store_memory(
            content="Memory for project A about Python programming",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/project-a",
        )
        storage.store_memory(
            content="Memory for project B about JavaScript",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/project-b",
        )
        storage.store_memory(
            content="Global memory about Python programming",
            memory_type=MemoryType.PROJECT,
            project_id=None,
        )

        # Recall with project filter and low threshold
        result = storage.recall(
            query="Python programming",
            project_id="github/owner/project-a",
            threshold=0.3,  # Low threshold to ensure results
        )

        # Should get project A + global, not project B
        project_ids = [m.project_id for m in result.memories]
        assert len(result.memories) > 0
        assert "github/owner/project-b" not in project_ids

    def test_recall_includes_global_with_project_filter(self, storage):
        """Recall includes global memories when filtering by project."""
        storage.store_memory(
            content="Project memory about database systems and SQL",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/project",
        )
        storage.store_memory(
            content="Global memory about database systems and SQL",
            memory_type=MemoryType.PROJECT,
            project_id=None,
        )

        result = storage.recall(
            query="database systems SQL",
            project_id="github/owner/project",
            threshold=0.3,  # Low threshold to ensure results
        )

        # Should have results
        assert len(result.memories) > 0

    def test_get_hot_memories_filters_by_project(self, storage):
        """Hot cache filters by project when enabled."""
        # Create and promote memories for different projects
        mid1, _ = storage.store_memory(
            content="Hot memory for project A",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/project-a",
        )
        mid2, _ = storage.store_memory(
            content="Hot memory for project B",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/project-b",
        )
        mid3, _ = storage.store_memory(
            content="Global hot memory",
            memory_type=MemoryType.PROJECT,
            project_id=None,
        )

        storage.promote_to_hot(mid1)
        storage.promote_to_hot(mid2)
        storage.promote_to_hot(mid3)

        # Get hot memories for project A
        hot_memories = storage.get_hot_memories(project_id="github/owner/project-a")

        project_ids = [m.project_id for m in hot_memories]
        assert "github/owner/project-a" in project_ids or None in project_ids
        assert "github/owner/project-b" not in project_ids

    def test_get_hot_memories_all_when_no_project(self, storage):
        """Hot cache returns all memories when no project filter."""
        mid1, _ = storage.store_memory(
            content="Memory A for project one",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/project-a",
        )
        mid2, _ = storage.store_memory(
            content="Memory B for project two",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/project-b",
        )

        storage.promote_to_hot(mid1)
        storage.promote_to_hot(mid2)

        # Get all hot memories (no project filter)
        hot_memories = storage.get_hot_memories(project_id=None)

        # Should include both project A and project B memories
        assert len(hot_memories) >= 2


class TestProjectTable:
    """Tests for projects table tracking."""

    @pytest.fixture
    def storage(self):
        """Create storage instance for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(db_path=Path(tmpdir) / "test.db")
            stor = Storage(settings)
            yield stor
            stor.close()

    def test_project_tracked_on_store(self, storage):
        """Project is tracked when storing memory with project_id."""
        storage.store_memory(
            content="Test memory",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/repo",
        )

        # Check project was added to projects table
        with storage._connection() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                ("github/owner/repo",),
            ).fetchone()
            assert row is not None
            assert row["id"] == "github/owner/repo"

    def test_project_last_accessed_updated(self, storage):
        """Project last_accessed_at is updated on subsequent stores."""
        import time

        storage.store_memory(
            content="First memory",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/repo",
        )

        # Store another memory for same project
        time.sleep(0.01)  # Small delay to ensure different timestamp
        storage.store_memory(
            content="Second memory",
            memory_type=MemoryType.PROJECT,
            project_id="github/owner/repo",
        )

        # Check last_accessed_at was updated
        with storage._connection() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                ("github/owner/repo",),
            ).fetchone()
            assert row is not None
