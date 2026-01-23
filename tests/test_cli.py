"""Tests for CLI commands."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_mcp.cli import main


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.dict("os.environ", {"MEMORY_MCP_DB_PATH": str(db_path)}):
            yield db_path


class TestLogOutputCommand:
    """Tests for the log-output CLI command."""

    def test_log_output_from_stdin(self, temp_db):
        """Should log content from stdin."""
        with patch("sys.stdin.read", return_value="Test content from stdin"):
            with patch("sys.argv", ["memory-mcp-cli", "log-output"]):
                result = main()
        assert result == 0

    def test_log_output_from_content_arg(self, temp_db):
        """Should log content from --content argument."""
        with patch("sys.argv", ["memory-mcp-cli", "log-output", "-c", "Test content from arg"]):
            result = main()
        assert result == 0

    def test_log_output_json_format(self, temp_db, capsys):
        """Should output JSON when --json flag is used."""
        with patch("sys.argv", ["memory-mcp-cli", "--json", "log-output", "-c", "Test content"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["success"] is True
        assert "log_id" in output

    def test_log_output_empty_content_fails(self, temp_db):
        """Should fail with empty content."""
        with patch("sys.stdin.read", return_value=""):
            with patch("sys.argv", ["memory-mcp-cli", "log-output"]):
                result = main()
        assert result == 1

    def test_log_output_from_file(self, temp_db):
        """Should log content from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content from file")
            f.flush()

            with patch("sys.argv", ["memory-mcp-cli", "log-output", "-f", f.name]):
                result = main()

        assert result == 0

    def test_log_output_stores_project_id(self, temp_db):
        """Should store project_id when project awareness is enabled.

        This is a regression test for the bug where log_output CLI
        didn't pass project_id to storage, causing run_mining to find
        0 outputs (since mining filters by project_id).
        """
        from memory_mcp.config import get_settings
        from memory_mcp.storage import Storage

        test_project_id = "github/test-org/test-repo"

        # Mock get_current_project_id to return a known project
        with patch("memory_mcp.cli.get_current_project_id", return_value=test_project_id):
            with patch("sys.argv", ["memory-mcp-cli", "log-output", "-c", "Test with project_id"]):
                result = main()

        assert result == 0

        # Verify the project_id was stored
        settings = get_settings()
        storage = Storage(settings)
        try:
            outputs = storage.get_recent_outputs(hours=1, project_id=test_project_id)
            assert len(outputs) >= 1
            # The output should be found when filtering by project_id
            contents = [content for _, content, _, _, _ in outputs]
            assert any("Test with project_id" in c for c in contents)
        finally:
            storage.close()

    def test_log_output_project_id_enables_mining(self, temp_db):
        """Mining should find outputs logged with matching project_id.

        This tests the full flow: log_output with project_id → run_mining
        finds the output because project_ids match.
        """
        test_project_id = "github/test-org/test-repo"

        # Log output with project_id
        with patch("memory_mcp.cli.get_current_project_id", return_value=test_project_id):
            with patch(
                "sys.argv", ["memory-mcp-cli", "log-output", "-c", "We use FastAPI for the API"]
            ):
                result = main()
        assert result == 0

        # Run mining with same project_id
        with patch("memory_mcp.cli.get_current_project_id", return_value=test_project_id):
            with patch("sys.argv", ["memory-mcp-cli", "--json", "run-mining"]):
                result = main()

        assert result == 0


class TestRunMiningCommand:
    """Tests for the run-mining CLI command."""

    def test_run_mining_basic(self, temp_db, capsys):
        """Should run mining without errors."""
        with patch("sys.argv", ["memory-mcp-cli", "run-mining"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Mining Results" in captured.out or "Outputs processed" in captured.out

    def test_run_mining_json_format(self, temp_db, capsys):
        """Should output JSON when --json flag is used."""
        with patch("sys.argv", ["memory-mcp-cli", "--json", "run-mining"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "outputs_processed" in output
        assert "patterns_found" in output

    def test_run_mining_with_hours(self, temp_db, capsys):
        """Should accept --hours argument."""
        with patch("sys.argv", ["memory-mcp-cli", "run-mining", "--hours", "48"]):
            result = main()

        assert result == 0


class TestSeedCommand:
    """Tests for the seed CLI command."""

    def test_seed_from_list_file(self, temp_db, capsys):
        """Should seed memories from a file with list items."""
        content = """Project facts:
- This project uses FastAPI for the web framework
- Database is PostgreSQL with pgvector extension
- Testing with pytest and coverage
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()

            with patch("sys.argv", ["memory-mcp-cli", "seed", f.name]):
                result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Created" in captured.out

    def test_seed_from_paragraph_file(self, temp_db, capsys):
        """Should seed memories from paragraphs."""
        content = """This is the first paragraph with important project information.

This is the second paragraph describing the architecture.

This is the third paragraph about dependencies.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()

            with patch("sys.argv", ["memory-mcp-cli", "seed", f.name]):
                result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Created" in captured.out

    def test_seed_with_type_option(self, temp_db, capsys):
        """Should accept --type option."""
        content = "- Pattern one for code\n- Pattern two for imports"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()

            with patch("sys.argv", ["memory-mcp-cli", "seed", f.name, "-t", "pattern"]):
                result = main()

        assert result == 0

    def test_seed_with_promote_option(self, temp_db, capsys):
        """Should accept --promote option."""
        content = "- Important fact to remember and promote"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()

            with patch("sys.argv", ["memory-mcp-cli", "seed", f.name, "--promote"]):
                result = main()

        assert result == 0

    def test_seed_json_output(self, temp_db, capsys):
        """Should output JSON when --json flag is used."""
        content = "- Fact one to seed\n- Fact two to seed"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()

            with patch("sys.argv", ["memory-mcp-cli", "--json", "seed", f.name]):
                result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "memories_created" in output
        assert "memories_skipped" in output

    def test_seed_nonexistent_file(self, temp_db):
        """Should fail gracefully for nonexistent file."""
        with patch("sys.argv", ["memory-mcp-cli", "seed", "/nonexistent/file.md"]):
            result = main()
        assert result == 1

    def test_seed_invalid_type(self, temp_db):
        """Should fail for invalid memory type."""
        content = "- Some content"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()

            with patch("sys.argv", ["memory-mcp-cli", "seed", f.name, "-t", "invalid"]):
                result = main()

        assert result == 1


class TestCliIntegration:
    """Integration tests using subprocess."""

    def test_cli_help(self):
        """Should show help text."""
        result = subprocess.run(
            [sys.executable, "-m", "memory_mcp.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "memory-mcp-cli" in result.stdout or "CLI commands" in result.stdout

    def test_log_output_help(self):
        """Should show log-output help."""
        result = subprocess.run(
            [sys.executable, "-m", "memory_mcp.cli", "log-output", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "content" in result.stdout.lower()

    def test_seed_help(self):
        """Should show seed help."""
        result = subprocess.run(
            [sys.executable, "-m", "memory_mcp.cli", "seed", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "file" in result.stdout.lower()
