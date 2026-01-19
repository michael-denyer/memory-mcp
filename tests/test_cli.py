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


class TestRunMiningCommand:
    """Tests for the run-mining CLI command."""

    def test_run_mining_basic(self, temp_db, capsys):
        """Should run mining without errors."""
        with patch("sys.argv", ["memory-mcp-cli", "run-mining"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Processed" in captured.out

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
