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
    """Create a temporary database path for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.dict(
            "os.environ",
            {"MEMORY_MCP_DB_PATH": str(db_path)},
        ):
            yield db_path


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
