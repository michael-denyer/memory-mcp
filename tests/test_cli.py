"""Tests for CLI commands."""

import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_mcp.cli import main
from memory_mcp.config import Settings
from memory_mcp.storage import MemoryType, Storage


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.dict("os.environ", {"MEMORY_MCP_DB_PATH": str(db_path)}):
            yield db_path


class TestLogResponseCommand:
    """Tests for the log-response CLI command (Stop hook entrypoint)."""

    def _hook_stdin(self, tmp_path, assistant_text):
        """Build the Stop hook's stdin JSON over a two-turn transcript."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "How do I deploy?"}],
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        return json.dumps({"transcript_path": str(transcript), "session_id": "sess-auto-mark-test"})

    def test_log_response_auto_marks_used_memories(self, temp_db, tmp_path, capsys):
        """A response echoing a distinctive token from an injected memory
        auto-marks that memory as used."""
        from memory_mcp.config import get_settings
        from memory_mcp.models import MemorySource, MemoryType
        from memory_mcp.storage import Storage

        storage = Storage(get_settings())
        memory_id, _ = storage.store_memory(
            content="use `make deploy-staging` for releases",
            memory_type=MemoryType.PATTERN,
            source=MemorySource.MINED,
        )
        storage.log_injection(memory_id, resource="hot-cache", session_id="s1")
        storage.close()

        hook_input = self._hook_stdin(tmp_path, assistant_text="ran make deploy-staging")

        with (
            patch("sys.stdin.read", return_value=hook_input),
            patch("sys.argv", ["memory-mcp-cli", "log-response"]),
        ):
            result = main()
        assert result == 0

        storage = Storage(get_settings())
        try:
            with storage._connection() as conn:
                used = conn.execute(
                    "SELECT used_count FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()[0]
        finally:
            storage.close()
        assert used == 1

    def test_log_response_demotes_stale_hot_memory(self, temp_db, tmp_path):
        """The Stop hook runs maintenance, so a hot memory nobody has touched
        for 30 days is cold by the time the turn ends."""
        from memory_mcp.config import get_settings

        storage = Storage(get_settings())
        memory_id, _ = storage.store_memory("stale hot fact about widgets", MemoryType.PROJECT)
        storage.promote_to_hot(memory_id)
        with storage.transaction() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed_at = datetime('now', '-30 days') WHERE id = ?",
                (memory_id,),
            )
        storage.close()

        hook_input = self._hook_stdin(tmp_path, assistant_text="some unrelated assistant reply")

        with (
            patch("sys.stdin.read", return_value=hook_input),
            patch("sys.argv", ["memory-mcp-cli", "log-response"]),
        ):
            assert main() == 0

        storage = Storage(get_settings())
        try:
            with storage._connection() as conn:
                is_hot = conn.execute(
                    "SELECT is_hot FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()[0]
        finally:
            storage.close()

        assert is_hot == 0

    def test_auto_mark_failure_never_blocks_log_response(self, temp_db, tmp_path):
        """mark_used_memories raising must not fail the Stop hook."""
        hook_input = self._hook_stdin(tmp_path, assistant_text="some unrelated assistant reply")

        with (
            patch.object(Storage, "mark_used_memories", side_effect=RuntimeError("boom")),
            patch("sys.stdin.read", return_value=hook_input),
            patch("sys.argv", ["memory-mcp-cli", "log-response"]),
        ):
            result = main()
        assert result == 0

    def test_log_response_still_marks_used_without_mining(self, temp_db, tmp_path):
        """The Stop hook feeds the recent-recalls slot with no mining in the chain."""
        storage = Storage(Settings(db_path=temp_db))
        memory_id, _ = storage.store_memory(
            "the deploy password hint is zebra-42", MemoryType.PROJECT
        )
        storage.log_injection(memory_id, resource="hook", session_id="lane3")
        storage.close()

        hook_input = self._hook_stdin(tmp_path, assistant_text="The hint is zebra-42, as I recall.")

        with (
            patch("sys.stdin.read", return_value=hook_input),
            patch("sys.argv", ["memory-mcp-cli", "log-response"]),
        ):
            assert main() == 0

        storage = Storage(Settings(db_path=temp_db))
        try:
            with storage._connection() as conn:
                used_rows = conn.execute(
                    "SELECT memory_id FROM retrieval_events WHERE was_used = 1"
                ).fetchall()
            recalled = [m.id for m in storage.get_recent_recalls()]
        finally:
            storage.close()

        assert [row["memory_id"] for row in used_rows] == [memory_id]
        assert memory_id in recalled


class TestBootstrapCommand:
    """Tests for the bootstrap CLI command."""

    def test_bootstrap_skips_claude_md_and_does_not_promote(self, temp_db, tmp_path, capsys):
        """Claude Code already injects CLAUDE.md, and bootstrap no longer promotes."""
        (tmp_path / "CLAUDE.md").write_text("- the claude instruction is aardvark-7\n")
        (tmp_path / "README.md").write_text("- the readme fact is buffalo-9\n")

        with patch("sys.argv", ["memory-mcp-cli", "bootstrap", "-r", str(tmp_path)]):
            assert main() == 0

        storage = Storage(Settings(db_path=temp_db))
        try:
            with storage._connection() as conn:
                rows = conn.execute("SELECT content, is_hot FROM memories").fetchall()
        finally:
            storage.close()

        contents = [row["content"] for row in rows]
        assert any("buffalo-9" in c for c in contents)
        assert not any("aardvark-7" in c for c in contents)
        assert [row["is_hot"] for row in rows] == [0] * len(rows)


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


class TestHotCacheCommand:
    """Tests for `hot-cache`, the command Claude Code's hooks run for injection."""

    def _seed_two_promoted(self, temp_db):
        storage = Storage(Settings(db_path=temp_db))
        try:
            for content in (
                "The deploy password hint is zebra-42.",
                "Run make lint before every push.",
            ):
                memory_id, _ = storage.store_memory(content, MemoryType.PROJECT)
                storage.promote_to_hot(memory_id)
        finally:
            storage.close()

    def _run(self, force=False, stdin='{"session_id":"s1"}'):
        argv = ["memory-mcp-cli", "hot-cache"]
        if force:
            argv.append("--force")
        with patch("sys.stdin", io.StringIO(stdin)), patch("sys.argv", argv):
            return main()

    def test_prints_nothing_when_empty(self, temp_db, capsys):
        assert self._run() == 0
        assert capsys.readouterr().out == ""

    def test_prints_memories_with_ids(self, temp_db, capsys):
        self._seed_two_promoted(temp_db)

        assert self._run() == 0

        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "[MEMORY: Hot cache]"
        assert len([ln for ln in lines if ln.startswith("- [id:")]) == 2
        assert any("zebra-42" in ln for ln in lines)
        assert lines[-1] == "Call mark_memory_used(id) when one of these was useful."

    def test_second_call_same_session_prints_nothing(self, temp_db, capsys):
        self._seed_two_promoted(temp_db)
        self._run()
        capsys.readouterr()

        assert self._run() == 0
        assert capsys.readouterr().out == ""

    def test_force_prints_again(self, temp_db, capsys):
        self._seed_two_promoted(temp_db)
        self._run()
        first = capsys.readouterr().out
        assert "zebra-42" in first

        assert self._run(force=True) == 0
        assert capsys.readouterr().out == first

    def test_force_clears_the_stamp_even_when_it_prints_nothing(self, temp_db, capsys):
        self._seed_two_promoted(temp_db)
        self._run()
        capsys.readouterr()

        with patch.object(Storage, "get_hot_cache", return_value=[]):
            assert self._run(force=True) == 0
        assert capsys.readouterr().out == ""

        assert self._run() == 0
        assert "zebra-42" in capsys.readouterr().out

    def test_logs_injection_rows_with_resource_hook(self, temp_db):
        self._seed_two_promoted(temp_db)

        assert self._run() == 0

        storage = Storage(Settings(db_path=temp_db))
        try:
            with storage._connection() as conn:
                rows = conn.execute(
                    "SELECT memory_id FROM injection_log WHERE resource = 'hook'"
                ).fetchall()
        finally:
            storage.close()
        assert len(rows) == 2

    def test_exception_goes_to_stderr_and_exit_zero(self, temp_db, capsys):
        with patch.object(Storage, "get_hot_cache", side_effect=RuntimeError("boom")):
            assert self._run() == 0

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "boom" in captured.err

    def test_stamp_filename_is_hashed_session_id(self, tmp_path):
        db_path = tmp_path / "data" / "memory.db"
        db_path.parent.mkdir()

        with patch.dict("os.environ", {"MEMORY_MCP_DB_PATH": str(db_path)}):
            self._seed_two_promoted(db_path)
            assert self._run(stdin='{"session_id": "../../escape"}') == 0

        assert not (tmp_path / "escape").exists()
        assert not (db_path.parent / "escape").exists()

        injected = db_path.parent / "injected"
        assert [p.name for p in injected.iterdir()] == [hashlib.sha256(b"../../escape").hexdigest()]
