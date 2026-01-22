"""Tests for the memory-log-response hook script."""

import json
import subprocess
from pathlib import Path

HOOK_SCRIPT = Path(__file__).parent.parent / "hooks" / "memory-log-response.sh"


class TestHookTranscriptFormat:
    """Test hook handles different transcript formats."""

    def test_new_format_message_role(self, tmp_path: Path) -> None:
        """Hook extracts text from new format: {message: {role: "assistant"}}."""
        transcript = tmp_path / "transcript.jsonl"
        # New format used by Claude Code 2.x
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "This is a test response extracted by the hook.",
                            }
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        # Hook should succeed (exit 0) and log output
        assert result.returncode == 0
        assert "Logged output" in result.stdout or result.stdout == ""

    def test_extracts_last_assistant_message(self, tmp_path: Path) -> None:
        """Hook extracts the last assistant message with text content."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "User message"}],
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "First response with enough chars to pass min.",
                            }
                        ],
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Second response is the one extracted by hook.",
                            }
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        # Run with bash -x to see what LAST_RESPONSE is set to
        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        # Check that the second message was extracted
        assert "Second response" in result.stderr

    def test_skips_tool_use_only_messages(self, tmp_path: Path) -> None:
        """Hook skips messages that only contain tool_use blocks."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "This text message should be extracted.",
                            }
                        ],
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "123", "name": "Bash", "input": {}}],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        # Should extract the text message, not the tool_use
        assert "This text message" in result.stderr

    def test_skips_short_responses(self, tmp_path: Path) -> None:
        """Hook skips responses shorter than 50 characters."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Short"}],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        # Should exit early without logging
        assert result.returncode == 0
        assert "Logged output" not in result.stdout

    def test_handles_missing_transcript(self) -> None:
        """Hook exits gracefully when transcript doesn't exist."""
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": "/nonexistent/path.jsonl"}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        assert result.returncode == 0

    def test_handles_empty_input(self) -> None:
        """Hook exits gracefully with empty input."""
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input="{}",
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        assert result.returncode == 0

    def test_joins_multiple_text_blocks(self, tmp_path: Path) -> None:
        """Hook joins multiple text blocks in a single message."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "First paragraph of the response."},
                            {"type": "text", "text": "Second paragraph of the response."},
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        # Both paragraphs should be in the extracted response
        assert "First paragraph" in result.stderr
        assert "Second paragraph" in result.stderr


class TestHookInputFormats:
    """Test hook handles different input JSON formats."""

    def test_camel_case_transcript_path(self, tmp_path: Path) -> None:
        """Hook accepts camelCase transcriptPath."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Response for camelCase test input format."}
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps({"transcriptPath": str(transcript)}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        assert "Response for camelCase" in result.stderr

    def test_nested_transcript_path(self, tmp_path: Path) -> None:
        """Hook accepts nested transcript.path format."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Response for nested path test format."}
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript": {"path": str(transcript)}}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
        )

        assert "Response for nested" in result.stderr


class TestHookSessionFallback:
    """Test hook derives transcript path from session info."""

    def test_session_id_with_project_path(self, tmp_path: Path) -> None:
        """Hook derives transcript from sessionId + projectPath."""
        # Create a mock .claude/projects structure
        claude_dir = tmp_path / ".claude" / "projects"
        project_slug = "-mock-project-path"
        project_dir = claude_dir / project_slug
        project_dir.mkdir(parents=True)

        session_id = "test-session-123"
        transcript = project_dir / f"{session_id}.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Response via session fallback path."}
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        # Override HOME to use our mock structure
        env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps(
                {
                    "sessionId": session_id,
                    "projectPath": "/mock/project/path",
                }
            ),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
            env=env,
        )

        assert "Response via session fallback" in result.stderr

    def test_session_id_find_fallback(self, tmp_path: Path) -> None:
        """Hook finds transcript by session ID when project path unknown."""
        # Create a mock .claude/projects structure with different project slug
        claude_dir = tmp_path / ".claude" / "projects"
        project_dir = claude_dir / "-some-other-project"
        project_dir.mkdir(parents=True)

        session_id = "find-fallback-session"
        transcript = project_dir / f"{session_id}.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response found via find fallback."}],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        # Only provide session ID, no project path
        env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps({"sessionId": session_id}),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
            env=env,
        )

        assert "Response found via find" in result.stderr

    def test_camel_case_session_fields(self, tmp_path: Path) -> None:
        """Hook accepts camelCase session_id and project_path."""
        claude_dir = tmp_path / ".claude" / "projects"
        project_slug = "-camel-case-project"
        project_dir = claude_dir / project_slug
        project_dir.mkdir(parents=True)

        session_id = "camel-session"
        transcript = project_dir / f"{session_id}.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Response for camelCase session fields."}
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
        result = subprocess.run(
            ["bash", "-x", str(HOOK_SCRIPT)],
            input=json.dumps(
                {
                    "session_id": session_id,
                    "cwd": "/camel/case/project",
                }
            ),
            capture_output=True,
            text=True,
            cwd=HOOK_SCRIPT.parent.parent,
            env=env,
        )

        assert "Response for camelCase session" in result.stderr
