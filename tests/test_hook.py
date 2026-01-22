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
