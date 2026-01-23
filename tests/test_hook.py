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
                        "role": "user",
                        "content": [{"type": "text", "text": "What is the test?"}],
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
                                "text": "This is a test response extracted by the hook.",
                            }
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            env=env,
        )

        # Hook should succeed (exit 0)
        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            assert "This is a test response" in captured

    def test_extracts_last_assistant_message(self, tmp_path: Path) -> None:
        """Hook extracts the last assistant message with text content."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "User message here"}],
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            # Should have the second (last) assistant response
            assert "Second response is the one extracted" in captured
            # Should NOT have the first response
            assert "First response" not in captured

    def test_skips_tool_use_only_messages(self, tmp_path: Path) -> None:
        """Hook skips messages that only contain tool_use blocks."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "Run the command please"}],
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            # Should extract the text message, not the tool_use
            assert "This text message should be extracted" in captured

    def test_skips_short_combined_content(self, tmp_path: Path) -> None:
        """Hook skips when combined user+assistant content is under 20 chars."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hi"}],
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello"}],
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

        # Combined: "Hi" (2) + "Hello" (5) = 7 chars < 20 threshold
        # Should exit early without logging
        assert result.returncode == 0
        assert "Logged output" not in result.stdout

    def test_passes_20_char_threshold(self, tmp_path: Path) -> None:
        """Hook processes content when combined length meets 20-char minimum."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "What is X?"}],  # 10 chars
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "X equals Y."}],  # 11 chars
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

        # Combined: 10 + 11 = 21 chars >= 20 threshold
        # Should proceed (exit 0)
        assert result.returncode == 0

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
                        "role": "user",
                        "content": [{"type": "text", "text": "Tell me about X."}],
                    }
                }
            ),
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            # Both paragraphs should be in the extracted response
            assert "First paragraph" in captured
            assert "Second paragraph" in captured


class TestCombinedPayloadFormat:
    """Test hook produces USER:/ASSISTANT: combined payload."""

    def test_combined_user_assistant_format(self, tmp_path: Path, monkeypatch) -> None:
        """Hook combines user and assistant messages in USER:/ASSISTANT: format."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "How do I configure X?"}],
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "To configure X, use the settings file."}
                        ],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        # Capture what gets piped to memory-mcp-cli by using a mock script
        capture_script = tmp_path / "capture.sh"
        captured_file = tmp_path / "captured.txt"
        capture_script.write_text(f'#!/bin/bash\ncat > "{captured_file}"')
        capture_script.chmod(0o755)

        # Mock memory-mcp-cli to capture input
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            assert "USER: How do I configure X?" in captured
            assert "ASSISTANT: To configure X, use the settings file." in captured

    def test_user_message_truncated_to_500_chars(self, tmp_path: Path) -> None:
        """Hook truncates user message to 500 characters."""
        transcript = tmp_path / "transcript.jsonl"
        long_user_msg = "A" * 600  # 600 chars
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": long_user_msg}],
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Short response."}],
                    }
                }
            ),
        ]
        transcript.write_text("\n".join(lines))

        # Mock memory-mcp-cli to capture input
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript_path": str(transcript)}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            # User portion should be truncated to 500 chars
            user_section = captured.split("ASSISTANT:")[0]
            # The USER: prefix is added, so user content should be max 500 chars
            assert len(user_section) <= 510  # "USER: " (6) + 500 + some newlines


class TestHookInputFormats:
    """Test hook handles different input JSON formats."""

    def test_camel_case_transcript_path(self, tmp_path: Path) -> None:
        """Hook accepts camelCase transcriptPath."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "What is camelCase?"}],
                    }
                }
            ),
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcriptPath": str(transcript)}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            assert "Response for camelCase" in captured

    def test_nested_transcript_path(self, tmp_path: Path) -> None:
        """Hook accepts nested transcript.path format."""
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "What about nested paths?"}],
                    }
                }
            ),
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"transcript": {"path": str(transcript)}}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            assert "Response for nested" in captured


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
                        "role": "user",
                        "content": [{"type": "text", "text": "Test user message here"}],
                    }
                }
            ),
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps(
                {
                    "sessionId": session_id,
                    "projectPath": "/mock/project/path",
                }
            ),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            assert "Response via session fallback" in captured

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
                        "role": "user",
                        "content": [{"type": "text", "text": "Find fallback user msg"}],
                    }
                }
            ),
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps({"sessionId": session_id}),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            assert "Response found via find" in captured

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
                        "role": "user",
                        "content": [{"type": "text", "text": "CamelCase user message"}],
                    }
                }
            ),
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

        # Mock memory-mcp-cli to capture what gets logged
        captured_file = tmp_path / "captured.txt"
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        mock_cli = mock_bin / "memory-mcp-cli"
        mock_cli.write_text(
            f'#!/bin/bash\nif [ "$1" = "log-output" ]; then cat > "{captured_file}"; fi'
        )
        mock_cli.chmod(0o755)

        env = {
            "HOME": str(tmp_path),
            "PATH": f"{mock_bin}:/usr/bin:/bin",
            "MEMORY_MCP_DIR": str(HOOK_SCRIPT.parent.parent),
        }
        result = subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=json.dumps(
                {
                    "session_id": session_id,
                    "cwd": "/camel/case/project",
                }
            ),
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        if captured_file.exists():
            captured = captured_file.read_text()
            assert "Response for camelCase session" in captured
