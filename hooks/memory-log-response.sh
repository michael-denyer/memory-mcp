#!/bin/bash
# Claude Code hook script for logging assistant responses to memory-mcp
#
# This script is called by Claude Code's "Stop" hook when Claude finishes responding.
# It extracts the assistant's last response from the transcript and logs it for
# pattern mining.
#
# Installation:
#   Add to ~/.claude/settings.json:
#   {
#     "hooks": {
#       "Stop": [
#         {
#           "hooks": [
#             {
#               "type": "command",
#               "command": "/path/to/memory-mcp/hooks/memory-log-response.sh"
#             }
#           ]
#         }
#       ]
#     }
#   }
#
# Environment:
#   MEMORY_MCP_DIR: Path to memory-mcp installation (auto-detected if not set)

set -e

# Ensure common user-level bin paths are available when run from VS Code
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# Check for required dependencies
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed." >&2
    echo "Install with: brew install jq (macOS) or apt install jq (Linux)" >&2
    exit 1
fi

# Extract transcript path from hook input (stdin)
# Read stdin once so we can parse it multiple ways
HOOK_INPUT="$(cat)"
if [ -z "$HOOK_INPUT" ]; then
    exit 0
fi

TRANSCRIPT_PATH=$(printf '%s' "$HOOK_INPUT" | jq -r '(.transcript_path // .transcriptPath // .transcript.path // empty)' 2>/dev/null || true)

# Fallback: derive transcript path from session + project info if provided
if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    SESSION_ID=$(printf '%s' "$HOOK_INPUT" | jq -r '(.session_id // .sessionId // .session.id // empty)' 2>/dev/null || true)
    PROJECT_PATH=$(printf '%s' "$HOOK_INPUT" | jq -r '(.project_path // .projectPath // .project.path // .cwd // .workspace_path // .rootPath // empty)' 2>/dev/null || true)

    if [ -n "$SESSION_ID" ]; then
        if [ -n "$PROJECT_PATH" ]; then
            PROJECT_SLUG="$(printf '%s' "$PROJECT_PATH" | sed 's#/#-#g')"
            CANDIDATE="$HOME/.claude/projects/$PROJECT_SLUG/$SESSION_ID.jsonl"
            if [ -f "$CANDIDATE" ]; then
                TRANSCRIPT_PATH="$CANDIDATE"
            fi
        fi

        if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
            CANDIDATE="$(find "$HOME/.claude/projects" -name "$SESSION_ID.jsonl" -print -quit 2>/dev/null || true)"
            if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE" ]; then
                TRANSCRIPT_PATH="$CANDIDATE"
            fi
        fi
    fi
fi

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    # No transcript available, exit silently
    exit 0
fi

# Find memory-mcp directory (script location or environment variable)
if [ -z "$MEMORY_MCP_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MEMORY_MCP_DIR="$(dirname "$SCRIPT_DIR")"
fi

# Extract the last assistant message with text content from the transcript
# Transcript format: JSONL with {message: {role: "assistant", content: [{type: "text", text: "..."}]}}
LAST_RESPONSE=$(tail -200 "$TRANSCRIPT_PATH" 2>/dev/null | \
    jq -rs '
        [.[] | select(.message.role? == "assistant") | select(.message.content[]?.type == "text")]
        | last
        | .message.content
        | map(select(.type == "text") | .text)
        | join("\n")
        | if . == "" then empty else . end
    ' 2>/dev/null)

if [ -z "$LAST_RESPONSE" ] || [ "$LAST_RESPONSE" = "null" ]; then
    # No response found, exit silently
    exit 0
fi

# Skip if response is too short (likely just acknowledgments)
if [ ${#LAST_RESPONSE} -lt 50 ]; then
    exit 0
fi

# Log the response using memory-mcp-cli
# Prefer uv if available, otherwise fall back to direct CLI
cd "$MEMORY_MCP_DIR"
LOG_CMD=()
if command -v uv &> /dev/null; then
    LOG_CMD=(uv run memory-mcp-cli log-output)
elif command -v memory-mcp-cli &> /dev/null; then
    LOG_CMD=(memory-mcp-cli log-output)
elif [ -x "$HOME/.local/bin/memory-mcp-cli" ]; then
    LOG_CMD=("$HOME/.local/bin/memory-mcp-cli" log-output)
fi

if [ ${#LOG_CMD[@]} -eq 0 ]; then
    exit 0
fi

echo "$LAST_RESPONSE" | "${LOG_CMD[@]}" 2>/dev/null || true

# Run mining to extract and store patterns as memories
# This runs in the background to avoid blocking Claude Code
MINE_CMD=()
if command -v uv &> /dev/null; then
    MINE_CMD=(uv run memory-mcp-cli run-mining --hours 1)
elif command -v memory-mcp-cli &> /dev/null; then
    MINE_CMD=(memory-mcp-cli run-mining --hours 1)
elif [ -x "$HOME/.local/bin/memory-mcp-cli" ]; then
    MINE_CMD=("$HOME/.local/bin/memory-mcp-cli" run-mining --hours 1)
fi

if [ ${#MINE_CMD[@]} -gt 0 ]; then
    # Run mining in foreground so output is visible
    LOG_FILE="${MEMORY_MCP_DIR}/.mining-hook.log"
    "${MINE_CMD[@]}" >> "$LOG_FILE" 2>&1
fi
