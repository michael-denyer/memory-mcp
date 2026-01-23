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

# Setup logging directory and file
LOG_DIR="$HOME/.memory-mcp"
mkdir -p "$LOG_DIR"
HOOK_LOG="$LOG_DIR/hook.log"

# Log function with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$HOOK_LOG"
}

# Run memory-mcp-cli command, trying multiple methods
# Usage: run_cli "command" "args..."
# Returns the exit code of the command
run_cli() {
    local cmd="$1"
    shift
    local args="$*"

    if command -v uv &> /dev/null; then
        (cd "$MEMORY_MCP_DIR" && uv run memory-mcp-cli "$cmd" $args) 2>>"$HOOK_LOG"
    elif command -v memory-mcp-cli &> /dev/null; then
        memory-mcp-cli "$cmd" $args 2>>"$HOOK_LOG"
    elif [ -x "$HOME/.local/bin/memory-mcp-cli" ]; then
        "$HOME/.local/bin/memory-mcp-cli" "$cmd" $args 2>>"$HOOK_LOG"
    else
        log_msg "ERROR: No memory-mcp-cli found (uv, memory-mcp-cli, ~/.local/bin)"
        return 1
    fi
}

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

# Extract session and project info (always, for passing to CLI)
SESSION_ID=$(printf '%s' "$HOOK_INPUT" | jq -r '(.session_id // .sessionId // .session.id // empty)' 2>/dev/null || true)
PROJECT_PATH=$(printf '%s' "$HOOK_INPUT" | jq -r '(.project_path // .projectPath // .project.path // .cwd // .workspace_path // .rootPath // empty)' 2>/dev/null || true)

TRANSCRIPT_PATH=$(printf '%s' "$HOOK_INPUT" | jq -r '(.transcript_path // .transcriptPath // .transcript.path // empty)' 2>/dev/null || true)

# Fallback: derive transcript path from session + project info if not provided directly
if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
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

# Extract the last assistant and user messages from the transcript
# Transcript format: JSONL with {message: {role: "...", content: [{type: "text", text: "..."}]}}
TRANSCRIPT_TAIL=$(tail -200 "$TRANSCRIPT_PATH" 2>/dev/null)

# Extract last assistant message with text content
LAST_RESPONSE=$(printf '%s' "$TRANSCRIPT_TAIL" | \
    jq -rs '
        [.[] | select(.message.role? == "assistant") | select(.message.content[]?.type == "text")]
        | last
        | .message.content
        | map(select(.type == "text") | .text)
        | join("\n")
        | if . == "" then empty else . end
    ' 2>/dev/null)

# Extract last user message (truncated to 500 chars to avoid noise)
LAST_USER_MSG=$(printf '%s' "$TRANSCRIPT_TAIL" | \
    jq -rs '
        [.[] | select(.message.role? == "user") | select(.message.content[]?.type == "text")]
        | last
        | .message.content
        | map(select(.type == "text") | .text)
        | join("\n")
        | if . == "" then empty else .[0:500] end
    ' 2>/dev/null)

if [ -z "$LAST_RESPONSE" ] || [ "$LAST_RESPONSE" = "null" ]; then
    # No response found, exit silently
    exit 0
fi

# Skip if combined content is too short (lowered from 50 to 20 for short constraints)
COMBINED_LEN=$(( ${#LAST_RESPONSE} + ${#LAST_USER_MSG} ))
if [ $COMBINED_LEN -lt 20 ]; then
    exit 0
fi

# Combine user message with assistant response for richer mining context
if [ -n "$LAST_USER_MSG" ] && [ "$LAST_USER_MSG" != "null" ]; then
    LAST_RESPONSE="USER: $LAST_USER_MSG

ASSISTANT: $LAST_RESPONSE"
fi

# Build CLI arguments from hook input
CLI_ARGS=""
if [ -n "$PROJECT_PATH" ]; then
    CLI_ARGS="$CLI_ARGS --project-id=$PROJECT_PATH"
fi
if [ -n "$SESSION_ID" ]; then
    CLI_ARGS="$CLI_ARGS --session-id=$SESSION_ID"
fi

# Log the response using memory-mcp-cli
log_msg "Processing response (${#LAST_RESPONSE} chars) project=${PROJECT_PATH:-none} session=${SESSION_ID:-none}"

if ! echo "$LAST_RESPONSE" | run_cli log-output $CLI_ARGS; then
    log_msg "ERROR: log-output failed with exit code $?"
fi

# Run mining to extract and store patterns as memories (non-critical)
run_cli run-mining --hours 1 $CLI_ARGS >> "$HOOK_LOG" 2>&1 || true

log_msg "Hook completed successfully"
