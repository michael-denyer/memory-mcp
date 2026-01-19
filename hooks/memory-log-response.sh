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

# Check for required dependencies
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed." >&2
    echo "Install with: brew install jq (macOS) or apt install jq (Linux)" >&2
    exit 1
fi

# Extract transcript path from hook input (stdin)
TRANSCRIPT_PATH=$(jq -r '.transcript_path // empty')

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    # No transcript available, exit silently
    exit 0
fi

# Find memory-mcp directory (script location or environment variable)
if [ -z "$MEMORY_MCP_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MEMORY_MCP_DIR="$(dirname "$SCRIPT_DIR")"
fi

# Extract the last assistant message from the transcript
# The transcript is JSONL format with message objects
LAST_RESPONSE=$(tail -100 "$TRANSCRIPT_PATH" 2>/dev/null | \
    jq -s '[.[] | select(.type == "assistant" or .role == "assistant")] | last | .content // .message // empty' 2>/dev/null | \
    jq -r 'if type == "array" then [.[] | select(.type == "text") | .text] | join("\n") else . end' 2>/dev/null)

if [ -z "$LAST_RESPONSE" ] || [ "$LAST_RESPONSE" = "null" ]; then
    # No response found, exit silently
    exit 0
fi

# Skip if response is too short (likely just acknowledgments)
if [ ${#LAST_RESPONSE} -lt 50 ]; then
    exit 0
fi

# Log the response using memory-mcp-cli
# Use uv run to ensure we're using the right environment
cd "$MEMORY_MCP_DIR"
echo "$LAST_RESPONSE" | uv run memory-mcp-cli log-output 2>/dev/null || true
