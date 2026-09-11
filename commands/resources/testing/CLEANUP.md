Run Phase 15 of Memory MCP testing: Cleanup.

## Cleanup Steps

**15.1 Find Test Memories**:
- `mcp__memory__recall_by_tag("test")` → list all test-tagged memories

**15.2 Delete Test Memories**:
For each memory found with "test" tag:
- `mcp__memory__forget(memory_id)`

**15.3 Verify Cleanup**:
- `mcp__memory__recall_by_tag("test")` → should return empty
- `mcp__memory__memory_stats()` → verify counts reduced

## After Cleanup

Prompt user to run `/compact` to:
- Reduce conversation context size
- Test that the session can be resumed after compaction
- Verify memories persist across conversation summarization

Note: `/compact` is a user-initiated command.
