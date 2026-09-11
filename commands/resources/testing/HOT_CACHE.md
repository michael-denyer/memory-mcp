Run Phase 2 of Memory MCP testing: Hot Cache Mechanics.

## Prerequisites
Run `/memory-mcp:test-core` first to create test memories.

## Tests to Execute

**2.1 Manual Promotion**:
- `mcp__memory__promote(memory_id)` one of the test memories
- `mcp__memory__hot_cache_status()` → verify it appears

**2.2 Manual Demotion**:
- `mcp__memory__demote(memory_id)`
- Verify removed from hot cache but still recallable via `mcp__memory__recall`

**2.3 Pin/Unpin**:
- `mcp__memory__pin(memory_id)` → pinned_count increases
- `mcp__memory__unpin(memory_id)` → pinned_count decreases

**2.4 Auto-Promotion** (optional - requires multiple recalls):
- Create a memory and recall it 3+ times
- Check if auto-promoted to hot cache

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 2.1 Promotion | ⬜ | |
| 2.2 Demotion | ⬜ | |
| 2.3 Pin/Unpin | ⬜ | |
| 2.4 Auto-Promotion | ⬜ | |

Proceed to Phase 3: Knowledge Graph.
