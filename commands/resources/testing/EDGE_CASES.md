Run Phase 14 of Memory MCP testing: Error Handling & Edge Cases.

## Tests to Execute

**14.1 Invalid IDs**:
- `mcp__memory__forget(memory_id=999999)` → should return error/not found
- `mcp__memory__promote(memory_id=-1)` → should handle gracefully
- `mcp__memory__get_related_memories(memory_id=0)` → should not crash

**14.2 Recall Edge Cases**:
- `mcp__memory__recall("", mode="precision")` → empty query handling
- `mcp__memory__recall("xyz", threshold=0.99)` → very high threshold, likely no results
- `mcp__memory__recall("test", limit=0)` → zero limit edge case
- `mcp__memory__recall("test", limit=1000)` → large limit handling

**14.3 Pagination Boundaries**:
- `mcp__memory__list_memories(offset=99999, limit=10)` → beyond data range
- `mcp__memory__list_memories(offset=-1)` → negative offset
- `mcp__memory__audit_history(limit=0)` → zero limit

**14.4 Link/Unlink Errors**:
- `mcp__memory__link_memories(id, id, "relates_to")` → self-link
- `mcp__memory__link_memories(999, 888, "relates_to")` → non-existent IDs
- `mcp__memory__unlink_memories(id_a, id_b)` → when no link exists

**14.5 Trust Boundaries**:
- `mcp__memory__validate_memory(id, boost=10.0)` → extreme boost (should cap at 1.0)
- `mcp__memory__invalidate_memory(id, penalty=10.0)` → extreme penalty (should floor at 0.0)

**14.6 Session Errors**:
- `mcp__memory__get_session("nonexistent-session-id")` → invalid session
- `mcp__memory__end_session("bad-id")` → non-existent session

**14.7 Mining Edge Cases**:
- `mcp__memory__run_mining(hours=0)` → zero hours
- `mcp__memory__approve_candidate(pattern_id=999)` → non-existent candidate

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 14.1 Invalid IDs | ⬜ | |
| 14.2 Recall Edge Cases | ⬜ | |
| 14.3 Pagination Boundaries | ⬜ | |
| 14.4 Link/Unlink Errors | ⬜ | |
| 14.5 Trust Boundaries | ⬜ | |
| 14.6 Session Errors | ⬜ | |
| 14.7 Mining Edge Cases | ⬜ | |

Proceed to Phase 15: Cleanup.
