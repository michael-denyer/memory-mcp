Run Phases 4-5 of Memory MCP testing: Trust Management & Contradictions.

## Phase 4: Trust Management

**4.1 Validate**:
- `mcp__memory__validate_memory(memory_id, reason="used_correctly")`
- Check trust_score increased

**4.2 Invalidate**:
- `mcp__memory__invalidate_memory(memory_id, reason="outdated", note="Testing invalidation")`
- Check trust_score decreased

**4.3 Trust History**:
- `mcp__memory__get_trust_history(memory_id)` → shows all changes

## Phase 5: Contradiction Detection

**5.1 Create Conflicting Memories**:
```
mcp__memory__remember("Test: Timeout is 30 seconds") → ID: X
mcp__memory__remember("Test: Timeout is 60 seconds") → ID: Y
```

**5.2 Find & Mark**:
- `mcp__memory__find_contradictions(X)` → should suggest Y
- `mcp__memory__mark_contradiction(X, Y)`
- `mcp__memory__get_contradictions()` → pair listed

**5.3 Resolve**:
- `mcp__memory__resolve_contradiction(X, Y, keep_id=X, resolution="supersedes")`
- Verify X supersedes Y, Y's trust reduced

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 4.1 Validate | ⬜ | |
| 4.2 Invalidate | ⬜ | |
| 4.3 Trust History | ⬜ | |
| 5.1 Contradictions | ⬜ | IDs: X=, Y= |
| 5.2 Mark/Find | ⬜ | |
| 5.3 Resolve | ⬜ | |

Proceed to Phase 6: Sessions.
