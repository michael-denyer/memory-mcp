Run Phase 6 of Memory MCP testing: Sessions & Episodic Memory.

## Tests to Execute

**6.1 Check Sessions**:
- `mcp__memory__get_sessions(limit=5)`

**6.2 Episodic Memories**:
```
mcp__memory__remember("Test: Debugging auth today", memory_type="episodic")
mcp__memory__remember("Test: Found token bug", memory_type="episodic")
```

**6.3 Session Topic**:
- `mcp__memory__set_session_topic(session_id, "Testing session")`

**6.4 Summarize Session**:
- `mcp__memory__summarize_session(session_id)` → structured summary with:
  - Decisions (choices made and rationale)
  - Insights (lessons, antipatterns, landmines, constraints)
  - Action Items (todos, bugs, tasks)
  - Context (background, conventions, preferences, architecture)

**6.5 Session Details**:
- `mcp__memory__get_session(session_id)` → specific session
- `mcp__memory__get_session_memories(session_id)` → memories from session
- `mcp__memory__cross_session_patterns()` → patterns across sessions

**6.6 End Session** (optional - ends current session):
- `mcp__memory__end_session(session_id, promote_top=true)`

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 6.1 Sessions | ⬜ | |
| 6.2 Episodic | ⬜ | |
| 6.3 Session Topic | ⬜ | |
| 6.4 Summarize Session | ⬜ | |
| 6.5 Session Details | ⬜ | |
| 6.6 End Session | ⬜ | |

Proceed to Phase 7: Mining.
