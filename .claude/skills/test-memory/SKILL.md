---
name: test-memory
description: Comprehensive interactive testing of all Memory MCP features
disable-model-invocation: true
---

# Memory MCP Live Testing Skill

Run `/test-memory` to start a guided testing session. Walk through each phase, execute tests, and track results.

## Phases

---

### Phase 1: Core Memory Operations

**1.1 Remember** - Store test memories:
```
remember("Test: FastAPI with async endpoints", memory_type="project", tags=["test", "tech-stack"])
remember("Test: Always use uv run pytest -v", memory_type="pattern", tags=["test", "commands"])
remember("Test: API rate limit 100 req/min", memory_type="reference", tags=["test", "api"])
```
Record the returned IDs for later tests.

**1.2 Recall** - Semantic search (use different phrasings):
- "what framework for backend" → should find FastAPI
- "how to run tests" → should find pytest
- "request throttling" → should find rate limit

Verify confidence levels returned.

**1.3 Recall Modes** - Compare precision vs exploratory:
- `recall("testing", mode="precision")` → fewer, higher-confidence
- `recall("testing", mode="exploratory")` → more results

**1.4 Recall by Tag**:
- `recall_by_tag("test")` → should return all test memories

**1.5 Forget** - Delete one test memory and verify it's gone.

---

### Phase 2: Hot Cache Mechanics

**2.1 Manual Promotion**:
- `promote(memory_id)` one of the test memories
- `hot_cache_status()` → verify it appears

**2.2 Manual Demotion**:
- `demote(memory_id)`
- Verify removed from hot cache but still recallable

**2.3 Pin/Unpin**:
- `pin(memory_id)` → pinned_count increases
- `unpin(memory_id)` → pinned_count decreases

**2.4 Auto-Promotion** (optional - requires multiple recalls):
- Create a memory and recall it 3+ times
- Check if auto-promoted

---

### Phase 3: Knowledge Graph

**3.1 Create Linked Memories**:
```
remember("Test: Database uses PostgreSQL") → ID: A
remember("Test: pgvector for embeddings") → ID: B
remember("Test: Vector search needs pgvector") → ID: C
link_memories(A, B, "relates_to")
link_memories(B, C, "depends_on")
```

**3.2 Traverse Graph**:
- `get_related_memories(B)` → should show A and C
- `get_related_memories(A, direction="outgoing")` → should show B

**3.3 Multi-Hop Recall**:
- `recall("PostgreSQL", expand_relations=true)` → should include related memories

**3.4 Unlink**:
- `unlink_memories(A, B)`
- Verify relationship removed

---

### Phase 4: Trust Management

**4.1 Validate**:
- `validate_memory(id, reason="used_correctly")`
- Check trust_score increased

**4.2 Invalidate**:
- `invalidate_memory(id, reason="outdated", note="Testing invalidation")`
- Check trust_score decreased

**4.3 Trust History**:
- `get_trust_history(memory_id)` → shows all changes

---

### Phase 5: Contradiction Detection

**5.1 Create Conflicting Memories**:
```
remember("Test: Timeout is 30 seconds") → ID: X
remember("Test: Timeout is 60 seconds") → ID: Y
```

**5.2 Find & Mark**:
- `find_contradictions(X)` → should suggest Y
- `mark_contradiction(X, Y)`
- `get_contradictions()` → pair listed

**5.3 Resolve**:
- `resolve_contradiction(X, Y, keep_id=X, resolution="supersedes")`
- Verify X supersedes Y, Y's trust reduced

---

### Phase 6: Sessions & Episodic Memory

**6.1 Check Sessions**:
- `get_sessions(limit=5)`

**6.2 Episodic Memories**:
```
remember("Test: Debugging auth today", memory_type="episodic")
remember("Test: Found token bug", memory_type="episodic")
```

**6.3 Session Topic**:
- `set_session_topic(session_id, "Testing session")`

**6.4 End Session** (optional - ends current session):
- `end_session(session_id, promote_top=true)`

---

### Phase 7: Pattern Mining

**7.1 Log Output**:
```
log_output("import pandas as pd")
log_output("import numpy as np")
log_output("uv run pytest -v")
```

**7.2 Run Mining**:
- `run_mining(hours=1)`
- `mining_status()`

**7.3 Review Candidates**:
- `review_candidates()`

**7.4 Approve/Reject** (if candidates exist):
- `approve_candidate(id)` or `reject_candidate(id)`

---

### Phase 8: Maintenance

**8.1 Stats & Observability**:
- `memory_stats()`
- `hot_cache_status()`
- `metrics_status()`

**8.2 Maintenance**:
- `db_maintenance()`
- `validate_embeddings()`

**8.3 Consolidation Preview**:
- `preview_consolidation()`

**8.4 Audit History**:
- `audit_history(limit=10)`

---

### Phase 9: Cleanup

After testing, clean up test data:
- `forget()` all memories tagged with "test"
- Or use `recall_by_tag("test")` to find them first

---

## Test Tracking

Track results as you go:

| Phase | Test | Status | Notes |
|-------|------|--------|-------|
| 1.1 | Remember | ⬜ | IDs: |
| 1.2 | Recall | ⬜ | |
| 1.3 | Recall Modes | ⬜ | |
| 1.4 | Recall by Tag | ⬜ | |
| 1.5 | Forget | ⬜ | |
| 2.1 | Promotion | ⬜ | |
| 2.2 | Demotion | ⬜ | |
| 2.3 | Pin/Unpin | ⬜ | |
| 3.1 | Link Memories | ⬜ | IDs: A=, B=, C= |
| 3.2 | Get Related | ⬜ | |
| 3.3 | Multi-Hop | ⬜ | |
| 3.4 | Unlink | ⬜ | |
| 4.1 | Validate | ⬜ | |
| 4.2 | Invalidate | ⬜ | |
| 4.3 | Trust History | ⬜ | |
| 5.1 | Contradictions | ⬜ | IDs: X=, Y= |
| 5.2 | Mark/Find | ⬜ | |
| 5.3 | Resolve | ⬜ | |
| 6.1 | Sessions | ⬜ | |
| 6.2 | Episodic | ⬜ | |
| 6.3 | Session Topic | ⬜ | |
| 7.1 | Log Output | ⬜ | |
| 7.2 | Run Mining | ⬜ | |
| 7.3 | Review | ⬜ | |
| 8.1 | Stats | ⬜ | |
| 8.2 | Maintenance | ⬜ | |
| 8.3 | Consolidation | ⬜ | |
| 8.4 | Audit | ⬜ | |
| 9 | Cleanup | ⬜ | |

## Quick Smoke Test

For a 2-minute sanity check:
1. `remember("Smoke test", tags=["smoke"])`
2. `recall("smoke")`
3. `promote(id)` → `hot_cache_status()`
4. `demote(id)` → `forget(id)`
5. `memory_stats()`

---

## Execution Mode

When running this skill:
1. Ask user which phase to start with (or start from Phase 1)
2. Execute each test, showing tool calls and results
3. Update the tracking table after each test
4. Pause between phases to let user review
5. Note any failures or unexpected behavior
6. Offer to skip phases or run specific tests
