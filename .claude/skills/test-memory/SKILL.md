---
name: test-memory
description: Comprehensive interactive testing of all Memory MCP features
allowed-tools: "mcp__memory__*"
---

# Memory MCP Live Testing Skill

Run `/test-memory` to start a guided testing session. Walk through each phase, execute tests, and track results.

## Prerequisites

Install memory-mcp from the marketplace:

```bash
# Add the marketplace
/plugin marketplace add michael-denyer/memory-mcp

# Install the plugin
/plugin install memory-mcp@michael-denyer/memory-mcp
```

Or install directly:
```bash
/plugin install github:michael-denyer/memory-mcp
```

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

**6.4 Summarize Session**:
- `summarize_session(session_id)` → structured summary with:
  - Decisions (choices made and rationale)
  - Insights (lessons, antipatterns, landmines, constraints)
  - Action Items (todos, bugs, tasks)
  - Context (background, conventions, preferences, architecture)
- Use before `end_session()` to review what will be promoted

**6.5 End Session** (optional - ends current session):
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

### Phase 8: Seeding & Bootstrap

**8.1 Seed from Text**:
```
seed_from_text("- Item one\n- Item two\n- Item three", memory_type="project")
```
Verify 3 memories created.

**8.2 Seed from File** (creates temp file):
- `seed_from_file(file_path, memory_type="reference")`

**8.3 Bootstrap Project**:
- `bootstrap_project(root_path=".", promote_to_hot=false)`
- Verify it finds CLAUDE.md, README.md, etc.

---

### Phase 9: Predictive Cache

**9.1 Check Status**:
- `predictive_cache_status()` → shows if enabled

**9.2 Access Patterns**:
- `access_patterns(limit=5)` → learned patterns

**9.3 Predict Next** (needs existing access history):
- `predict_next(memory_id)` → predicted memories

**9.4 Warm Cache**:
- `warm_cache(memory_id)` → pre-promote predicted

---

### Phase 10: Retrieval Quality

**10.1 Mark Memory Used**:
- `mark_memory_used(memory_id, feedback="helpful")`

**10.2 Retrieval Stats**:
- `retrieval_quality_stats()` → global stats
- `retrieval_quality_stats(memory_id=X)` → per-memory

---

### Phase 11: Maintenance & DB Info

**11.1 Stats & Observability**:
- `memory_stats()`
- `hot_cache_status()`
- `metrics_status()`

**11.2 Database Info**:
- `db_info()` → schema version, size
- `embedding_info()` → provider, cache info

**11.3 Maintenance Operations**:
- `db_maintenance()`
- `validate_embeddings()`
- `run_cleanup()` → comprehensive cleanup

**11.4 Consolidation**:
- `preview_consolidation()` → dry run
- `run_consolidation(dry_run=true)` → preview
- `run_consolidation(dry_run=false)` → actual merge (careful!)

**11.5 Audit History**:
- `audit_history(limit=10)`

**11.6 Vector Rebuild** (use with caution - rebuilds all embeddings):
- `db_rebuild_vectors(batch_size=100)` → re-embed all memories
- Use when: switching embedding models, fixing dimension mismatches, recovering from corruption

---

### Phase 12: MCP Resources

**12.1 Hot Cache Resource**:
Read `memory://hot-cache` directly (auto-injected to Claude):
- Contains all promoted memories for instant recall
- Verify contents match `hot_cache_status()` items

**12.2 Working Set Resource**:
Read `memory://working-set` directly:
- Session-aware active context (~10 items max)
- Combines: recently recalled, predicted next, top salience
- Verify smaller/more focused than hot-cache

**12.3 Project Context Resource**:
Read `memory://project-context` directly:
- Project-scoped memories for current working directory
- Should filter to current project only

---

### Phase 13: Additional Tools

**13.1 List Memories**:
- `list_memories(limit=5)` → paginated browse
- `list_memories(memory_type="pattern")` → filtered
- `list_memories(offset=5, limit=5)` → pagination

**13.2 Recall with Fallback**:
- `recall_with_fallback("query")` → tries patterns → project → all

**13.3 Relationship Stats**:
- `relationship_stats()` → knowledge graph overview

**13.4 Session Details**:
- `get_session(session_id)` → specific session
- `get_session_memories(session_id)` → memories from session
- `cross_session_patterns()` → patterns across sessions

---

### Phase 14: Error Handling & Edge Cases

**14.1 Invalid IDs**:
- `forget(memory_id=999999)` → should return error/not found
- `promote(memory_id=-1)` → should handle gracefully
- `get_related_memories(memory_id=0)` → should not crash

**14.2 Recall Edge Cases**:
- `recall("", mode="precision")` → empty query handling
- `recall("xyz", threshold=0.99)` → very high threshold, likely no results
- `recall("test", limit=0)` → zero limit edge case
- `recall("test", limit=1000)` → large limit handling

**14.3 Pagination Boundaries**:
- `list_memories(offset=99999, limit=10)` → beyond data range
- `list_memories(offset=-1)` → negative offset
- `audit_history(limit=0)` → zero limit

**14.4 Link/Unlink Errors**:
- `link_memories(id, id, "relates_to")` → self-link
- `link_memories(999, 888, "relates_to")` → non-existent IDs
- `unlink_memories(id_a, id_b)` → when no link exists

**14.5 Trust Boundaries**:
- `validate_memory(id, boost=10.0)` → extreme boost (should cap at 1.0)
- `invalidate_memory(id, penalty=10.0)` → extreme penalty (should floor at 0.0)

**14.6 Session Errors**:
- `get_session("nonexistent-session-id")` → invalid session
- `end_session("bad-id")` → non-existent session

**14.7 Mining Edge Cases**:
- `run_mining(hours=0)` → zero hours
- `approve_candidate(pattern_id=999)` → non-existent candidate

---

### Phase 15: Cleanup

After testing, clean up test data:
- `forget()` all memories tagged with "test"
- Or use `recall_by_tag("test")` to find them first

---

### Phase 16: Compact Conversation

**16.1 Run Compact**:
After completing all tests, prompt the user to run `/compact` to:
- Reduce conversation context size
- Test that the session can be resumed after compaction
- Verify memories persist across conversation summarization

Note: `/compact` is a user-initiated command and cannot be run programmatically by the assistant.

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
| 6.4 | Summarize Session | ⬜ | |
| 7.1 | Log Output | ⬜ | |
| 7.2 | Run Mining | ⬜ | |
| 7.3 | Review | ⬜ | |
| 8.1 | Seed from Text | ⬜ | |
| 8.2 | Seed from File | ⬜ | |
| 8.3 | Bootstrap | ⬜ | |
| 9.1 | Predictive Status | ⬜ | |
| 9.2 | Access Patterns | ⬜ | |
| 9.3 | Predict Next | ⬜ | |
| 9.4 | Warm Cache | ⬜ | |
| 10.1 | Mark Used | ⬜ | |
| 10.2 | Retrieval Stats | ⬜ | |
| 11.1 | Stats | ⬜ | |
| 11.2 | DB Info | ⬜ | |
| 11.3 | Maintenance | ⬜ | |
| 11.4 | Consolidation | ⬜ | |
| 11.5 | Audit | ⬜ | |
| 11.6 | Vector Rebuild | ⬜ | |
| 12.1 | Hot Cache Resource | ⬜ | |
| 12.2 | Working Set Resource | ⬜ | |
| 12.3 | Project Context Resource | ⬜ | |
| 13.1 | List Memories | ⬜ | |
| 13.2 | Recall Fallback | ⬜ | |
| 13.3 | Relationship Stats | ⬜ | |
| 13.4 | Session Details | ⬜ | |
| 14.1 | Invalid IDs | ⬜ | |
| 14.2 | Recall Edge Cases | ⬜ | |
| 14.3 | Pagination Boundaries | ⬜ | |
| 14.4 | Link/Unlink Errors | ⬜ | |
| 14.5 | Trust Boundaries | ⬜ | |
| 14.6 | Session Errors | ⬜ | |
| 14.7 | Mining Edge Cases | ⬜ | |
| 15 | Cleanup | ⬜ | |
| 16.1 | Compact Conversation | ⬜ | |

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
