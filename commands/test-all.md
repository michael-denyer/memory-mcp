---
description: Comprehensive interactive testing of all Memory MCP features
---

# Memory MCP Live Testing

Run `/memory-mcp:test-all` to start a guided testing session. Walk through each phase, execute tests, and track results.

## Execution Modes

Ask user which approach:
1. **Full suite** - Run all 12 phases sequentially with pauses between
2. **Specific phase** - Run just one phase (1-12)
3. **Smoke test** - Quick 2-minute sanity check

## Quick Smoke Test

For a 2-minute sanity check:
```
mcp__memory__remember("Smoke test", tags=["smoke"])  # → get ID
mcp__memory__recall("smoke")
mcp__memory__promote(id)
mcp__memory__hot_cache_status()
mcp__memory__demote(id)
mcp__memory__forget(id)
mcp__memory__memory_stats()
```

## Test Phases

| Phase | Name | Tests |
|-------|------|-------|
| 1 | [Core Operations](resources/testing/CORE.md) | remember, recall, forget |
| 2 | [Hot Cache](resources/testing/HOT_CACHE.md) | promote, demote, pin, unpin |
| 3 | [Knowledge Graph](resources/testing/GRAPH.md) | link, unlink, traverse |
| 4-5 | [Trust & Contradictions](resources/testing/TRUST.md) | validate, invalidate, resolve |
| 6 | [Sessions](resources/testing/SESSIONS.md) | episodic memory, summarize |
| 7 | [Mining](resources/testing/MINING.md) | log_output, run_mining, approve |
| 8 | [Seeding](resources/testing/SEEDING.md) | seed_from_text, bootstrap |
| 9-10 | [Predictive & Quality](resources/testing/PREDICTIVE.md) | predict_next, mark_used |
| 11 | [Maintenance](resources/testing/MAINTENANCE.md) | db_info, cleanup, consolidate |
| 12 | [Resources](resources/testing/RESOURCES.md) | hot-cache, working-set |
| 14 | [Edge Cases](resources/testing/EDGE_CASES.md) | error handling |
| 15 | [Cleanup](resources/testing/CLEANUP.md) | remove test data |

## Test Tracking

Track results as you go:

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Core | ⬜ | |
| 2. Hot Cache | ⬜ | |
| 3. Graph | ⬜ | |
| 4-5. Trust | ⬜ | |
| 6. Sessions | ⬜ | |
| 7. Mining | ⬜ | |
| 8. Seeding | ⬜ | |
| 9-10. Predictive | ⬜ | |
| 11. Maintenance | ⬜ | |
| 12. Resources | ⬜ | |
| 14. Edge Cases | ⬜ | |
| 15. Cleanup | ⬜ | |

## Execution Instructions

When running a phase:
1. Read the corresponding resource file for detailed test cases
2. Execute each test, showing tool calls and results
3. Update the tracking table after each test
4. Pause between phases to let user review
5. Note any failures or unexpected behavior

After all tests complete, prompt user to run `/compact` to test conversation resumption.
