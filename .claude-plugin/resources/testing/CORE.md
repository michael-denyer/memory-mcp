Run Phase 1 of Memory MCP testing: Core Memory Operations.

## Tests to Execute

**1.1 Remember** - Store test memories:
```
mcp__memory__remember("Test: FastAPI with async endpoints", memory_type="project", tags=["test", "tech-stack"])
mcp__memory__remember("Test: Always use uv run pytest -v", memory_type="pattern", tags=["test", "commands"])
mcp__memory__remember("Test: API rate limit 100 req/min", memory_type="reference", tags=["test", "api"])
```
Record the returned IDs for later tests.

**1.2 Recall** - Semantic search with different phrasings:
- `mcp__memory__recall("what framework for backend")` → should find FastAPI
- `mcp__memory__recall("how to run tests")` → should find pytest
- `mcp__memory__recall("request throttling")` → should find rate limit

Verify confidence levels returned.

**1.3 Recall Modes** - Compare precision vs exploratory:
- `mcp__memory__recall("testing", mode="precision")` → fewer, higher-confidence
- `mcp__memory__recall("testing", mode="exploratory")` → more results

**1.4 Recall by Tag**:
- `mcp__memory__recall_by_tag("test")` → should return all test memories

**1.5 Forget** - Delete one test memory:
- `mcp__memory__forget(memory_id)` and verify it's gone

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 1.1 Remember | ⬜ | IDs: |
| 1.2 Recall | ⬜ | |
| 1.3 Recall Modes | ⬜ | |
| 1.4 Recall by Tag | ⬜ | |
| 1.5 Forget | ⬜ | |

Proceed to Phase 2: Hot Cache.
