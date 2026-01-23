Run Phase 3 of Memory MCP testing: Knowledge Graph.

## Tests to Execute

**3.1 Create Linked Memories**:
```
mcp__memory__remember("Test: Database uses PostgreSQL") → ID: A
mcp__memory__remember("Test: pgvector for embeddings") → ID: B
mcp__memory__remember("Test: Vector search needs pgvector") → ID: C
mcp__memory__link_memories(A, B, "relates_to")
mcp__memory__link_memories(B, C, "depends_on")
```

**3.2 Traverse Graph**:
- `mcp__memory__get_related_memories(B)` → should show A and C
- `mcp__memory__get_related_memories(A, direction="outgoing")` → should show B

**3.3 Multi-Hop Recall**:
- `mcp__memory__recall("PostgreSQL", expand_relations=true)` → should include related memories

**3.4 Unlink**:
- `mcp__memory__unlink_memories(A, B)`
- Verify relationship removed with `mcp__memory__get_related_memories(A)`

**3.5 Relationship Stats**:
- `mcp__memory__relationship_stats()` → knowledge graph overview

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 3.1 Link Memories | ⬜ | IDs: A=, B=, C= |
| 3.2 Get Related | ⬜ | |
| 3.3 Multi-Hop | ⬜ | |
| 3.4 Unlink | ⬜ | |
| 3.5 Relationship Stats | ⬜ | |

Proceed to Phase 4-5: Trust & Contradictions.
