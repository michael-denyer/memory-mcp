Run Phase 11 of Memory MCP testing: Maintenance & DB Info.

## Tests to Execute

**11.1 Stats & Observability**:
- `mcp__memory__memory_stats()`
- `mcp__memory__hot_cache_status()`
- `mcp__memory__metrics_status()`

**11.2 Database Info**:
- `mcp__memory__db_info()` → schema version, size
- `mcp__memory__embedding_info()` → provider, cache info

**11.3 Maintenance Operations**:
- `mcp__memory__db_maintenance()`
- `mcp__memory__validate_embeddings()`
- `mcp__memory__run_cleanup()` → comprehensive cleanup

**11.4 Consolidation**:
- `mcp__memory__preview_consolidation()` → dry run
- `mcp__memory__run_consolidation(dry_run=true)` → preview
- `mcp__memory__run_consolidation(dry_run=false)` → actual merge (careful!)

**11.5 Audit History**:
- `mcp__memory__audit_history(limit=10)`

**11.6 Vector Rebuild** (use with caution):
- `mcp__memory__db_rebuild_vectors(batch_size=100)` → re-embed all memories
- Use when: switching embedding models, fixing dimension mismatches

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 11.1 Stats | ⬜ | |
| 11.2 DB Info | ⬜ | |
| 11.3 Maintenance | ⬜ | |
| 11.4 Consolidation | ⬜ | |
| 11.5 Audit | ⬜ | |
| 11.6 Vector Rebuild | ⬜ | |

Proceed to Phase 12: Resources.
