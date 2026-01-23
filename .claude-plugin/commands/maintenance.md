---
description: Run database maintenance
argument-hint: [cleanup|vacuum|validate]
---

Perform database maintenance operations.

**Subcommands:**
- No args or `vacuum`: Run basic maintenance (`mcp__memory__db_maintenance`)
- `cleanup`: Run comprehensive cleanup (`mcp__memory__run_cleanup`)
- `validate`: Check embedding model compatibility (`mcp__memory__validate_embeddings`)

**db_maintenance** performs:
- VACUUM to reclaim unused space
- ANALYZE to update query planner stats
- Auto-demote stale hot memories

**run_cleanup** additionally:
- Expires old mining patterns
- Deletes old output logs
- Applies type-specific retention policies

**validate_embeddings** checks if embedding model changed since database was created.
