---
description: View and manage the hot cache
argument-hint: [promote|demote|pin|unpin] [memory-id]
---

Manage the hot cache for instant memory recall.

**Subcommands:**
- No args: Show current hot cache status
- `promote <id>`: Add memory to hot cache
- `demote <id>`: Remove memory from hot cache (keeps in cold storage)
- `pin <id>`: Pin memory to prevent auto-eviction
- `unpin <id>`: Allow memory to be auto-evicted

Use appropriate MCP tools:
- `mcp__memory__hot_cache_status` for status
- `mcp__memory__promote` to add to hot cache
- `mcp__memory__demote` to remove from hot cache
- `mcp__memory__pin` to pin
- `mcp__memory__unpin` to unpin

When showing status, explain:
- Items sorted by hot_score (highest first)
- Pinned items won't be evicted
- Hit rate and effectiveness metrics
