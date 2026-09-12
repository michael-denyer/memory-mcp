---
name: memory-mcp
description: >
  Persistent memory for Claude Code with two-tier architecture: hot cache for instant
  recall (0ms) and semantic search for everything else (~50ms). Automatically learns
  what you use and promotes it.
allowed-tools: "mcp__memory__*"
version: "0.6.0"
author: "Michael Denyer <https://github.com/michael-denyer>"
license: "MIT"
---

# Memory MCP - Persistent Memory for Claude Code

Give your AI assistant a second brain that persists across sessions.

## Two-Tier Architecture

| Tier | Latency | How it works |
|------|---------|--------------|
| **Hot Cache** | 0ms | Printed into context by the `SessionStart` and `UserPromptSubmit` hooks |
| **Cold Storage** | ~50ms | Semantic search via a `recall()` tool call |

The hooks run `memory-mcp-cli hot-cache`, whose stdout Claude Code adds to the conversation. No
tool call is involved. The footer of that text asks you to call `mark_memory_used(id)` when one of
the injected memories was useful, which is the signal that keeps it in the hot cache.

## Quick Start

```
remember("FastAPI with async endpoints for all APIs", memory_type="project", tags=["tech-stack"])
recall("what framework for backend")
mark_memory_used(id)
```

## Tools

### Storage
| Tool | Purpose |
|------|---------|
| `remember(content, memory_type, tags)` | Store new memory |
| `recall(query, mode, limit, expand_relations)` | Semantic search |
| `forget(memory_id)` | Delete memory |
| `list_memories(limit, offset)` | Browse all |
| `memory_stats()` | Overview stats |

**Memory types**: `project`, `pattern`, `reference`, `conversation`, `episodic`

**Recall modes**: `precision` (few, high-confidence), `balanced` (default), `exploratory` (many results)

### Hot Cache
| Tool | Purpose |
|------|---------|
| `hot_cache_status()` | View hot cache contents |
| `promote(memory_id)` | Add to hot cache |
| `demote(memory_id)` | Remove from hot cache |
| `pin(memory_id)` | Prevent auto-eviction |
| `unpin(memory_id)` | Allow auto-eviction |
| `mark_memory_used(memory_id)` | Record that a memory was useful |

### Knowledge Graph
| Tool | Purpose |
|------|---------|
| `link_memories(from_id, to_id, relation)` | Connect memories |
| `get_related_memories(memory_id)` | Find connected |

**Relation types**: `relates_to`, `depends_on`, `supersedes`, `refines`, `contradicts`, `elaborates`

### Sessions and setup
| Tool | Purpose |
|------|---------|
| `end_session(session_id)` | Consolidate a session's episodic memories |
| `bootstrap_project(root_path)` | Seed from project documentation |
| `db_maintenance()` | Vacuum, analyze, demote stale entries |

These sixteen tools are the whole surface.

## MCP Resources

| Resource | Contents |
|----------|----------|
| `memory://hot-cache` | Session-aware active context |
| `memory://promoted-memories` | The promoted backing store |
| `memory://project-context` | Current project memories |

Resources are not injected on their own. Reach one with an `@` mention or a resource tool. The
hooks are what put the hot cache in front of Claude.

## Auto-Promotion Rules

A memory is promoted when its salience score reaches 0.5 and it has been accessed three times.
Salience combines importance, trust, access count and recency. A memory is demoted after 14 days
without access.

## Common Workflows

### Project setup
```
bootstrap_project(root_path=".")
```

### Daily work
```
remember("Decided to use PostgreSQL for main DB", memory_type="project", tags=["decision", "database"])
recall("database decision")
```

### Session end
```
end_session(session_id, promote_top=true)
```

### Knowledge linking
```
link_memories(postgres_id, pgvector_id, "depends_on")
recall("PostgreSQL", expand_relations=true)
```

## Tips

1. **Tag consistently** with tags like `decision`, `convention`, `tech-stack`, `gotcha`
2. **Call `mark_memory_used`** when an injected memory helped, so promotion tracks real use
3. **Use `episodic`** for session context that may earn promotion later
4. **Link related memories** to build a graph that `expand_relations` can walk
5. **Trust auto-promotion** rather than promoting by hand
