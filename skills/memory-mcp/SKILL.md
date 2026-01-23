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
| **Hot Cache** | 0ms | Auto-injected into context before Claude thinks |
| **Cold Storage** | ~50ms | Semantic search via `recall()` tool call |

The system learns what you use and automatically promotes frequently-accessed memories to the hot cache.

## Quick Start

### Store a Memory
```
remember("FastAPI with async endpoints for all APIs", memory_type="project", tags=["tech-stack"])
```

### Recall by Meaning
```
recall("what framework for backend")  # Finds FastAPI memory
```

### Check What's Hot
```
hot_cache_status()  # See what's instantly available
```

## Core Tools

### Storage
| Tool | Purpose |
|------|---------|
| `remember(content, memory_type, tags)` | Store new memory |
| `recall(query, mode, limit)` | Semantic search |
| `recall_by_tag(tag)` | Find by tag |
| `forget(memory_id)` | Delete memory |
| `list_memories(limit, offset)` | Browse all |

**Memory types**: `project`, `pattern`, `reference`, `episodic`, `conversation`

**Recall modes**: `precision` (few, high-confidence), `balanced` (default), `exploratory` (many results)

### Hot Cache
| Tool | Purpose |
|------|---------|
| `promote(memory_id)` | Add to hot cache |
| `demote(memory_id)` | Remove from hot cache |
| `pin(memory_id)` | Prevent auto-eviction |
| `unpin(memory_id)` | Allow auto-eviction |
| `hot_cache_status()` | View hot cache contents |

### Knowledge Graph
| Tool | Purpose |
|------|---------|
| `link_memories(from_id, to_id, relation)` | Connect memories |
| `unlink_memories(from_id, to_id)` | Remove connection |
| `get_related_memories(memory_id)` | Find connected |
| `relationship_stats()` | Graph overview |

**Relation types**: `relates_to`, `depends_on`, `supersedes`, `refines`, `contradicts`, `elaborates`

### Trust Management
| Tool | Purpose |
|------|---------|
| `validate_memory(id, reason)` | Increase trust |
| `invalidate_memory(id, reason)` | Decrease trust |
| `get_trust_history(memory_id)` | View changes |

### Sessions
| Tool | Purpose |
|------|---------|
| `get_sessions()` | List sessions |
| `summarize_session(session_id)` | Structured summary |
| `end_session(session_id)` | Promote top memories |

### Pattern Mining
| Tool | Purpose |
|------|---------|
| `mining_status()` | View mining stats |
| `review_candidates()` | See patterns found |
| `approve_candidate(id)` | Promote to memory |
| `reject_candidate(id)` | Discard pattern |

### Maintenance
| Tool | Purpose |
|------|---------|
| `memory_stats()` | Overview stats |
| `db_info()` | Database details |
| `run_cleanup()` | Clean stale data |
| `preview_consolidation()` | Find duplicates |

## MCP Resources

These are auto-injected into Claude's context:

| Resource | Contents |
|----------|----------|
| `memory://hot-cache` | All promoted memories |
| `memory://working-set` | Session-aware context (~10 items) |
| `memory://project-context` | Current project memories |

## Auto-Promotion Rules

Memories are auto-promoted to hot cache when:
- Salience score ≥ 0.5 AND access count ≥ 3
- Salience = importance + trust + access_count + recency

Memories are auto-demoted after 14 days without access.

## Common Workflows

### Project Setup
```
# Bootstrap from project docs (CLAUDE.md, README.md, etc.)
bootstrap_project(promote_to_hot=true)
```

### Daily Work
```
# Store decisions and patterns as you work
remember("Decided to use PostgreSQL for main DB", memory_type="project", tags=["decision", "database"])

# Recall when needed
recall("database decision")
```

### Session End
```
# Review what you learned
summarize_session(session_id)

# Promote valuable memories to long-term storage
end_session(session_id, promote_top=true)
```

### Knowledge Linking
```
# Connect related concepts
link_memories(postgres_id, pgvector_id, "depends_on")

# Recall with graph expansion
recall("PostgreSQL", expand_relations=true)
```

## Tips

1. **Tag consistently** - Use tags like `decision`, `convention`, `tech-stack`, `gotcha`
2. **Use episodic for session context** - Short-term memories that may get promoted
3. **Link related memories** - Build a knowledge graph for better recall
4. **Trust the auto-promotion** - Don't over-promote manually
5. **Check hot cache periodically** - `hot_cache_status()` shows what's instantly available
