# Reference

Complete API and configuration reference for Memory MCP.

## Slash Commands

With the Claude Code plugin (`claude plugins add michael-denyer/memory-mcp`), these commands are available:

| Command | Description |
|---------|-------------|
| `/memory-mcp:remember` | Store a memory interactively |
| `/memory-mcp:recall` | Search memories by query |
| `/memory-mcp:hot-cache` | View and manage hot cache |
| `/memory-mcp:stats` | Show memory statistics |
| `/memory-mcp:bootstrap` | Seed from project docs |
| `/memory-mcp:trust` | Validate or invalidate memories |
| `/memory-mcp:link` | Connect related memories |
| `/memory-mcp:mining` | Review mined patterns |
| `/memory-mcp:list` | Browse all memories |
| `/memory-mcp:forget` | Delete a memory |
| `/memory-mcp:consolidate` | Merge duplicate memories |
| `/memory-mcp:maintenance` | Run database maintenance |
| `/memory-mcp:session` | Manage session context |
| `/memory-mcp:test-all` | Run internal tests |

## Tools

### Memory Operations

| Tool | Description |
|------|-------------|
| `remember(content, type, tags)` | Store a memory with semantic embedding |
| `recall(query, limit, threshold, expand_relations)` | Semantic search with confidence gating and optional multi-hop expansion |
| `recall_by_tag(tag)` | Filter memories by tag |
| `forget(memory_id)` | Delete a memory |
| `list_memories(limit, offset, type)` | Browse all memories |

### Hot Cache Management

| Tool | Description |
|------|-------------|
| `hot_cache_status()` | Show contents, metrics, and effectiveness |
| `promote(memory_id)` | Manually promote to hot cache |
| `demote(memory_id)` | Remove from hot cache (keeps in cold storage) |
| `pin_memory(memory_id)` | Pin memory (prevents auto-eviction) |
| `unpin_memory(memory_id)` | Unpin memory (allows auto-eviction) |

### Pattern Mining

| Tool | Description |
|------|-------------|
| `log_output(content)` | Log content for pattern extraction |
| `run_mining(hours)` | Extract patterns from recent logs |
| `review_candidates()` | See patterns ready for promotion |
| `approve_candidate(id)` / `reject_candidate(id)` | Accept or reject patterns |
| `bulk_reject_candidates(pattern_ids, pattern_type_prefix)` | Bulk reject by ID list or type prefix |

### Cold Start / Seeding

| Tool | Description |
|------|-------------|
| `bootstrap_project(root, files, promote)` | Auto-detect and seed from project docs (README.md, CLAUDE.md, etc.) |
| `seed_from_text(content, type, promote)` | Parse text into memories |
| `seed_from_file(path, type, promote)` | Import from file (e.g., CLAUDE.md) |

### Knowledge Graph

| Tool | Description |
|------|-------------|
| `link_memories(from_id, to_id, relation, metadata)` | Create relationship between memories |
| `unlink_memories(from_id, to_id, relation)` | Remove relationship(s) |
| `get_related_memories(memory_id, relation, direction)` | Find connected memories |

Relation types: `relates_to`, `depends_on`, `supersedes`, `refines`, `contradicts`, `elaborates`

### Trust Management

| Tool | Description |
|------|-------------|
| `strengthen_trust(memory_id, amount, reason)` | Increase confidence in a memory |
| `weaken_trust(memory_id, amount, reason)` | Decrease confidence (e.g., found outdated) |

### Retrieval Quality

| Tool | Description |
|------|-------------|
| `mark_memory_used(memory_id, feedback)` | Mark a recalled memory as actually helpful |
| `retrieval_quality_stats(memory_id, days)` | Get stats on which memories are retrieved vs used |

### Session Tracking

| Tool | Description |
|------|-------------|
| `get_or_create_session(session_id, topic)` | Track conversation context |
| `get_session_memories(session_id)` | Retrieve memories from a session |
| `end_session(session_id, promote_top)` | End session and promote top episodic memories to long-term storage |

## Memory Types

| Type | Use for |
|------|---------|
| `project` | Architecture, conventions, tech stack |
| `pattern` | Reusable code patterns, commands |
| `reference` | API docs, external references |
| `conversation` | Facts from discussions |
| `episodic` | Session-bound short-term context (auto-expires after 7 days) |

## Confidence Gating

Recall results include confidence levels based on semantic similarity:

| Confidence | Similarity | Recommended action |
|------------|------------|-------------------|
| **high** | > 0.85 | Use directly |
| **medium** | 0.70 - 0.85 | Verify context |
| **low** | < 0.70 | Reason from scratch |

## Configuration

Environment variables (prefix `MEMORY_MCP_`):

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `~/.memory-mcp/memory.db` | SQLite database location |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_BACKEND` | `auto` | `auto`, `mlx`, or `sentence-transformers` |

### Promoted Memories

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMOTED_MAX_ITEMS` | `20` | Maximum items in promoted memories |
| `PROMOTION_THRESHOLD` | `3` | Access count for auto-promotion |
| `DEMOTION_DAYS` | `14` | Days without access before demotion |
| `AUTO_PROMOTE` | `true` | Enable automatic promotion |
| `AUTO_DEMOTE` | `true` | Enable automatic demotion |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_RECALL_LIMIT` | `5` | Default results per recall |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.7` | Minimum similarity for results |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.85` | Threshold for "high" confidence |
| `RECALL_EXPAND_RELATIONS` | `false` | Enable multi-hop recall via knowledge graph |

### Salience & Promotion

| Variable | Default | Description |
|----------|---------|-------------|
| `SALIENCE_PROMOTION_THRESHOLD` | `0.5` | Minimum salience score for auto-promotion |
| `SALIENCE_IMPORTANCE_WEIGHT` | `0.25` | Weight for importance in salience |
| `SALIENCE_TRUST_WEIGHT` | `0.25` | Weight for trust in salience |
| `SALIENCE_ACCESS_WEIGHT` | `0.25` | Weight for access count in salience |
| `SALIENCE_RECENCY_WEIGHT` | `0.25` | Weight for recency in salience |

### Episodic Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `EPISODIC_PROMOTE_TOP_N` | `3` | Top N episodic memories to promote on session end |
| `EPISODIC_PROMOTE_THRESHOLD` | `0.6` | Minimum salience for episodic promotion |
| `RETENTION_EPISODIC_DAYS` | `7` | Days to retain episodic memories |

### Hot Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HOT_CACHE_ENABLED` | `true` | Enable memory://hot-cache resource |
| `HOT_CACHE_MAX_ITEMS` | `10` | Maximum items in hot cache |

### Project Awareness

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_AWARENESS_ENABLED` | `true` | Auto-detect git project for memories |
| `PROJECT_FILTER_RECALL` | `true` | Filter recall to current project |
| `PROJECT_FILTER_HOT_CACHE` | `true` | Filter hot cache/promoted to current project |
| `PROJECT_INCLUDE_GLOBAL` | `true` | Include global memories with project |

## MCP Resources

The server exposes MCP resources for instant memory access:

### Hot Cache (`memory://hot-cache`)

Session-aware active memory context (Engram-inspired). Provides contextually relevant memories:

1. Recently recalled memories (that were actually used)
2. Predicted next memories (from access pattern learning)
3. Top salience promoted items (to fill remaining slots)

Focused context (~10 items) designed for active work. **Auto-bootstrap**: If empty, auto-seeds from project docs.

### Promoted Memories (`memory://promoted-memories`)

Backing store of frequently-used memories. Contents available via MCP resource (disabled by default).

- Memories auto-promoted after 3+ uses appear here
- Keeps system prompts lean (~20 items max)
- Enable injection with `MEMORY_MCP_PROMOTED_RESOURCE_ENABLED=true`

### Project Context (`memory://project-context`)

Shows the current project (detected from git) and its associated memories:

- Project ID (e.g., `github/owner/repo`)
- Project-specific promoted memories
- Useful for debugging project awareness

## CLI Commands

```bash
# Bootstrap promoted memories from project docs
memory-mcp-cli bootstrap

# Bootstrap from specific directory
memory-mcp-cli bootstrap -r /path/to/project

# Log content for mining
echo "Some content" | memory-mcp-cli log-output

# Run pattern extraction
memory-mcp-cli run-mining --hours 24

# Seed from a file
memory-mcp-cli seed ~/project/CLAUDE.md -t project --promote

# Consolidate similar memories
memory-mcp-cli consolidate --dry-run
memory-mcp-cli consolidate

# Show memory system status
memory-mcp-cli status

# Launch web dashboard
memory-mcp-cli dashboard
```

## Multi-Client Setup

Memory MCP works with any MCP-compatible client (Claude Code, Codex, etc.).

### Shared Memory (Recommended)

Both clients share the same database - memories learned in one are available in the other:

**Claude Code** (`~/.claude.json`):
```json
{
  "mcpServers": {
    "memory": {
      "command": "memory-mcp"
    }
  }
}
```

### Separate Memory per Client

Use different database paths via `MEMORY_MCP_DB_PATH` environment variable:

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory-mcp",
      "env": {
        "MEMORY_MCP_DB_PATH": "~/.memory-mcp/claude.db"
      }
    }
  }
}
```

## Automatic Output Logging

For pattern mining to work automatically, install the Claude Code hook.

### Prerequisites

The hook script requires `jq` for JSON parsing:

```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt install jq
```

### Installation

```bash
chmod +x hooks/memory-log-response.sh
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "/path/to/memory-mcp/hooks/memory-log-response.sh"
      }]
    }]
  }
}
```
