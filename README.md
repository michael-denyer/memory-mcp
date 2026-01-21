<!-- mcp-name: io.github.michael-denyer/hot-memory-mcp -->
<div align="center">

# 🧠 Memory MCP Server

### Give your AI assistant a persistent second brain

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Works with Claude Code](https://img.shields.io/badge/Works%20with-Claude%20Code-blueviolet)](https://claude.ai/claude-code)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

<br />

**Stop re-explaining your project every session.**

Memory MCP learns what matters and keeps it ready — instant recall for the stuff you use most, semantic search for everything else.

</div>

---

## The Problem

Every new chat starts from scratch. You repeat yourself. Context balloons. Tool calls add latency.

**Memory MCP fixes this.** It gives Claude persistent memory with a two-tier architecture: a hot cache for instant access to frequently-used knowledge, and cold storage with semantic search for everything else.

The system learns what you use and automatically promotes it. No manual curation required.

## Before & After

| 😤 Without Memory MCP | 🎯 With Memory MCP |
|----------------------|-------------------|
| "Let me explain our architecture again..." | Project facts persist forever |
| Copy-paste the same patterns | Patterns auto-promoted to instant access |
| 500k+ token context windows | Hot cache keeps it lean (~20 items) |
| Tool call latency on every lookup | Hot cache: **0ms** — already in context |

## Key Features

🚀 **Instant recall hot cache** — Frequently-used memories auto-injected into context. No tool calls needed.

🔍 **Semantic search** — Find memories by meaning, not just keywords. Knowledge graph connects related concepts.

🤖 **Self-organizing** — Learns what you use. Auto-promotes frequent patterns. Auto-demotes stale ones.

🗂️ **Project-aware** — Memories auto-tagged by git repo. Switch projects, get relevant context automatically.

📦 **Local & private** — All data in SQLite. No cloud. No API keys. Works offline.

🍎 **Apple Silicon optimized** — MLX backend auto-detected on M-series Macs for faster embeddings.

## Quick Start

### Install

```bash
# uv (recommended)
uv tool install hot-memory-mcp

# pip
pip install hot-memory-mcp

# From source
git clone https://github.com/michael-denyer/memory-mcp.git
cd memory-mcp && uv sync
```

**Apple Silicon?** Add MLX support for faster embeddings:
```bash
uv tool install hot-memory-mcp[mlx]
# or: pip install hot-memory-mcp[mlx]
```

### Configure

Add to your MCP client config (e.g., `~/.claude.json` for Claude Code):

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory-mcp"
    }
  }
}
```

<details>
<summary>From source? Use this config instead</summary>

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/memory-mcp", "memory-mcp"]
    }
  }
}
```
</details>

Restart your client. That's it. The hot cache auto-populates from your project docs.

> **First run**: Embedding model (~90MB) downloads automatically. Takes 30-60 seconds once.

## How It Works

```mermaid
flowchart LR
    subgraph LLM["Your AI Assistant"]
        REQ((Request))
    end

    subgraph Hot["HOT CACHE · 0ms"]
        HC[Frequent memories]
        WS[Working set]
    end

    subgraph Cold["COLD STORAGE · ~50ms"]
        VS[(Vector search)]
        KG[(Knowledge graph)]
    end

    REQ -->|"auto-injected"| HC
    REQ -->|"recall()"| VS
    VS <-->|"related"| KG
```

**Two tiers, automatic promotion:**

| Tier | Latency | What happens |
|------|---------|--------------|
| **Hot Cache** | 0ms | Auto-injected every request. No tool call needed. |
| **Cold Storage** | ~50ms | Semantic search when you need deeper recall. |

Memories used 3+ times automatically promote to hot cache. Unused memories demote after 14 days. Pin important ones to keep them hot forever.

## What Makes It Different

| | Memory MCP | Others |
|---|------------|--------|
| **Hot cache** | Auto-injected, 0ms | Most require tool calls |
| **Self-organizing** | Learns from usage | Manual curation |
| **Project-aware** | Auto-isolates by git repo | Manual tagging |
| **Pattern mining** | Extracts from outputs | Not available |
| **Setup** | One command, local SQLite | Often needs cloud/services |

---

# Reference

Everything below is detailed documentation. You don't need to read it to get started.

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

### Hot Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HOT_CACHE_MAX_ITEMS` | `20` | Maximum items in hot cache |
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

### Working Set

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKING_SET_ENABLED` | `true` | Enable memory://working-set resource |
| `WORKING_SET_MAX_ITEMS` | `10` | Maximum items in working set |

### Project Awareness

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_AWARENESS_ENABLED` | `true` | Auto-detect git project for memories |
| `PROJECT_FILTER_RECALL` | `true` | Filter recall to current project |
| `PROJECT_FILTER_HOT_CACHE` | `true` | Filter hot cache to current project |
| `PROJECT_INCLUDE_GLOBAL` | `true` | Include global memories with project |

## MCP Resources

The server exposes two MCP resources for instant memory access:

### Hot Cache (`memory://hot-cache`)

Auto-injectable system context with high-confidence patterns. Contents are automatically available in Claude's context without tool calls.

- Memories promoted to hot cache appear here
- Keeps system prompts lean (~10-20 items max)
- **Auto-bootstrap**: If empty, auto-seeds from project docs (README.md, CLAUDE.md, etc.)

### Working Set (`memory://working-set`)

Session-aware active memory context (Engram-inspired). Provides contextually relevant memories:

1. Recently recalled memories (that were actually used)
2. Predicted next memories (from access pattern learning)
3. Top salience hot items (to fill remaining slots)

Smaller and more focused than hot-cache (~10 items) - designed for active work context.

### Project Context (`memory://project-context`)

Shows the current project (detected from git) and its associated memories:

- Project ID (e.g., `github/owner/repo`)
- Project-specific hot cache memories
- Useful for debugging project awareness

### Enabling Auto-Injection

Add the MCP server to your settings (see Quick Start). Both resources are automatically available. Verify with `/mcp` in Claude Code.

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

**Codex** (or other MCP client):
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

## CLI Commands

```bash
# Bootstrap hot cache from project docs (auto-detects README.md, CLAUDE.md, etc.)
memory-mcp-cli bootstrap

# Bootstrap from specific directory
memory-mcp-cli bootstrap -r /path/to/project

# Bootstrap specific files only
memory-mcp-cli bootstrap -f README.md -f ARCHITECTURE.md

# Log content for mining
echo "Some content" | memory-mcp-cli log-output

# Run pattern extraction
memory-mcp-cli run-mining --hours 24

# Seed from a file
memory-mcp-cli seed ~/project/CLAUDE.md -t project --promote

# Consolidate similar memories (preview first with --dry-run)
memory-mcp-cli consolidate --dry-run
memory-mcp-cli consolidate

# Show memory system status
memory-mcp-cli status
```

<details>
<summary>From source? Prefix commands with `uv run`</summary>

```bash
uv run memory-mcp-cli bootstrap
uv run memory-mcp-cli status
# etc.
```
</details>

## Development

```bash
# Run tests
uv run pytest -v

# Run with debug logging
uv run memory-mcp 2>&1 | head -50
```

### System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.11+ recommended for performance |
| Disk | ~2-3 GB | Dependencies (~2GB) + embedding model (~90MB) + database |
| RAM | 200-400 MB | During embedding operations |
| First Run | 30-60 seconds | One-time ~90MB model download |
| Startup | 2-5 seconds | After model is cached |

**Apple Silicon Users**: Install with MLX support for faster embeddings:
```bash
pip install memory-mcp[mlx]
```

## Example Usage

```
You: "Remember that this project uses PostgreSQL with pgvector"
Claude: [calls remember(..., memory_type="project")]
→ Stored as memory #1

You: "What database do we use?"
Claude: [calls recall("database configuration")]
→ {confidence: "high", memories: [{content: "PostgreSQL with pgvector..."}]}

You: "Promote that to hot cache"
Claude: [calls promote(1)]
→ Memory #1 now in hot cache - available instantly next session
```

## Troubleshooting

### Server Won't Start

**Symptom**: Claude Code shows "memory" server as disconnected

1. **Check the command works directly**:
   ```bash
   memory-mcp
   ```

2. **Verify installation**:
   ```bash
   which memory-mcp  # Should return a path
   ```

3. **Check Python version**: Requires 3.10+
   ```bash
   python --version
   ```

### Dimension Mismatch Error

**Symptom**: `Vector dimension mismatch` error during recall

This happens when the embedding model changes. Rebuild vectors:

```bash
memory-mcp-cli db-rebuild-vectors
```

### Hot Cache Not Updating

**Symptom**: Promoted memories don't appear in hot cache

1. **Check hot cache status**:
   ```bash
   memory-mcp-cli status
   ```

2. **Verify memory exists**:
   ```
   [In Claude] list_memories(limit=10)
   ```

3. **Manually promote**:
   ```
   [In Claude] promote(memory_id)
   ```

### Pattern Mining Not Working

**Symptom**: `run_mining` finds no patterns

1. **Check mining is enabled**:
   ```bash
   echo $MEMORY_MCP_MINING_ENABLED  # Should not be "false"
   ```

2. **Verify logs exist**:
   ```bash
   memory-mcp-cli run-mining --hours 24
   ```

3. **Check hook is installed** (see [Automatic Output Logging](#automatic-output-logging))

### Hook Script Fails

**Symptom**: Hook runs but nothing is logged

1. **Check jq is installed**:
   ```bash
   which jq  # Should return a path
   ```

2. **Make script executable**:
   ```bash
   chmod +x hooks/memory-log-response.sh
   ```

3. **Test manually**:
   ```bash
   echo "test content" | memory-mcp-cli log-output
   ```

### Slow First Startup

**Symptom**: First run takes 30-60 seconds

This is expected - the embedding model (~90MB) downloads on first use. Subsequent starts take 2-5 seconds.

### Database Corruption

**Symptom**: SQLite errors or unexpected behavior

1. **Backup and recreate**:
   ```bash
   mv ~/.memory-mcp/memory.db ~/.memory-mcp/memory.db.bak
   # Server will create fresh database on next start
   ```

2. **Re-bootstrap from project docs**:
   ```bash
   memory-mcp-cli bootstrap
   ```

## Security Note

This server is designed for **local use only**. It runs unauthenticated over STDIO transport and should not be exposed to networks or untrusted clients.

## License

MIT
