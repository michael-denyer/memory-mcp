<!-- mcp-name: io.github.michael-denyer/hot-memory-mcp -->
<div align="center">

# 🧠 Memory MCP

### Give your AI assistant a persistent second brain

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP 1.0](https://img.shields.io/badge/MCP-1.0-green.svg)](https://modelcontextprotocol.io)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-2.1+-blueviolet.svg)](https://claude.ai/code)
[![PyPI](https://img.shields.io/pypi/v/hot-memory-mcp.svg)](https://pypi.org/project/hot-memory-mcp/)
[![CI](https://github.com/michael-denyer/memory-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/michael-denyer/memory-mcp/actions)

<br />

**Stop re-explaining your project every session.**

Memory MCP learns what matters and keeps it ready — instant recall for the stuff you use most, semantic search for everything else.

</div>

---

## The Problem

Every new chat starts from scratch. You explain your architecture *again*. You paste the same patterns *again*. Your context window bloats with repetition.

Other memory solutions help, but they still require tool calls for every lookup — adding latency and eating into Claude's thinking budget.

**Memory MCP fixes this with a two-tier architecture:**

1. **Hot cache (0ms)** — Frequently-used knowledge auto-injected into context *before Claude even starts thinking*. No tool call needed.
2. **Cold storage (~50ms)** — Everything else, searchable by meaning via semantic similarity.

The system learns what you use and promotes it automatically. Your most valuable knowledge becomes instantly available. No manual curation required.

## Before & After

| 😤 Without Memory MCP | 🎯 With Memory MCP |
|----------------------|-------------------|
| "Let me explain our architecture again..." | Project facts persist and isolate per repo |
| Copy-paste the same patterns every session | Patterns auto-promoted to instant access |
| 500k+ token context windows | Hot cache keeps it lean (~20 items) |
| Tool call latency on every memory lookup | Hot cache: **0ms** — already in context |
| Stale information lingers forever | Trust scoring demotes outdated facts |
| Flat list of disconnected facts | Knowledge graph connects related concepts |

## Install

```bash
# Install package
uv tool install hot-memory-mcp   # or: pip install hot-memory-mcp

# Add plugin (recommended)
claude plugins add michael-denyer/memory-mcp
```

The plugin gives you auto-configured hooks, slash commands, and the Memory Analyst agent. MLX is auto-detected on Apple Silicon.

<details>
<summary>Manual config (no plugin)</summary>

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory-mcp"
    }
  }
}
```

See [Reference](docs/REFERENCE.md) for full configuration options.
</details>

Restart Claude Code. The hot cache auto-populates from your project docs.

> **First run**: Embedding model (~90MB) downloads automatically. Takes 30-60 seconds once.

## How It Works

```mermaid
flowchart LR
    subgraph LLM["Claude"]
        REQ((Request))
    end

    subgraph Hot["HOT CACHE · 0ms"]
        HC[Session context]
        PM[(Promoted memories)]
    end

    subgraph Cold["COLD STORAGE · ~50ms"]
        VS[(Vector search)]
        KG[(Knowledge graph)]
    end

    REQ -->|"auto-injected"| HC
    HC -.->|"draws from"| PM
    REQ -->|"recall()"| VS
    VS <-->|"related"| KG
```

The **hot cache** (~10 items) is injected into every request — it combines recent recalls, predicted next memories, and top promoted items. **Promoted memories** (~20 items) is the backing store of frequently-used memories. Memories used 3+ times auto-promote; unused ones demote after 14 days.

## What Makes It Different

Most memory systems make you pay a tool-call tax on every lookup. Memory MCP's **hot cache bypasses this entirely** — your most-used knowledge is already in context when Claude starts thinking.

| | Memory MCP | Generic Memory Servers |
|---|------------|------------------------|
| **Hot cache** | Auto-injected at 0ms | Every lookup = tool call |
| **Self-organizing** | Learns and promotes automatically | Manual curation required |
| **Project-aware** | Auto-isolates by git repo | One big pile of memories |
| **Knowledge graph** | Multi-hop recall across concepts | Flat list of facts |
| **Pattern mining** | Learns from Claude's outputs | Not available |
| **Trust scoring** | Outdated info decays and sinks | All memories equal |
| **Setup** | One command, local SQLite | Often needs cloud setup |

**The Engram Insight**: Human memory doesn't search — frequently-used patterns are *already there*. That's what hot cache does for Claude.

## Quick Reference

| Slash Command | Tool | Description |
|---------------|------|-------------|
| `/memory-mcp:remember` | `remember` | Store a memory with semantic embedding |
| `/memory-mcp:recall` | `recall` | Search memories by meaning |
| `/memory-mcp:hot-cache` | `promote` / `demote` | Manage promoted memories |
| `/memory-mcp:stats` | `memory_stats` | Show statistics |
| `/memory-mcp:bootstrap` | `bootstrap_project` | Seed from project docs |
| — | `link_memories` | Knowledge graph connections |

See [Reference](docs/REFERENCE.md) for all 14 slash commands and full tool API.

### Dashboard

```bash
memory-mcp-cli dashboard    # Opens at http://localhost:8765
```

![Dashboard](docs/images/dashboard.png)

Browse memories, hot cache, mining candidates, sessions, and knowledge graph.

## How to Use

Memory MCP is designed to run as three complementary components:

| Component | Purpose |
|-----------|---------|
| **Claude Code Plugin** | Hooks, slash commands, and Memory Analyst agent for seamless integration |
| **MCP Server** | Core memory tools available to Claude via Model Context Protocol |
| **Dashboard** | Web UI to browse, manage, and debug your memory database |

The plugin is recommended for most users — it auto-configures the MCP server and adds productivity features. Run the dashboard alongside when you want visibility into what's being stored.

## Documentation

| Document | Description |
|----------|-------------|
| [Reference](docs/REFERENCE.md) | Full API, CLI, configuration, MCP resources |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

## License

MIT
