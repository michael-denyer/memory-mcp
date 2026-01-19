# Memory MCP - Project Instructions

## Core Value Proposition

**The Engram Insight**: Frequently-used patterns should be instantly available, not searched for.

This project differentiates from generic memory servers (like mcp-memory-service) through:

1. **Two-Tier Memory Architecture**
   - **Hot Cache**: Zero-latency access via MCP resource injection (no tool call needed)
   - **Cold Storage**: Semantic search for everything else

2. **Automatic Pattern Promotion** (planned)
   - Patterns accessed 3+ times auto-promote to hot cache
   - Stale patterns auto-demote after N days without access
   - The system learns what you use frequently

3. **Pattern Mining from Usage**
   - Extracts imports, commands, project facts from Claude's outputs
   - Frequency-based promotion candidates
   - Human approval before promotion

## Architecture

```
Hot Cache (instant)     Cold Storage (tool call)
    │                         │
    ▼                         ▼
┌─────────────┐        ┌─────────────┐
│ MCP Resource│        │ Vector DB   │
│ (injected)  │        │ (sqlite-vec)│
└─────────────┘        └─────────────┘
    ▲                         ▲
    │                         │
    └────── Promotion ────────┘
          (frequency-based)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/memory_mcp/server.py` | MCP tools and resources |
| `src/memory_mcp/storage.py` | SQLite + vector operations |
| `src/memory_mcp/mining.py` | Pattern extraction |
| `src/memory_mcp/cli.py` | CLI commands for hooks |
| `hooks/memory-log-response.sh` | Claude Code Stop hook |

## Development Priorities

1. **P1: Strengthen the differentiator**
   - Auto-promotion on access threshold (MemoryMCP-6p2)
   - Hot cache feedback loop showing value (MemoryMCP-2x3)

2. **P2: Simplify**
   - Replace custom LRU with stdlib (MemoryMCP-x5s)
   - Reduce config options to essentials

3. **P3: Polish**
   - Rich CLI output
   - Better error messages

## Testing

```bash
uv run pytest -v        # All tests
uv run pytest -k hot    # Hot cache tests only
```

## When Working on This Project

- The hot cache is the differentiator - keep it simple and automatic
- Avoid adding features that don't serve the two-tier memory model
- If a feature requires manual user action, question whether it's worth it
- Test with real Claude Code usage, not just unit tests
