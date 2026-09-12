# Memory MCP API Reference

This document is the reference for MCP tools, resources, and CLI commands.

## Table of Contents

- [MCP Tools](#mcp-tools)
  - [Memory Operations](#memory-operations)
  - [Hot Cache Management](#hot-cache-management)
  - [Bootstrap](#bootstrap)
  - [Knowledge Graph](#knowledge-graph)
  - [Maintenance](#maintenance)
- [MCP Resources](#mcp-resources)
- [CLI Commands](#cli-commands)
- [Response Types](#response-types)
- [Configuration](#configuration)

---

## MCP Tools

### Memory Operations

#### `remember`

Store a new memory with semantic embedding.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | Content to remember |
| `memory_type` | string | No | `"project"` | Type: `project`, `pattern`, `reference`, `conversation` |
| `tags` | list[str] | No | `null` | Tags for categorization |
| `session_id` | string | No | `null` | Session ID for provenance tracking |

**Returns**: `{success, message, memory_id, was_duplicate?}`

**Example**:
```
remember(content="This project uses PostgreSQL with pgvector", memory_type="project", tags=["database"])
```

---

#### `recall`

Semantic search with confidence gating and composite ranking.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `mode` | string | No | `null` | `precision`, `balanced`, or `exploratory` |
| `limit` | int | No | `null` | Max results (overrides mode default) |
| `threshold` | float | No | `null` | Min similarity (overrides mode default) |
| `memory_type` | string | No | `null` | Filter by type |
| `include_related` | bool | No | `false` | Include related memories from knowledge graph |

**Modes**:
| Mode | Threshold | Limit | Use Case |
|------|-----------|-------|----------|
| `precision` | 0.8 | 3 | High confidence, specific answers |
| `balanced` | 0.7 | 5 | General use |
| `exploratory` | 0.5 | 10 | Broad discovery |

**Returns**: `RecallResponse` with memories, confidence level, gated count, and LLM-friendly formatted context.

---

#### `list_memories`

Browse stored memories with pagination.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | `20` | Maximum results |
| `offset` | int | No | `0` | Skip first N results |
| `memory_type` | string | No | `null` | Filter by type |

---

#### `forget`

Delete a memory permanently.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | ID of memory to delete |

---

#### `memory_stats`

Get overall memory statistics.

**Returns**: `{total_memories, hot_cache_count, by_type, by_source}`

---

### Hot Cache Management

#### `hot_cache_status`

Show current hot cache contents, metrics, and effectiveness.

**Returns**: `HotCacheResponse` with:
- `items`: Current hot memories
- `max_items`, `current_count`, `pinned_count`
- `metrics`: hits, misses, evictions, promotions
- `effectiveness`: hit_rate_percent, estimated_tool_calls_saved

---

#### `promote`

Manually promote a memory to hot cache.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to promote |

---

#### `demote`

Remove a memory from hot cache (keeps in cold storage).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to demote |

---

#### `pin`

Pin a hot cache memory to prevent auto-eviction.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Hot memory to pin |

---

#### `unpin`

Unpin a memory, allowing auto-eviction.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to unpin |

---

### Bootstrap

#### `bootstrap_project`

Bootstrap hot cache from project documentation files.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `root_path` | string | No | `"."` | Project root directory |
| `file_patterns` | list[str] | No | `null` | Specific files (auto-detects if null) |
| `promote_to_hot` | bool | No | `true` | Promote to hot cache |
| `memory_type` | string | No | `"project"` | Memory type for content |
| `tags` | list[str] | No | `null` | Tags to apply |

**Auto-detected files** (priority order):
1. README.md, README
2. CONTRIBUTING.md
3. docs/README.md
4. ARCHITECTURE.md

**Returns**: `BootstrapResponse` with files_found, files_processed, memories_created, etc.

---

### Knowledge Graph

#### `link_memories`

Create a typed relationship between memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from_memory_id` | int | Yes | - | Source memory |
| `to_memory_id` | int | Yes | - | Target memory |
| `relation_type` | string | Yes | - | Relationship type |

**Relation types**:
| Type | Description |
|------|-------------|
| `relates_to` | General association |
| `depends_on` | Prerequisite relationship |
| `supersedes` | Replaces older information |
| `refines` | More specific version |
| `contradicts` | Conflicting information |
| `elaborates` | More detail |

---

#### `get_related_memories`

Get memories related to a given memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to find relations for |
| `relation_type` | string | No | `null` | Filter by type |
| `direction` | string | No | `"both"` | `outgoing`, `incoming`, or `both` |

---

### Maintenance

#### `db_maintenance`

Run database maintenance (vacuum, analyze, auto-demote).

**Returns**: `MaintenanceResponse` with bytes_reclaimed, memory_count, auto_demoted_count.

---

## MCP Resources

### `memory://hot-cache`

Auto-injectable system context with high-confidence patterns.

- Instant recall (no tool call needed)
- Auto-bootstraps from README.md, CONTRIBUTING.md if empty
- Records hit/miss metrics

**Content format**:
```
[MEMORY: Hot Cache - High-confidence patterns]
- Memory content 1 [tag1, tag2]
- Memory content 2
...
```

---

## CLI Commands

All commands support `--json` flag for machine-readable output.

### `memory-mcp-cli bootstrap`

Bootstrap hot cache from project documentation.

```bash
# Auto-detect and bootstrap
memory-mcp-cli bootstrap

# From specific directory
memory-mcp-cli bootstrap -r /path/to/project

# Specific files only
memory-mcp-cli bootstrap -f README.md -f ARCHITECTURE.md

# Promote seeded memories to the hot cache
memory-mcp-cli bootstrap --promote

# JSON output
memory-mcp-cli --json bootstrap
```

### `memory-mcp-cli seed`

Seed memories from a file.

```bash
memory-mcp-cli seed ~/project/CLAUDE.md -t project --promote
```

### `memory-mcp-cli status`

Show memory system status with hot cache contents.

```bash
memory-mcp-cli status
```

### `memory-mcp-cli db-rebuild-vectors`

Rebuild all memory vectors.

```bash
# Full rebuild
memory-mcp-cli db-rebuild-vectors

# Just clear vectors
memory-mcp-cli db-rebuild-vectors --clear-only
```

---

## Response Types

### MemoryResponse

```python
{
    "id": int,
    "content": str,
    "memory_type": str,  # project, pattern, reference, conversation
    "source": str,       # manual, mined
    "is_hot": bool,
    "is_pinned": bool,
    "tags": list[str],
    "access_count": int,
    "trust_score": float,
    "similarity": float | None,      # Set during recall
    "hot_score": float | None,
    "composite_score": float | None,
    "created_at": str  # ISO format
}
```

### RecallResponse

```python
{
    "memories": list[MemoryResponse],
    "confidence": str,       # high, medium, low
    "gated_count": int,      # Results filtered by threshold
    "mode": str,
    "guidance": str,         # Hallucination prevention hint
    "ranking_factors": str,  # Scoring explanation
    "formatted_context": list[FormattedMemory] | None,
    "context_summary": str | None,
    "promotion_suggestions": list[dict] | None,
    "related_memories": list[RelatedMemoryResponse] | None
}
```

### FormattedMemory (LLM-friendly)

```python
{
    "summary": str,      # Concise one-line summary
    "memory_type": str,
    "tags": list[str],
    "age": str,          # Human-readable: "2 hours", "3 days"
    "confidence": str,   # high, medium, low
    "source_hint": str   # "hot cache" or "cold storage"
}
```

---

## Configuration

All settings via environment variables with `MEMORY_MCP_` prefix.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `~/.memory-mcp/memory.db` | SQLite database location |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_DIM` | `384` | Embedding dimension |

### Hot Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HOT_CACHE_MAX_ITEMS` | `20` | Maximum hot cache size |
| `PROMOTION_THRESHOLD` | `3` | Access count for auto-promotion |
| `DEMOTION_DAYS` | `14` | Days without access before demotion |
| `AUTO_PROMOTE` | `true` | Enable automatic promotion |
| `AUTO_DEMOTE` | `true` | Enable automatic demotion |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_RECALL_LIMIT` | `5` | Default results per recall |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.7` | Minimum similarity |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.85` | "High" confidence threshold |

### Predictive Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTIVE_CACHE_ENABLED` | `true` | Enable predictive warming |
| `PREDICTION_THRESHOLD` | `0.3` | Min transition probability |
| `MAX_PREDICTIONS` | `3` | Max memories to predict |
| `SEQUENCE_DECAY_DAYS` | `30` | Days before sequence decay |

### Trust

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUST_SCORE_MANUAL` | `1.0` | Trust for manual memories |
| `TRUST_SCORE_MINED` | `0.7` | Trust for mined memories |
| `TRUST_DECAY_HALFLIFE_DAYS` | `90.0` | Default trust decay half-life |

See [config.py](../src/memory_mcp/config.py) for complete configuration options.
