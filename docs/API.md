# Memory MCP API Reference

This document provides complete reference for all MCP tools, resources, and CLI commands.

## Table of Contents

- [MCP Tools](#mcp-tools)
  - [Memory Operations](#memory-operations)
  - [Hot Cache Management](#hot-cache-management)
  - [Pattern Mining](#pattern-mining)
  - [Seeding & Bootstrap](#seeding--bootstrap)
  - [Knowledge Graph](#knowledge-graph)
  - [Trust Management](#trust-management)
  - [Contradiction Detection](#contradiction-detection)
  - [Session Tracking](#session-tracking)
  - [Predictive Cache](#predictive-cache)
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

#### `recall_with_fallback`

Automatic fallback through memory types until results found.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `mode` | string | No | `null` | Recall mode |
| `min_results` | int | No | `1` | Minimum results before trying next fallback |

Tries: patterns → project facts → all types.

---

#### `recall_by_tag`

Filter memories by tag.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tag` | string | Yes | - | Tag to filter by |
| `limit` | int | No | `10` | Maximum results |

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

### Pattern Mining

#### `log_output`

Log content for pattern mining.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | Output content to log |
| `session_id` | string | No | `null` | Session ID for provenance |

---

#### `run_mining`

Run pattern extraction on recent logs.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `hours` | int | No | `24` | Hours of logs to process |

**Returns**: `{outputs_processed, patterns_found, new_patterns, updated_patterns, auto_approved}`

Patterns meeting auto-approve thresholds (default: confidence ≥ 0.8, occurrences ≥ 3) are automatically promoted to hot cache.

---

#### `mining_status`

Show pattern mining statistics.

**Returns**: `{enabled, promotion_threshold, candidates_ready, outputs_last_24h, candidates[]}`

---

#### `review_candidates`

Review mined patterns ready for promotion.

**Returns**: List of candidate patterns with id, pattern, type, occurrences, first_seen, last_seen.

---

#### `approve_candidate`

Approve a mined pattern, storing as memory and promoting to hot cache.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pattern_id` | int | Yes | - | Candidate pattern ID |
| `memory_type` | string | No | `"pattern"` | Type to assign |
| `tags` | list[str] | No | `null` | Tags to assign |

---

#### `reject_candidate`

Reject a mined pattern, removing from candidates.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pattern_id` | int | Yes | - | Pattern to reject |

---

### Seeding & Bootstrap

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
1. CLAUDE.md, .claude/CLAUDE.md
2. README.md, README
3. CONTRIBUTING.md
4. docs/README.md
5. ARCHITECTURE.md

**Returns**: `BootstrapResponse` with files_found, files_processed, memories_created, etc.

---

#### `seed_from_text`

Parse text content and create memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | Text to parse |
| `memory_type` | string | No | `"project"` | Memory type |
| `promote_to_hot` | bool | No | `false` | Promote all to hot cache |

Splits on paragraphs, list items, and numbered lists.

---

#### `seed_from_file`

Import memories from a file.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Path to file |
| `memory_type` | string | No | `"project"` | Memory type |
| `promote_to_hot` | bool | No | `false` | Promote to hot cache |

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

#### `unlink_memories`

Remove relationship(s) between memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from_memory_id` | int | Yes | - | Source memory |
| `to_memory_id` | int | Yes | - | Target memory |
| `relation_type` | string | No | `null` | Specific type, or all if null |

---

#### `get_related_memories`

Get memories related to a given memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to find relations for |
| `relation_type` | string | No | `null` | Filter by type |
| `direction` | string | No | `"both"` | `outgoing`, `incoming`, or `both` |

---

#### `relationship_stats`

Get knowledge graph statistics.

**Returns**: `{total_relationships, by_type, linked_memories}`

---

### Trust Management

#### `validate_memory`

Mark a memory as validated/confirmed useful. Increases trust score.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to validate |
| `reason` | string | No | `null` | `used_correctly` (+0.05), `explicitly_confirmed` (+0.15), `cross_validated` (+0.20) |
| `boost` | float | No | `null` | Custom boost (overrides reason default) |
| `note` | string | No | `null` | Context note |

---

#### `invalidate_memory`

Mark a memory as incorrect or outdated. Decreases trust score.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to invalidate |
| `reason` | string | No | `null` | `outdated` (-0.10), `partially_incorrect` (-0.15), `factually_wrong` (-0.30), `superseded` (-0.05), `low_utility` (-0.05) |
| `penalty` | float | No | `null` | Custom penalty |
| `note` | string | No | `null` | Context note |

---

#### `get_trust_history`

Get trust adjustment history for a memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory ID |
| `limit` | int | No | `20` | Max entries |

**Returns**: `TrustHistoryResponse` with entries and current_trust.

---

### Contradiction Detection

#### `find_contradictions`

Find memories that may contradict a given memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to check |
| `similarity_threshold` | float | No | `0.75` | Min similarity (same topic) |
| `limit` | int | No | `5` | Max contradictions |

---

#### `get_contradictions`

Get all memory pairs marked as contradictions.

---

#### `mark_contradiction`

Mark two memories as contradicting each other.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id_a` | int | Yes | - | First memory |
| `memory_id_b` | int | Yes | - | Second memory |

---

#### `resolve_contradiction`

Resolve a contradiction by keeping one memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id_a` | int | Yes | - | First memory |
| `memory_id_b` | int | Yes | - | Second memory |
| `keep_id` | int | Yes | - | Memory to keep (must be one of the two) |
| `resolution` | string | No | `"supersedes"` | `supersedes`, `delete`, or `weaken` |

---

### Session Tracking

#### `get_sessions`

Get recent conversation sessions.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | `20` | Max sessions |
| `project_path` | string | No | `null` | Filter by project |

---

#### `get_session`

Get details for a specific session.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session ID |

---

#### `get_session_memories`

Get all memories from a session.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session ID |
| `limit` | int | No | `100` | Max memories |

---

#### `cross_session_patterns`

Find content appearing across multiple sessions.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_sessions` | int | No | `2` | Minimum sessions |

---

#### `set_session_topic`

Set or update session topic.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session ID |
| `topic` | string | Yes | - | Topic description |

---

### Predictive Cache

#### `access_patterns`

Get learned access patterns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | No | `null` | Specific memory, or all if null |
| `min_count` | int | No | `2` | Minimum access count |
| `limit` | int | No | `20` | Max patterns |

---

#### `predict_next`

Predict which memories might be needed next.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to predict from |
| `threshold` | float | No | `null` | Min probability |
| `limit` | int | No | `null` | Max predictions |

---

#### `warm_cache`

Pre-warm hot cache with predicted memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | int | Yes | - | Memory to predict from |

---

#### `predictive_cache_status`

Get predictive cache system status.

**Returns**: `{enabled, config, stats}`

---

### Maintenance

#### `db_maintenance`

Run database maintenance (vacuum, analyze, auto-demote).

**Returns**: `MaintenanceResponse` with bytes_reclaimed, memory_count, auto_demoted_count.

---

#### `run_cleanup`

Comprehensive cleanup of stale data.

**Returns**: `{hot_cache_demoted, patterns_expired, logs_deleted, memories_deleted}`

---

#### `validate_embeddings`

Check if embedding model changed since database creation.

---

#### `db_info`

Get database path, size, schema version, and stats.

---

#### `embedding_info`

Get embedding provider and cache information.

---

#### `audit_history`

Get audit log for destructive operations.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | `50` | Max entries |
| `operation` | string | No | `null` | Filter by operation type |

---

#### `db_rebuild_vectors`

Rebuild all memory vectors with current embedding model.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `batch_size` | int | No | `100` | Memories per batch |

Use when switching models or fixing dimension mismatches.

---

#### `metrics_status`

Get observability metrics for recall, store, mining, and hot cache.

---

## MCP Resources

### `memory://hot-cache`

Auto-injectable system context with high-confidence patterns.

- Instant recall (no tool call needed)
- Auto-bootstraps from README.md, CLAUDE.md if empty
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

# Without promoting to hot cache
memory-mcp-cli bootstrap --no-promote

# JSON output
memory-mcp-cli --json bootstrap
```

### `memory-mcp-cli log-output`

Log content for pattern mining.

```bash
# From stdin
echo "Some content" | memory-mcp-cli log-output

# From argument
memory-mcp-cli log-output -c "Some content"

# From file
memory-mcp-cli log-output -f /path/to/file
```

### `memory-mcp-cli run-mining`

Run pattern extraction.

```bash
memory-mcp-cli run-mining --hours 24
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
| `EMBEDDING_BACKEND` | `auto` | `auto`, `mlx`, or `sentence-transformers` |

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

### Mining

| Variable | Default | Description |
|----------|---------|-------------|
| `MINING_ENABLED` | `true` | Enable pattern mining |
| `LOG_RETENTION_DAYS` | `7` | Days to retain output logs |
| `MINING_AUTO_APPROVE_ENABLED` | `true` | Auto-approve high-confidence patterns |
| `MINING_AUTO_APPROVE_CONFIDENCE` | `0.8` | Min confidence for auto-approval |
| `MINING_AUTO_APPROVE_OCCURRENCES` | `3` | Min occurrences for auto-approval |

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
