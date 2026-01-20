# Changelog

All notable changes to Memory MCP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Episodic memory type** - New `EPISODIC` memory type for session-bound short-term context
  - `end_session()` tool to consolidate episodic memories at session end
  - Promotes top memories by salience score to PROJECT or PATTERN type
  - Configurable retention (7 days default) and trust decay settings

- **Working-set resource** - `memory://working-set` MCP resource for session-aware active memory
  - Combines recently recalled memories, predicted next memories, and top salience hot items
  - Smaller and more focused than hot-cache for active work context

- **Multi-hop recall** - `expand_relations` parameter on recall for associative memory
  - Traverses knowledge graph relations to find related memories
  - Score decay for expanded results prevents dilution
  - Handles SUPERSEDES relation to skip outdated memories

- **Consolidation CLI** - `memory-mcp-cli consolidate` command for memory deduplication
  - Finds clusters of semantically similar memories
  - Merges near-duplicates, keeping best representative
  - `--dry-run` flag for preview, `--threshold` for custom similarity

- **Unified salience score** - Engram-inspired metric for promotion/eviction decisions
  - Combines importance, trust, access count, and recency
  - Configurable weights for each component
  - Used for smarter hot cache promotion alongside access count threshold

### Changed

- Hot cache promotion now considers both access count threshold AND salience score
- `remember()` tool accepts new `episodic` memory type

## [0.3.0] - 2026-01-19

### Added

- **Research-inspired memory features**
  - Importance scoring at admission time (MemGPT-inspired)
  - Retrieval tracking to learn which memories are actually used (RAG-inspired)
  - Memory consolidation infrastructure (MemoryBank-inspired)

- **Per-result scoring components** for recall transparency
  - Shows similarity, recency, access, and trust components in results
  - Helps understand why memories are ranked the way they are

- **Fine-grained trust management**
  - Contextual reasons for trust adjustments (USED_CORRECTLY, OUTDATED, etc.)
  - Audit trail for trust changes
  - Per-memory-type trust decay rates

### Changed

- Replaced argparse with click in CLI for better UX
- Improved documentation with troubleshooting guide

## [0.2.0] - 2026-01-18

### Added

- **Hot cache truncation** for context efficiency
  - Configurable max chars per item (default 150)
  - Prevents context bloat from long memories

- **Helper functions module** (`helpers.py`)
  - Extracted from server.py for cleaner organization
  - Content summarization, age formatting, confidence helpers

- **Database migrations module** (`migrations.py`)
  - Schema versioning (v1-v10)
  - Automatic migration on startup

- **Response models module** (`responses.py`)
  - Pydantic models for all MCP tool responses
  - Better type safety and documentation

- **Data models module** (`models.py`)
  - Enums and dataclasses separated from storage.py
  - Cleaner imports and organization

### Changed

- Refactored server.py to use helper functions
- Improved code organization across modules

## [0.1.0] - 2026-01-17

### Added

- **Two-tier memory architecture**
  - Hot cache with zero-latency MCP resource injection
  - Cold storage with semantic search via sqlite-vec

- **Auto-bootstrap** from project documentation
  - Detects README.md, CLAUDE.md, CONTRIBUTING.md, etc.
  - Seeds hot cache when empty

- **Pattern mining** from Claude outputs
  - Extracts imports, commands, project facts
  - Frequency-based promotion candidates
  - Human approval workflow

- **Knowledge graph** with typed relationships
  - RELATES_TO, DEPENDS_ON, SUPERSEDES, etc.
  - Link and unlink memories

- **Trust management**
  - strengthen_trust() and weaken_trust() tools
  - Trust decay over time by memory type

- **Session tracking** for provenance
  - Track which session created each memory
  - Cross-session pattern detection

- **Predictive hot cache warming**
  - Learns access patterns between memories
  - Pre-warms cache with predicted next memories

- **Apple Silicon optimization**
  - MLX backend auto-detected on M-series Macs
  - Falls back to sentence-transformers otherwise

- **CLI tools**
  - `bootstrap` - Seed from project docs
  - `seed` - Import from file
  - `log-output` - Log content for mining
  - `run-mining` - Extract patterns
  - `db-rebuild-vectors` - Fix dimension mismatches
  - `status` - Show system health

### Configuration

- Environment variables with `MEMORY_MCP_` prefix
- Sensible defaults for all settings
- Hot cache: 20 items max, 3 access threshold, 14 day demotion
- Retrieval: 0.7 confidence threshold, 5 result limit

[Unreleased]: https://github.com/michael-denyer/memory-mcp/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/michael-denyer/memory-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/michael-denyer/memory-mcp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/michael-denyer/memory-mcp/releases/tag/v0.1.0
