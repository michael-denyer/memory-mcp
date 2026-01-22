# Changelog

All notable changes to Memory MCP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.8] - 2026-01-22

### Added

- **New memory categories** - Better organization of mined patterns
  - `workflow` - Deployment and operational processes (ssh, curl, deploy scripts)
  - `snippet` - Code snippets with language markers (never promoted to hot cache)
  - `command` - Short CLI commands (never promoted to hot cache)
  - `reference` - External resources, URLs, documentation pointers
  - `observation` - Factual findings and discoveries

- **recategorize CLI command** - Re-run category inference on existing memories
  - `memory-mcp-cli recategorize --dry-run` to preview changes
  - `memory-mcp-cli recategorize --all` to recategorize all memories
  - Useful after category pattern updates

- **Auto-mark recalled memories as used** - Automatic helpfulness tracking
  - New `retrieval_auto_mark_used` setting (default: True)
  - Memories returned by `recall()` are automatically marked as used
  - Fixes helpfulness metrics showing 0% (previously required manual `mark_memory_used()` calls)

### Changed

- **Low-value categories never promoted** - `command` and `snippet` categories are blocked from hot cache promotion
  - These are easily discoverable or have low recall value
  - Keeps hot cache focused on high-value tacit knowledge

### Fixed

- **Dashboard sessions page** - Fixed SQL queries using wrong column names
  - `session_id` → `id`, `created_at` → `started_at`
  - Sessions page now loads correctly

- **Mining button UX** - Added loading spinner and "Running..." text while mining executes

## [0.5.7] - 2026-01-22

### Added

- **Dashboard enhancements** - Full-featured web dashboard at http://127.0.0.1:8765
  - **Mining page** (`/mining`) - Review and approve/reject mined patterns
  - **Injections page** (`/injections`) - Track hot cache/working-set injections over time
  - **Sessions page** (`/sessions`) - Browse conversation sessions and their memories
  - **Graph page** (`/graph`) - Knowledge graph visualization with force-directed layout
  - **Memories over time chart** - Bar chart on overview page showing daily memory counts
  - **Helpfulness metrics** - Trust score and used/retrieved counts in memory tables
  - **Category distribution** - Visual breakdown by memory category on overview

- **Auto-mining in hook** - Pattern mining now runs automatically after each Claude response
  - No manual intervention needed - patterns become memories immediately
  - High-confidence patterns stored as memories on first extraction
  - Hot cache promotion after reaching occurrence threshold

### Changed

- **Mining stores memories immediately** - Patterns no longer wait for 3+ occurrences
  - First extraction creates a memory (if confidence >= threshold)
  - Hot cache promotion still requires occurrence threshold
  - Existing patterns migrated to memories on first mining run

## [0.5.6] - 2026-01-22

### Fixed

- **log_output CLI now passes project_id** - Pattern mining was finding 0 outputs because logs were stored without project_id
  - Root cause: `log-output` CLI wasn't calling `get_current_project_id()`
  - Mining filters by project_id, so logs without it were invisible
  - Added regression tests to prevent this from recurring

### Added

- **Code map documentation** - Visual architecture guide with bidirectional source links
  - Mermaid diagrams for system overview, data flows, and schema
  - Tables mapping components to file:line locations
  - Quick navigation for key entry points

## [0.5.5] - 2026-01-22

### Added

- **Entity extraction for pattern mining** - Extracts technology and decision entities from Claude outputs
  - Technology entities: databases, frameworks, languages, tools with context-aware patterns
  - Decision entities: architecture/design choices with rationale extraction
  - Confidence scoring based on context quality (rationale, alternatives boost confidence)

- **MENTIONS relation type** - New knowledge graph relation for entity linking
  - Source memories auto-link to extracted technology entities via MENTIONS
  - Decision entities auto-link to mentioned technologies via DEPENDS_ON
  - Creates rich semantic connections during mining

- **Auto-linking during mining** - `run_mining()` now creates knowledge graph links
  - Tracks which memories came from each output log via `source_log_id`
  - Links source content to extracted entities automatically
  - New helper: `get_memories_by_source_log()` for provenance queries

## [0.5.4] - 2026-01-22

### Added

- **Helpfulness tracking** - Track which memories are actually useful (schema v16)
  - `retrieved_count` - how often memory appears in recall results
  - `used_count` - how often memory is marked as helpful
  - `last_used_at` - when memory was last marked as used
  - `utility_score` - precomputed Bayesian helpfulness score

- **Bayesian helpfulness scoring** - Uses Beta-Binomial posterior `(used + α) / (retrieved + α + β)`
  - Cold start gives benefit of doubt (0.25)
  - Low utility detection emerges naturally from retrievals without usage
  - New helper: `get_bayesian_helpfulness()` in helpers.py

- **Helpfulness-weighted recall ranking** - New `recall_helpfulness_weight` config (default 0.05)
  - `utility_score` now factors into composite recall score
  - Memories that prove helpful rank higher

- **Used-rate promotion gate** - Auto-promotion now requires helpfulness signal
  - If `retrieved_count >= 5`, requires `used_rate >= 0.25` (25% usage)
  - Memories below warmup threshold get benefit of doubt
  - Prevents low-utility memories from being promoted to hot cache

- **Hot cache session recency ordering** - Hot memories ordered for optimal injection
  - Primary: `last_used_at` (most recently helpful first)
  - Secondary: `trust_score` (reliability)
  - Tertiary: `last_accessed_at` (general recency)

## [0.5.3] - 2026-01-22

### Added

- **ML-based category classification** - Uses embedding similarity to category prototypes instead of regex
  - Categories: antipattern, landmine, decision, convention, preference, lesson, constraint, architecture, context, bug, todo
  - Hybrid approach: ML first, falls back to regex for explicit patterns
  - Configurable via `ML_CLASSIFICATION_ENABLED` (default: true) and `ML_CLASSIFICATION_THRESHOLD` (default: 0.40)

- **Category-aware hot cache thresholds** - High-value categories promoted faster
  - `antipattern` and `landmine` categories get 0.3 salience threshold (vs 0.5 default)
  - Temporal-scope-aware demotion: durable categories (2x), stable (1x), transient (0.5x)

- **Feedback nudge in hot cache** - Hot cache resource now includes memory IDs and a hint to call `mark_memory_used(memory_id)` when a memory was helpful

- **Web dashboard fixes** - Fixed hot cache stats display (`current_count` vs `count`) and pinned status

### Changed

- **NER now standard dependency** - `transformers` moved from optional `[ner]` to core dependencies
  - NER entity extraction enabled by default during pattern mining
  - Can be disabled via `NER_ENABLED=false`

## [0.5.1] - 2026-01-21

### Fixed

- **Project-scoped mining** - Mining now respects project boundaries
  - Output logs store `project_id` for filtering (schema v13)
  - `run_mining` only processes logs from current project
  - Prevents cross-project pattern leakage and auto-approval

- **API endpoint extraction** - Fixed malformed "GET /path /path" patterns
  - Regex was capturing both full match and path, causing duplication
  - Now correctly produces "GET /path" format

- **Config extraction security** - Hardened against secret leakage
  - Added `_may_contain_secrets()` filter to config pattern extraction
  - Tightened regex patterns to capture only safe descriptive values
  - Prevents sensitive data from being auto-approved to hot cache

## [0.5.0] - 2026-01-21

### Changed

- **Major internal refactoring** - Split large monolithic modules into focused packages
  - `storage.py` (4,577 lines) → `storage/` package with 16 mixin modules
  - `server.py` (2,268 lines) → `server/` package with 12 tool modules
  - No API changes - all imports remain backwards compatible
  - Each module now follows single responsibility principle
  - Easier to navigate, test, and maintain

## [0.4.5] - 2026-01-21

### Added

- **Enhanced pattern mining** - Expanded from 5 to 16 pattern types for better extraction coverage
  - New extractors: decisions, architecture, tech stack, explanations, config, dependencies, API endpoints
  - NER-based entity extraction (person, organization, location) when `transformers` is installed
  - Auto-enables NER with `uv tool install hot-memory-mcp[ner]` - no config needed

- **Bootstrap context preservation** - Markdown files now preserve section context in chunks
  - Chunks include source file and section: `[CLAUDE.md > Testing] Use pytest for tests`
  - Short fact-like items (`Port: 8080`) are preserved instead of being filtered
  - Non-markdown files get source file prefix: `[README.txt] ...`

### Fixed

- **Project-aware deduplication** - Same content in different projects now stays separate
  - Content hash includes project_id to prevent cross-project merging
  - Semantic dedup search is now project-scoped

- **Recency ranking** - Recall now uses `last_accessed_at` instead of `created_at`
  - Recently accessed memories rank higher, improving relevance

- **Secret detection in mining** - Config extraction no longer captures env var values
  - Sensitive patterns (passwords, API keys, tokens) are filtered from auto-approval
  - Only env var names are stored, never values

- **NER context** - Named entities now include surrounding context for better recall
  - Format: `...context... [Entity is a organization]` instead of bare entity

- **Mining provenance** - Auto-approved patterns now preserve `source_log_id` for traceability

- **Transaction nesting** - Fixed `clear_vectors` to avoid nested transactions

## [0.4.4] - 2026-01-21

### Fixed

- **Hook transcript extraction** - Fixed jq selector to use correct `.message.content` path for Claude Code transcript format. Previously logged raw JSON instead of extracted text.
- **Pattern mining auto-approval** - Lowered `mining_auto_approve_confidence` default from 0.8 to 0.5 to match extractor defaults. Patterns meeting occurrence threshold now auto-approve as intended.

## [0.4.3] - 2026-01-21

### Fixed

- Removed unused import in mining module
- Use correct settings variable in mining provenance

## [0.4.2] - 2026-01-21

### Added

- **Hybrid search** - Combines semantic similarity with keyword matching for improved recall
  - FTS5 full-text search table synced with memories via triggers
  - Boosts results when queries use indirect phrasings (e.g., "FastAPI" matches "framework")
  - Configurable via `MEMORY_MCP_HYBRID_SEARCH_ENABLED` (default: true)
  - Adjustable keyword weight and boost threshold settings

### Changed

- Database schema version bumped to 12 (auto-migrates existing databases)

## [0.4.1] - 2026-01-21

### Fixed

- Use full server name in mcp-name for registry validation

### Added

- Release skill for automated publishing workflow

## [0.4.0] - 2026-01-21

### Added

- **Project awareness** - Memories are automatically tagged with the current git project
  - Auto-detects project from git remote URL (e.g., `github/owner/repo`)
  - Recall and hot cache filter to current project + global memories
  - `memory://project-context` MCP resource shows project-specific context
  - Configurable via `MEMORY_MCP_PROJECT_AWARENESS_ENABLED` (default: true)
  - Seamless switching between projects - each sees its own relevant memories

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

- **Unified salience score** - Engram-inspired metric for promotion decisions
  - Combines importance, trust, access count, and recency
  - Configurable weights for each component
  - Used for smarter hot cache promotion alongside access count threshold

- **Semantic clustering for display** - RePo-inspired cognitive load reduction
  - Groups similar memories in hot cache and working set displays
  - Auto-generates human-readable cluster labels from tags
  - Configurable threshold (0.70 default), max clusters (5), and min size (2)

### Changed

- Hot cache promotion now considers both access count threshold AND salience score
- `remember()` tool accepts new `episodic` memory type

## [0.3.0] - 2026-01-19

### Added

- **Research-inspired memory features**
  - Importance scoring at admission time
  - Retrieval tracking to learn which memories are actually used
  - Memory consolidation infrastructure

- **Per-result scoring** for recall transparency
  - Returns similarity, recency, and composite scores in results
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
  - Hot cache with instant recall via MCP resource injection
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
  - validate_memory() and invalidate_memory() tools
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

[0.4.2]: https://github.com/michael-denyer/memory-mcp/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/michael-denyer/memory-mcp/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/michael-denyer/memory-mcp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/michael-denyer/memory-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/michael-denyer/memory-mcp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/michael-denyer/memory-mcp/releases/tag/v0.1.0
