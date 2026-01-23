# Changelog

All notable changes to Memory MCP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.3] - 2026-01-23

### Added

- **Auto-link related memories** - Automatically creates knowledge graph links when storing
  - New memories are linked to semantically similar existing memories
  - Configurable: `MEMORY_MCP_AUTO_LINK_ENABLED` (default: true)
  - Threshold: `MEMORY_MCP_AUTO_LINK_THRESHOLD` (default: 0.75)
  - Max links: `MEMORY_MCP_AUTO_LINK_MAX` (default: 3)

- **Auto-detect contradictions** - Flags potential conflicts automatically
  - Very similar memories (same topic) are marked as `CONTRADICTS`
  - Helps identify outdated or conflicting information
  - Configurable: `MEMORY_MCP_AUTO_DETECT_CONTRADICTIONS` (default: true)
  - Threshold: `MEMORY_MCP_CONTRADICTION_THRESHOLD` (default: 0.80)

- **Dashboard version display** - Shows Memory MCP and MCP versions in footer

### Fixed

- **Hook script project_id bug** - Hook now passes project_id explicitly to CLI
  - Previously, `cd` to MEMORY_MCP_DIR broke project_id derivation
  - Now extracts project_path from hook input and passes via `--project-id`

- **CLI session_id support** - Hook now passes session_id for provenance tracking
  - `log-output` and `run-mining` accept `--session-id` option
  - Hook extracts session_id from input and passes to CLI

- **Nested transcript.path parsing** - CLI now handles `{transcript: {path: ...}}` format
  - Previously only checked `transcript_path` and `transcriptPath`

- **server.json version sync** - Version now matches pyproject.toml

## [0.6.2] - 2026-01-23

### Added

- **PreCompact hook** - Automatically save session memories before conversation compaction
  - New `memory-mcp-cli pre-compact` CLI command
  - Promotes episodic memories to long-term storage before context is lost
  - Silent operation - never blocks compaction

- **Memory Analyst agent** - Specialized agent for memory health analysis
  - 5-phase analysis workflow (overview, quality, issues, sessions, recommendations)
  - Detects contradictions, consolidation opportunities, stale memories
  - Provides prioritized recommendations (critical/recommended/optional)
  - Use via Task tool with `subagent_type: "memory-analyst"`

### Fixed

- **CI linting** - Formatted 3 files that were failing ruff format check

## [0.6.1] - 2026-01-23

### Added

- **Slash commands for Claude Code** - 14 new `/memory-mcp:*` commands
  - `/memory-mcp:remember` - Store a new memory
  - `/memory-mcp:recall` - Search memories semantically
  - `/memory-mcp:list` - List stored memories
  - `/memory-mcp:hot-cache` - Manage hot cache (promote/demote/pin/unpin)
  - `/memory-mcp:stats` - Show memory statistics
  - `/memory-mcp:bootstrap` - Bootstrap from project docs
  - `/memory-mcp:link` - Link related memories in knowledge graph
  - `/memory-mcp:session` - Manage conversation sessions
  - `/memory-mcp:mining` - Pattern mining from usage
  - `/memory-mcp:trust` - Manage memory trust scores
  - `/memory-mcp:consolidate` - Consolidate duplicate memories
  - `/memory-mcp:forget` - Delete a memory permanently
  - `/memory-mcp:maintenance` - Run database maintenance
  - `/memory-mcp:test-all` - Comprehensive interactive testing suite

## [0.6.0] - 2026-01-23

### Added

- **Claude Code plugin support** - Install as a plugin for automatic integration
  - `.claude-plugin/plugin.json` manifest with MCP server config and hooks
  - `skills/memory-mcp/SKILL.md` comprehensive usage guide
  - Install via: `claude plugins add michael-denyer/memory-mcp`

- **SessionStart hook** - Auto-bootstrap hot cache from project docs
  - Runs `memory-mcp-cli bootstrap --quiet` on session start
  - Automatically seeds from CLAUDE.md, README.md, etc.

- **Stop hook with log-response** - Automatic pattern mining from Claude's responses
  - New `memory-mcp-cli log-response` command for hook integration
  - Reads transcript, extracts last response, logs for mining

- **Bootstrap --quiet flag** - Suppress output for hook usage

## [0.5.17] - 2026-01-23

### Added

- **MLX auto-install on Apple Silicon** - No more manual `[mlx]` extra needed
  - `mlx-embeddings` now included by default for `darwin + arm64`
  - 10x faster embeddings work out of the box on M1/M2/M3 Macs

### Changed

- **Dashboard timestamp shows time** - Date and HH:MM now displayed in two lines
  - Previously only showed date; time helps distinguish same-day memories

### Fixed

- **vec0 vector insert errors** - Handle orphaned/duplicate vector entries
  - vec0 extension doesn't support INSERT OR REPLACE
  - Now DELETE before INSERT in both `store_memory()` and `rebuild_vectors()`
  - Fixes potential "UNIQUE constraint failed" errors during vector operations

## [0.5.15] - 2026-01-23

### Added

- **Intent-based recall ranking** - Boost memories matching query intent
  - `infer_query_intent()` detects query purpose (debugging, howto, architecture, etc.)
  - `compute_intent_boost()` applies category-based ranking boost (up to 15%)
  - Cheap regex heuristic with zero latency impact
  - Intents: debugging, howto, architecture, decision, convention, setup, api, todo

- **Mining minimum length threshold** - Skip short fragments
  - `mining_min_pattern_length` config (default: 30 chars)
  - Prevents storing noise like "Yes" or "OK" as memories

- **Dashboard timestamp column** - Memory creation dates visible
  - "Created" column shows when each memory was stored
  - Helps distinguish new issues from legacy problems

### Changed

- **Command/snippet blocking in mining** - Low-value patterns stored as mined_patterns only
  - Commands and snippets no longer auto-stored as memories
  - Still tracked in mined_patterns for reference if needed

## [0.5.14] - 2026-01-23

### Added

- **Injection logging in recall** - Track which memories are retrieved for feedback analysis
  - `recall()` now logs all returned memories via `log_injections_batch()`
  - Enables injection pattern analysis and hot cache quality improvement

- **Hot cache metric persistence** - Metrics survive server restarts
  - Promotions, hits, misses, evictions stored in metadata table
  - Lazy-loaded on first access for performance

- **Session summarization tool** - Extract structured knowledge from conversations
  - `summarize_session(session_id)` groups memories by category
  - Categories: decisions, insights (lessons/antipatterns/landmines), action items (todos/bugs), context
  - Returns top 20 items per group sorted by importance
  - Use before `end_session()` to review what will be promoted

### Changed

- **Category gate for auto-promotion** - `command` and `snippet` categories blocked
  - Low-value categories now rejected with `category_ineligible` reason
  - `should_promote_category()` helper for promotion decisions

- **Knowledge graph links for existing entities** - Entity patterns now tracked even when merged
  - Fixed bug where semantic dedup prevented relationship creation
  - Entities linked regardless of `is_new` status

### Fixed

- **run_mining KeyError** - MCP tool was reading `new_patterns` instead of `new_memories`
- **Staleness thresholds** - Now use `MemoryType` enum keys for type safety
- **Recall logging level** - Changed from DEBUG to INFO per changelog claim
- **Auto-promotion logging** - Changed from INFO to DEBUG (routine operation)

## [0.5.12] - 2026-01-23

### Added

- **Secret redaction** - Defense-in-depth for output logging
  - New `redaction.py` module with `may_contain_secrets()` and `redact_secrets()` functions
  - Secrets redacted BEFORE storage in `log_output()` to prevent persistence
  - Patterns: API keys (OpenAI, GitHub, AWS), connection strings, bearer tokens, key-value secrets

- **Staleness indicators** - Visual hints for memory freshness in hot cache
  - `_get_staleness_indicator()` returns "never verified" or "stale Xd" based on `last_used_at`
  - Type-specific thresholds: episodic (3d), conversation (7d), pattern (14d), project (21d), reference (30d)
  - Integrated into formatted memory output for injection context

- **Injection feedback loop** - Hot cache quality improvement from usage patterns
  - `analyze_injection_patterns()` identifies high-value, low-utility, and promotion candidates
  - `improve_hot_cache_from_injections()` auto-promotes high-value cold memories
  - Runs during `run_full_cleanup()` (dry_run=False, actual promotions happen)

- **Recall logging enhancements** - Better observability for debugging
  - Logs query preview (first 50 chars), mode, and elapsed time
  - INFO level for visibility into recall patterns

- **Ranking explanation improvements** - More complete factor breakdown
  - `build_ranking_factors()` now includes trust_weight, helpfulness_weight
  - Shows "[keyword boost active]" when hybrid search triggers

### Changed

- **Usage-aware trust decay** - Frequently-used memories decay slower
  - Log-based multiplier: `min(1.5, 1.0 + 0.1 * log(access_count + 1))`
  - Applied in both recall scoring and hot cache ordering
  - Rewards memories that prove useful over time

- **Hot cache ordering uses decayed trust** - Penalizes stale items
  - Sort key: session recency → decayed trust → real usage ratio
  - `real_usage_ratio = used_count / access_count` filters auto-marked noise
  - Prevents stale high-trust items from dominating injection

- **Log level optimization** - Reduced noise, better signal
  - Eviction: DEBUG → WARNING (cache pressure is notable)
  - Auto-promotion: INFO → DEBUG (routine operation)
  - Trust changes: INFO for Δ ≥ 0.1, DEBUG for minor tweaks

## [0.5.11] - 2026-01-23

### Added

- **Promotion rejection tracking** - Observability for hot cache promotion decisions
  - `record_promotion_rejection()` tracks reasons: category_ineligible, threshold_not_met, low_helpfulness
  - `get_promotion_rejection_summary()` exposes counts in hot cache stats
  - Helps tune thresholds and understand promotion patterns

- **Exact-match pattern promotion** - Prevent semantic drift in mining
  - `memory_id` column on mined_patterns (schema v17) links patterns to created memories
  - `link_pattern_to_memory()` creates the link during pattern storage
  - Auto-promotion prefers exact match, falls back to semantic search

- **Structured logging for scoring functions** - DEBUG logs for all scoring decisions
  - `infer_category()`, `compute_importance_score()`, `compute_salience_score()`
  - `get_bayesian_helpfulness()`, `compute_helpfulness_with_decay()`
  - Parameter breakdown helps debug promotion/ranking decisions

### Changed

- **Salience weights rebalanced for usage** - Hot cache now access-heavy
  - Access: 0.25 → 0.40 (usage is key signal)
  - Recency: 0.25 → 0.30 (recent patterns more relevant)
  - Trust: 0.25 → 0.15 (still matters but secondary)
  - Importance: 0.25 → 0.15 (content value secondary to usage)

- **Cold-start threshold lowered** - Faster feedback loop
  - Bayesian helpfulness gate now triggers at 3 retrievals (was 5)
  - New memories get benefit of doubt; heavily-retrieved-but-unused blocked

## [0.5.10] - 2026-01-23

### Added

- **Insight extractor** - Extract long-form contextual content from output logs
  - Pattern matching for architectural decisions, conventions, lessons learned

## [0.5.9] - 2026-01-23

### Added

- **Category-aware helpfulness decay** - Transient categories decay faster
  - `compute_helpfulness_with_decay()` applies per-category TTL
  - `todo`, `bug`, `context` decay in 7-14 days
  - `constraint`, `architecture` persist 60-90 days
  - Integrated into recall scoring for dynamic ranking

- **Auto-pin for high-value memories** - Prevent eviction of critical guardrails
  - `should_auto_pin()` helper for constraint/antipattern/landmine categories
  - Requires trust >= 0.8 to auto-pin (validated content only)
  - `constraint` added to `_HIGH_VALUE_CATEGORIES` for lower promotion threshold

- **Low-utility memory penalty** - Auto-weaken frequently-recalled but unused memories
  - `penalize_low_utility_memories()` runs during maintenance
  - Trust -= 0.03 when retrieved >= 5 times but used_count = 0
  - Helps demote noise from recall results

- **Input validation for list_memories** - Reject invalid pagination params
  - Returns error for offset < 0 or limit < 1 instead of undefined behavior

### Changed

- **Trust now affects recall ranking** - `recall_trust_weight` default: 0.0 → 0.1
  - Trusted memories rank higher in composite score
  - Still a small factor vs similarity (0.7 weight)

- **LOW_UTILITY trust penalty reduced** - From -0.05 to -0.03
  - More gradual decay for low-utility memories

### Documentation

- **ML classification guidance** - Added docstring clarifying appropriate uses
  - Confidence is for category assignment, NOT ranking
  - Use composite_score for ranking, hot_score for eviction

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
  - High-confidence patterns stored as memories on first extraction (if content ≥ 20 chars)
  - Hook processes only text blocks from Claude's response
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
