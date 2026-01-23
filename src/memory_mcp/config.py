"""Configuration settings for memory MCP server."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Memory MCP configuration."""

    # Database
    db_path: Path = Field(
        default=Path.home() / ".memory-mcp" / "memory.db",
        description="Path to SQLite database",
    )

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    embedding_dim: int = Field(default=384, description="Embedding dimension")
    embedding_backend: str = Field(
        default="auto",
        description=(
            "Embedding backend: 'auto' (MLX on Apple Silicon, else sentence-transformers), "
            "'mlx' (force MLX), 'sentence-transformers' (force ST)"
        ),
    )

    # Hot cache
    hot_cache_max_items: int = Field(default=20, description="Maximum items in hot cache")
    promotion_threshold: int = Field(default=3, description="Access count to promote to hot cache")
    demotion_days: int = Field(default=14, description="Days without access before demotion")
    auto_promote: bool = Field(
        default=True, description="Auto-promote memories when access count reaches threshold"
    )
    auto_demote: bool = Field(
        default=True, description="Auto-demote stale hot memories during maintenance"
    )
    auto_bootstrap: bool = Field(
        default=False,
        description="Auto-bootstrap from markdown files when hot cache is empty",
    )
    hot_cache_display_max_chars: int = Field(
        default=150,
        description="Max chars per item in hot cache resource (truncates for context efficiency)",
    )

    # Hot cache scoring weights (for LRU eviction)
    hot_score_access_weight: float = Field(
        default=1.0, description="Weight for access_count in hot score"
    )
    hot_score_recency_weight: float = Field(
        default=0.5, description="Weight for recency boost in hot score"
    )
    hot_score_recency_halflife_days: float = Field(
        default=7.0, description="Half-life in days for recency decay"
    )

    # Mining
    mining_enabled: bool = Field(default=True, description="Enable pattern mining")
    mining_min_pattern_length: int = Field(
        default=30, description="Minimum character length for mined patterns (skip short fragments)"
    )
    log_retention_days: int = Field(default=7, description="Days to retain output logs")

    # Auto-approve high-confidence patterns (reduces manual intervention)
    mining_auto_approve_enabled: bool = Field(
        default=True, description="Auto-approve patterns meeting confidence/occurrence thresholds"
    )
    mining_auto_approve_confidence: float = Field(
        default=0.5, description="Minimum confidence for auto-approval"
    )
    mining_auto_approve_occurrences: int = Field(
        default=3, description="Minimum occurrences for auto-approval"
    )

    # NER-based entity extraction (requires optional transformers dependency)
    ner_enabled: bool = Field(
        default=True, description="Enable NER entity extraction during pattern mining"
    )
    ner_confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence for NER entity extraction (0-1)"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    log_format: str = Field(
        default="pretty", description="Log format: 'pretty' (human-readable) or 'json' (structured)"
    )

    # Retrieval
    default_recall_limit: int = Field(default=5, description="Default recall result limit")
    default_confidence_threshold: float = Field(
        default=0.7, description="Default similarity threshold for recall"
    )
    high_confidence_threshold: float = Field(
        default=0.85, description="Threshold for high confidence results"
    )

    # Recall scoring weights (composite ranking)
    recall_similarity_weight: float = Field(
        default=0.7, description="Weight for semantic similarity in recall score"
    )
    recall_recency_weight: float = Field(
        default=0.2, description="Weight for recency in recall score"
    )
    recall_access_weight: float = Field(
        default=0.1, description="Weight for access count in recall score"
    )
    recall_recency_halflife_days: float = Field(
        default=30.0, description="Half-life in days for recency decay in recall"
    )

    # Trust scoring
    trust_score_manual: float = Field(
        default=1.0, description="Trust score for manually added memories"
    )
    trust_score_mined: float = Field(default=0.7, description="Trust score for mined memories")
    trust_decay_halflife_days: float = Field(
        default=90.0, description="Default half-life in days for trust decay"
    )
    recall_trust_weight: float = Field(
        default=0.1, description="Weight for trust score in recall ranking (0 to disable)"
    )
    recall_helpfulness_weight: float = Field(
        default=0.05,
        description="Weight for helpfulness (utility_score) in recall ranking",
    )

    # Per-memory-type trust decay rates (in days)
    # Project facts decay slowest (architecture rarely changes)
    # Patterns decay faster (code evolves)
    # Conversation facts decay fastest (context-dependent)
    trust_decay_project_days: float = Field(
        default=180.0, description="Trust decay half-life for project memories"
    )
    trust_decay_pattern_days: float = Field(
        default=60.0, description="Trust decay half-life for pattern memories"
    )
    trust_decay_reference_days: float = Field(
        default=120.0, description="Trust decay half-life for reference memories"
    )
    trust_decay_conversation_days: float = Field(
        default=30.0, description="Trust decay half-life for conversation memories"
    )
    trust_decay_episodic_days: float = Field(
        default=7.0, description="Trust decay half-life for episodic memories"
    )

    # Confidence-weighted trust updates
    trust_auto_strengthen_on_recall: bool = Field(
        default=True, description="Auto-strengthen trust on high-similarity recall"
    )
    trust_high_similarity_threshold: float = Field(
        default=0.90, description="Similarity threshold for auto trust boost"
    )
    trust_high_similarity_boost: float = Field(
        default=0.03, description="Trust boost for high-similarity recall"
    )

    # Input limits
    max_content_length: int = Field(
        default=100_000, description="Maximum content length for memories/logs"
    )
    max_recall_limit: int = Field(default=100, description="Maximum results per recall")
    max_tags: int = Field(default=20, description="Maximum tags per memory")

    # Memory retention (days before archival, 0 = never expire)
    # These control auto-cleanup of old unused memories
    retention_project_days: int = Field(
        default=0, description="Days to retain project memories (0 = forever)"
    )
    retention_pattern_days: int = Field(
        default=180, description="Days to retain pattern memories (0 = forever)"
    )
    retention_reference_days: int = Field(
        default=365, description="Days to retain reference memories (0 = forever)"
    )
    retention_conversation_days: int = Field(
        default=90, description="Days to retain conversation memories (0 = forever)"
    )
    retention_episodic_days: int = Field(
        default=7, description="Days to retain episodic memories (0 = forever)"
    )

    # Episodic memory (session-bound short-term context)
    episodic_promote_top_n: int = Field(
        default=3, description="Top N episodic memories to promote on session end"
    )
    episodic_promote_threshold: float = Field(
        default=0.6, description="Minimum salience score for episodic promotion"
    )

    # Predictive hot cache warming (enabled by default for maximum value)
    predictive_cache_enabled: bool = Field(
        default=True, description="Enable predictive hot cache pre-warming"
    )
    prediction_threshold: float = Field(
        default=0.3, description="Minimum transition probability for prediction"
    )
    max_predictions: int = Field(default=3, description="Maximum memories to predict per recall")
    sequence_decay_days: int = Field(
        default=30, description="Days before access sequence counts decay"
    )

    # Semantic deduplication
    semantic_dedup_enabled: bool = Field(
        default=True, description="Merge semantically similar memories on store"
    )
    semantic_dedup_threshold: float = Field(
        default=0.92, description="Similarity threshold for merging (0.92 = very similar)"
    )

    # Memory consolidation (MemoryBank-inspired)
    consolidation_threshold: float = Field(
        default=0.85, description="Similarity threshold for consolidation clusters"
    )
    consolidation_min_cluster_size: int = Field(
        default=2, description="Minimum memories in a cluster to consolidate"
    )

    # Importance scoring at admission (MemGPT-inspired)
    importance_scoring_enabled: bool = Field(
        default=True, description="Score content importance at remember() time"
    )
    importance_length_weight: float = Field(
        default=0.3, description="Weight for content length in importance score"
    )
    importance_code_weight: float = Field(
        default=0.4, description="Weight for code content in importance score"
    )
    importance_entity_weight: float = Field(
        default=0.3, description="Weight for entity density in importance score"
    )

    # ML-based category classification
    ml_classification_enabled: bool = Field(
        default=True,
        description="Use ML (embedding similarity) for category inference instead of regex",
    )
    ml_classification_threshold: float = Field(
        default=0.40,
        description="Minimum similarity to category prototype for ML classification",
    )

    # Retrieval quality tracking (RAG-inspired)
    retrieval_tracking_enabled: bool = Field(
        default=True, description="Track which recalled memories were actually used"
    )
    retrieval_auto_mark_used: bool = Field(
        default=True,
        description=(
            "Auto-mark recalled memories as used. When True, memories returned by recall() "
            "are automatically marked as used since the LLM explicitly requested them."
        ),
    )

    # Salience scoring (Engram-inspired unified metric)
    # Combines importance + trust + access + recency for promotion/eviction decisions
    # Weights are access-heavy: hot cache exists to surface frequently-used patterns
    # Access (0.40) + Recency (0.30) = 70% usage-based, 30% content-based
    salience_importance_weight: float = Field(
        default=0.15, description="Weight for importance score in salience"
    )
    salience_trust_weight: float = Field(
        default=0.15, description="Weight for trust score in salience"
    )
    salience_access_weight: float = Field(
        default=0.40, description="Weight for normalized access count in salience"
    )
    salience_recency_weight: float = Field(
        default=0.30, description="Weight for recency in salience"
    )
    salience_recency_halflife_days: float = Field(
        default=14.0, description="Half-life for recency decay in salience"
    )
    salience_promotion_threshold: float = Field(
        default=0.5, description="Minimum salience score for auto-promotion (0-1)"
    )

    # Multi-hop recall via knowledge graph (Engram-inspired associative recall)
    recall_expand_relations: bool = Field(
        default=False, description="Expand recall results via knowledge graph relations"
    )
    recall_max_expansion: int = Field(
        default=3, description="Maximum related memories to add per recall result"
    )
    recall_expansion_decay: float = Field(
        default=0.8, description="Score decay for expanded results (0-1)"
    )

    # Working-set resource (Engram-inspired active memory)
    working_set_enabled: bool = Field(
        default=True, description="Enable memory://working-set resource"
    )
    working_set_max_items: int = Field(
        default=10, description="Maximum items in working-set resource"
    )
    working_set_recent_recalls_limit: int = Field(
        default=5, description="Recent recalls to include in working set"
    )
    working_set_predictions_limit: int = Field(
        default=3, description="Predicted memories to include in working set"
    )

    # Project awareness (per-project memory isolation)
    project_awareness_enabled: bool = Field(
        default=True, description="Enable automatic project detection from git"
    )
    project_filter_recall: bool = Field(
        default=True,
        description="Filter recall results to current project (with global fallback)",
    )
    project_filter_hot_cache: bool = Field(
        default=True, description="Filter hot cache to current project"
    )
    project_include_global: bool = Field(
        default=True, description="Include global (non-project) memories in results"
    )

    # Hook configuration warnings
    warn_missing_hook: bool = Field(
        default=True,
        description="Show warning on startup if Stop hook not configured (for pattern mining)",
    )

    # Semantic clustering for display (RePo-inspired cognitive load reduction)
    clustering_display_enabled: bool = Field(
        default=True, description="Group similar memories in display contexts (hot cache, recall)"
    )
    clustering_display_threshold: float = Field(
        default=0.70,
        description="Similarity threshold for display clustering (lower than consolidation)",
    )
    clustering_min_size: int = Field(default=2, description="Minimum items to form a named cluster")
    clustering_max_clusters: int = Field(
        default=5, description="Maximum distinct clusters before remaining go to 'Other'"
    )

    # Hybrid search (combines semantic + keyword matching)
    hybrid_search_enabled: bool = Field(
        default=True,
        description="Enable hybrid search combining semantic similarity with keyword matching",
    )
    hybrid_keyword_weight: float = Field(
        default=0.3,
        description="Weight for keyword score in hybrid ranking (0-1)",
    )
    hybrid_keyword_boost_threshold: float = Field(
        default=0.4,
        description="Semantic threshold below which keyword boost applies",
    )

    # Auto-link related memories on store (knowledge graph automation)
    auto_link_enabled: bool = Field(
        default=True, description="Auto-link semantically related memories on store"
    )
    auto_link_threshold: float = Field(
        default=0.75,
        description="Similarity threshold for auto-linking (lower than dedup, higher than recall)",
    )
    auto_link_max: int = Field(default=3, description="Maximum auto-links to create per new memory")

    # Auto-detect contradictions on store
    auto_detect_contradictions: bool = Field(
        default=True, description="Auto-detect potential contradictions when storing"
    )
    contradiction_threshold: float = Field(
        default=0.80,
        description="Similarity threshold for contradiction detection (same topic = conflict)",
    )

    # Recall mode presets
    # Precision mode: high threshold, few results, prioritize similarity
    precision_threshold: float = Field(
        default=0.8, description="Threshold for precision recall mode"
    )
    precision_limit: int = Field(default=3, description="Limit for precision recall mode")
    precision_similarity_weight: float = Field(
        default=0.85, description="Similarity weight for precision mode"
    )
    precision_recency_weight: float = Field(
        default=0.1, description="Recency weight for precision mode"
    )
    precision_access_weight: float = Field(
        default=0.05, description="Access weight for precision mode"
    )

    # Exploratory mode: low threshold, more results, balance factors
    exploratory_threshold: float = Field(
        default=0.5, description="Threshold for exploratory recall mode"
    )
    exploratory_limit: int = Field(default=10, description="Limit for exploratory recall mode")
    exploratory_similarity_weight: float = Field(
        default=0.5, description="Similarity weight for exploratory mode"
    )
    exploratory_recency_weight: float = Field(
        default=0.3, description="Recency weight for exploratory mode"
    )
    exploratory_access_weight: float = Field(
        default=0.2, description="Access weight for exploratory mode"
    )

    model_config = {"env_prefix": "MEMORY_MCP_"}


# Default files to auto-detect for bootstrap (priority order)
BOOTSTRAP_DEFAULT_FILES = (
    "CLAUDE.md",
    ".claude/CLAUDE.md",
    "README.md",
    "README",
    "CONTRIBUTING.md",
    "docs/README.md",
    "ARCHITECTURE.md",
)


def find_bootstrap_files(root: Path) -> list[Path]:
    """Find existing bootstrap files in a directory.

    Args:
        root: Directory to search for documentation files.

    Returns:
        List of existing file paths, in priority order.
    """
    return [root / f for f in BOOTSTRAP_DEFAULT_FILES if (root / f).exists()]


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()


def ensure_data_dir(settings: Settings) -> None:
    """Ensure data directory exists."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)


def check_stop_hook_configured() -> bool:
    """Check if Claude Code Stop hook is configured for memory output logging.

    Returns:
        True if hook is configured, False otherwise.
    """
    import json

    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return False

    try:
        with open(settings_path) as f:
            settings = json.load(f)

        hooks = settings.get("hooks", {})
        stop_hooks = hooks.get("Stop", [])

        # Check if any Stop hook references memory-mcp
        for hook_group in stop_hooks:
            for hook in hook_group.get("hooks", []):
                command = hook.get("command", "")
                if "memory" in command.lower() and "log" in command.lower():
                    return True

        return False
    except (json.JSONDecodeError, OSError):
        return False


def get_hook_install_instructions() -> str:
    """Get instructions for installing the Stop hook."""

    # Try to detect the memory-mcp install path
    script_path = Path(__file__).parent.parent.parent / "hooks" / "memory-log-response.sh"
    if not script_path.exists():
        script_path = Path("<path-to-memory-mcp>") / "hooks" / "memory-log-response.sh"

    return f"""Pattern mining requires a Claude Code hook to log outputs.

To install, ask Claude: "Add the memory-mcp Stop hook to my settings"

Or manually add to ~/.claude/settings.json:
{{
  "hooks": {{
    "Stop": [{{
      "matcher": "",
      "hooks": [{{
        "type": "command",
        "command": "{script_path}"
      }}]
    }}]
  }}
}}

To disable this warning, set MEMORY_MCP_WARN_MISSING_HOOK=false"""
