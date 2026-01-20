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
    log_retention_days: int = Field(default=7, description="Days to retain output logs")

    # Auto-approve high-confidence patterns (reduces manual intervention)
    mining_auto_approve_enabled: bool = Field(
        default=True, description="Auto-approve patterns meeting confidence/occurrence thresholds"
    )
    mining_auto_approve_confidence: float = Field(
        default=0.8, description="Minimum confidence for auto-approval"
    )
    mining_auto_approve_occurrences: int = Field(
        default=3, description="Minimum occurrences for auto-approval"
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
        default=0.0, description="Weight for trust score in recall ranking (0 to disable)"
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
