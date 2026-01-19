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

    # Hot cache
    hot_cache_max_items: int = Field(default=20, description="Maximum items in hot cache")
    promotion_threshold: int = Field(
        default=3, description="Access count to promote to hot cache"
    )
    demotion_days: int = Field(
        default=14, description="Days without access before demotion"
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
    trust_score_mined: float = Field(
        default=0.7, description="Trust score for mined memories"
    )
    trust_decay_halflife_days: float = Field(
        default=90.0, description="Half-life in days for trust decay"
    )
    recall_trust_weight: float = Field(
        default=0.0, description="Weight for trust score in recall ranking (0 to disable)"
    )

    # Input limits
    max_content_length: int = Field(
        default=100_000, description="Maximum content length for memories/logs"
    )
    max_recall_limit: int = Field(default=100, description="Maximum results per recall")
    max_tags: int = Field(default=20, description="Maximum tags per memory")

    model_config = {"env_prefix": "MEMORY_MCP_"}


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()


def ensure_data_dir(settings: Settings) -> None:
    """Ensure data directory exists."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
