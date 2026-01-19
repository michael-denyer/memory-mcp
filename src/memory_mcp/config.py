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

    # Input limits
    max_content_length: int = Field(
        default=100_000, description="Maximum content length for memories/logs"
    )
    max_recall_limit: int = Field(default=100, description="Maximum results per recall")

    model_config = {"env_prefix": "MEMORY_MCP_"}


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()


def ensure_data_dir(settings: Settings) -> None:
    """Ensure data directory exists."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
