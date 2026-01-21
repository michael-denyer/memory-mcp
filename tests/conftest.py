"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_mcp.config import Settings
from memory_mcp.embeddings import CachedEmbeddingProvider, MockEmbeddingProvider

# Shared mock provider for all tests (avoids recreating per fixture call)
_mock_provider = MockEmbeddingProvider(dimension=384)
_cached_mock = CachedEmbeddingProvider(_mock_provider, cache_size=100)


class MockEmbeddingEngine:
    """Test double for EmbeddingEngine that uses MockEmbeddingProvider.

    Avoids loading the ~90MB sentence-transformer model during tests.
    """

    def __init__(self, settings=None):
        self.settings = settings
        self._provider = _cached_mock

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed(self, text: str):
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]):
        return self._provider.embed_batch(texts)

    def similarity(self, e1, e2) -> float:
        return self._provider.similarity(e1, e2)

    def cache_stats(self) -> dict:
        return self._provider.cache_stats()

    def clear_cache(self) -> None:
        self._provider.clear_cache()


@pytest.fixture(autouse=True)
def mock_embedding_engine():
    """Auto-use fixture that replaces real embeddings with mock for all tests.

    This dramatically speeds up tests by avoiding the ~90MB model load.
    Embeddings are deterministic based on content hash for reproducibility.
    """
    with (
        patch("memory_mcp.embeddings.get_embedding_engine", MockEmbeddingEngine),
        patch("memory_mcp.embeddings.EmbeddingEngine", MockEmbeddingEngine),
        patch("memory_mcp.storage.core.EmbeddingEngine", MockEmbeddingEngine),
    ):
        yield


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.dict("os.environ", {"MEMORY_MCP_DB_PATH": str(db_path)}):
            yield db_path


@pytest.fixture
def temp_settings():
    """Create settings with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield Settings(db_path=db_path)
