"""Tests for embedding provider interface."""

import numpy as np
import pytest

from memory_mcp.config import Settings
from memory_mcp.embeddings import (
    BaseEmbeddingProvider,
    CachedEmbeddingProvider,
    EmbeddingEngine,
    EmbeddingProvider,
    SentenceTransformerProvider,
    content_hash,
    create_provider,
)


class MockProvider(BaseEmbeddingProvider):
    """Mock provider for testing."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.embed_calls = 0
        self.batch_calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "mock"

    def embed(self, text: str) -> np.ndarray:
        self.embed_calls += 1
        # Deterministic fake embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)  # Normalize

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self.batch_calls += 1
        return [self.embed(t) for t in texts]


class TestEmbeddingProviderProtocol:
    """Tests for the EmbeddingProvider protocol."""

    def test_mock_provider_implements_protocol(self):
        """Mock provider should implement EmbeddingProvider protocol."""
        provider = MockProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_provider_properties(self):
        """Provider should expose dimension and name."""
        provider = MockProvider(dim=512)
        assert provider.dimension == 512
        assert provider.name == "mock"

    def test_embed_returns_correct_shape(self):
        """embed() should return array of correct dimension."""
        provider = MockProvider(dim=384)
        embedding = provider.embed("test text")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_batch_returns_list(self):
        """embed_batch() should return list of embeddings."""
        provider = MockProvider(dim=384)
        texts = ["first", "second", "third"]
        embeddings = provider.embed_batch(texts)
        assert len(embeddings) == 3
        for e in embeddings:
            assert e.shape == (384,)

    def test_embed_batch_empty(self):
        """embed_batch() should handle empty list."""
        provider = MockProvider()
        embeddings = provider.embed_batch([])
        assert embeddings == []

    def test_similarity(self):
        """similarity() should compute cosine similarity correctly."""
        provider = MockProvider()
        # Same text should have similarity ~1.0
        e1 = provider.embed("hello world")
        e2 = provider.embed("hello world")
        sim = provider.similarity(e1, e2)
        assert abs(sim - 1.0) < 0.001

        # Different texts should have lower similarity
        e3 = provider.embed("completely different xyz123")
        sim_diff = provider.similarity(e1, e3)
        assert sim_diff < 1.0


class TestCachedEmbeddingProvider:
    """Tests for the caching wrapper."""

    def test_cache_hit(self):
        """Second embed of same text should hit cache."""
        inner = MockProvider()
        cached = CachedEmbeddingProvider(inner, cache_size=100)

        # First call computes
        e1 = cached.embed("test")
        assert inner.embed_calls == 1

        # Second call hits cache
        e2 = cached.embed("test")
        assert inner.embed_calls == 1  # No new compute
        np.testing.assert_array_equal(e1, e2)

    def test_cache_miss(self):
        """Different texts should not hit cache."""
        inner = MockProvider()
        cached = CachedEmbeddingProvider(inner, cache_size=100)

        cached.embed("first")
        cached.embed("second")
        assert inner.embed_calls == 2

    def test_cache_eviction(self):
        """Cache should evict oldest entries when full."""
        inner = MockProvider()
        cached = CachedEmbeddingProvider(inner, cache_size=3)

        # Fill cache
        cached.embed("a")
        cached.embed("b")
        cached.embed("c")
        assert cached.cache_stats()["size"] == 3

        # Add one more, should evict oldest ("a")
        cached.embed("d")
        assert cached.cache_stats()["size"] == 3

        # "a" should now miss
        cached.embed("a")
        assert inner.embed_calls == 5  # 4 + 1 for re-computing "a"

    def test_cache_lru_update(self):
        """Accessing cached item should make it most recently used."""
        inner = MockProvider()
        cached = CachedEmbeddingProvider(inner, cache_size=3)

        cached.embed("a")
        cached.embed("b")
        cached.embed("c")

        # Access "a" again to make it most recent
        cached.embed("a")

        # Add two more, should evict "b" and "c", not "a"
        cached.embed("d")
        cached.embed("e")

        # "a" should still be cached
        initial_calls = inner.embed_calls
        cached.embed("a")
        assert inner.embed_calls == initial_calls  # No new computation

    def test_batch_caching(self):
        """Batch embed should use cache efficiently."""
        inner = MockProvider()
        cached = CachedEmbeddingProvider(inner, cache_size=100)

        # Pre-cache one item
        cached.embed("cached")

        # Batch with mix of cached and uncached
        texts = ["cached", "new1", "new2"]
        embeddings = cached.embed_batch(texts)

        assert len(embeddings) == 3
        # Only uncached items should trigger batch call
        assert inner.batch_calls == 1

    def test_cache_stats(self):
        """cache_stats() should return correct info."""
        inner = MockProvider()
        cached = CachedEmbeddingProvider(inner, cache_size=50)

        assert cached.cache_stats() == {
            "size": 0,
            "max_size": 50,
            "provider": "mock",
        }

        cached.embed("test")
        assert cached.cache_stats()["size"] == 1

    def test_clear_cache(self):
        """clear_cache() should empty the cache."""
        inner = MockProvider()
        cached = CachedEmbeddingProvider(inner, cache_size=100)

        cached.embed("a")
        cached.embed("b")
        assert cached.cache_stats()["size"] == 2

        cached.clear_cache()
        assert cached.cache_stats()["size"] == 0

        # Should recompute after clear
        cached.embed("a")
        assert inner.embed_calls == 3


class TestSentenceTransformerProvider:
    """Tests for the SentenceTransformer provider."""

    @pytest.fixture
    def provider(self):
        """Create a real SentenceTransformer provider."""
        return SentenceTransformerProvider(
            "sentence-transformers/all-MiniLM-L6-v2", 384
        )

    def test_lazy_loading(self, provider):
        """Model should not load until first use."""
        assert provider._model is None

    def test_embed_loads_model(self, provider):
        """embed() should lazy-load the model."""
        embedding = provider.embed("test")
        assert provider._model is not None
        assert embedding.shape == (384,)

    def test_normalized_embeddings(self, provider):
        """Embeddings should be normalized (unit length)."""
        embedding = provider.embed("test sentence")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.001

    def test_provider_name(self, provider):
        """Provider name should include model name."""
        assert "all-MiniLM-L6-v2" in provider.name


class TestCreateProvider:
    """Tests for the provider factory."""

    def test_creates_sentence_transformer_by_default(self):
        """Factory should create SentenceTransformerProvider by default."""
        settings = Settings(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        provider = create_provider(settings)
        assert isinstance(provider, SentenceTransformerProvider)

    def test_handles_model_with_slash(self):
        """Models with slash should use SentenceTransformer."""
        settings = Settings(embedding_model="custom/model")
        provider = create_provider(settings)
        assert isinstance(provider, SentenceTransformerProvider)


class TestEmbeddingEngine:
    """Tests for the legacy EmbeddingEngine wrapper."""

    @pytest.fixture
    def engine(self):
        """Create an EmbeddingEngine."""
        return EmbeddingEngine()

    def test_engine_has_dimension(self, engine):
        """Engine should expose dimension."""
        assert engine.dimension == 384

    def test_engine_caches(self, engine):
        """Engine should cache embeddings."""
        engine.embed("test")
        stats = engine.cache_stats()
        assert stats["size"] == 1

    def test_engine_clear_cache(self, engine):
        """Engine should support clearing cache."""
        engine.embed("test")
        engine.clear_cache()
        assert engine.cache_stats()["size"] == 0


class TestContentHash:
    """Tests for content_hash utility."""

    def test_deterministic(self):
        """Same content should produce same hash."""
        h1 = content_hash("test content")
        h2 = content_hash("test content")
        assert h1 == h2

    def test_different_content(self):
        """Different content should produce different hash."""
        h1 = content_hash("content a")
        h2 = content_hash("content b")
        assert h1 != h2

    def test_hash_format(self):
        """Hash should be hex string of correct length (SHA256 = 64 chars)."""
        h = content_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
