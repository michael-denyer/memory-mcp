"""Tests for embedding provider interface."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memory_mcp.config import Settings
from memory_mcp.embeddings import (
    MLX_MODEL_MAPPINGS,
    BaseEmbeddingProvider,
    CachedEmbeddingProvider,
    EmbeddingEngine,
    EmbeddingProvider,
    MLXEmbeddingProvider,
    SentenceTransformerProvider,
    _get_mlx_model_name,
    _should_use_mlx,
    content_hash,
    create_provider,
    is_apple_silicon,
    is_mlx_available,
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
        return SentenceTransformerProvider("sentence-transformers/all-MiniLM-L6-v2", 384)

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

    def test_force_sentence_transformers_backend(self):
        """Setting backend to sentence-transformers should force SentenceTransformerProvider."""
        settings = Settings(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_backend="sentence-transformers",
        )
        provider = create_provider(settings)
        assert isinstance(provider, SentenceTransformerProvider)

    def test_force_st_backend_shorthand(self):
        """Setting backend to 'st' should force SentenceTransformerProvider."""
        settings = Settings(
            embedding_model="custom/model",
            embedding_backend="st",
        )
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


# ========== MLX Provider Tests ==========


class TestPlatformDetection:
    """Tests for Apple Silicon and MLX detection functions."""

    def test_is_apple_silicon_darwin_arm64(self):
        """Should return True on Darwin arm64."""
        with patch("memory_mcp.embeddings.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "arm64"
            assert is_apple_silicon() is True

    def test_is_apple_silicon_darwin_x86(self):
        """Should return False on Darwin x86_64."""
        with patch("memory_mcp.embeddings.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "x86_64"
            assert is_apple_silicon() is False

    def test_is_apple_silicon_linux(self):
        """Should return False on Linux."""
        with patch("memory_mcp.embeddings.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "arm64"
            assert is_apple_silicon() is False

    def test_is_mlx_available_when_installed(self):
        """Should return True when mlx.core can be imported."""
        with patch.dict("sys.modules", {"mlx.core": MagicMock()}):
            # Need to reload to pick up the mock
            with patch("builtins.__import__", side_effect=lambda name, *args: MagicMock()):
                # Direct test: mlx.core import succeeds
                assert is_mlx_available() is True

    def test_is_mlx_available_when_not_installed(self):
        """Should return False when mlx.core import fails."""
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'mlx'")):
                assert is_mlx_available() is False


class TestMLXModelMappings:
    """Tests for MLX model name mappings."""

    def test_direct_mapping_exists(self):
        """Known models should have direct mappings."""
        assert "sentence-transformers/all-MiniLM-L6-v2" in MLX_MODEL_MAPPINGS
        assert "all-MiniLM-L6-v2" in MLX_MODEL_MAPPINGS

    def test_get_mlx_model_name_direct_mapping(self):
        """Should return mapped model for known models."""
        result = _get_mlx_model_name("sentence-transformers/all-MiniLM-L6-v2")
        assert result == "mlx-community/all-MiniLM-L6-v2-4bit"

    def test_get_mlx_model_name_short_name(self):
        """Should handle short model names."""
        result = _get_mlx_model_name("all-MiniLM-L6-v2")
        assert result == "mlx-community/all-MiniLM-L6-v2-4bit"

    def test_get_mlx_model_name_already_mlx(self):
        """Should pass through mlx-community models unchanged."""
        result = _get_mlx_model_name("mlx-community/custom-model")
        assert result == "mlx-community/custom-model"

    def test_get_mlx_model_name_unmapped_st_model(self):
        """Should construct mlx-community name for unmapped ST models."""
        result = _get_mlx_model_name("sentence-transformers/unknown-model")
        assert result == "mlx-community/unknown-model-4bit"

    def test_get_mlx_model_name_passthrough(self):
        """Should pass through unknown models unchanged."""
        result = _get_mlx_model_name("custom-model")
        assert result == "custom-model"


class TestShouldUseMLX:
    """Tests for _should_use_mlx backend selection logic."""

    def test_force_mlx_backend(self):
        """Should use MLX when backend is 'mlx'."""
        settings = Settings(embedding_backend="mlx")
        assert _should_use_mlx(settings) is True

    def test_force_sentence_transformers_backend(self):
        """Should not use MLX when backend is 'sentence-transformers'."""
        settings = Settings(embedding_backend="sentence-transformers")
        assert _should_use_mlx(settings) is False

    def test_force_st_shorthand(self):
        """Should not use MLX when backend is 'st'."""
        settings = Settings(embedding_backend="st")
        assert _should_use_mlx(settings) is False

    def test_auto_on_apple_silicon_with_mlx(self):
        """Should use MLX on Apple Silicon when available."""
        settings = Settings(embedding_backend="auto")
        with patch("memory_mcp.embeddings.is_apple_silicon", return_value=True):
            with patch("memory_mcp.embeddings.is_mlx_available", return_value=True):
                assert _should_use_mlx(settings) is True

    def test_auto_on_apple_silicon_without_mlx(self):
        """Should not use MLX on Apple Silicon when not installed."""
        settings = Settings(embedding_backend="auto")
        with patch("memory_mcp.embeddings.is_apple_silicon", return_value=True):
            with patch("memory_mcp.embeddings.is_mlx_available", return_value=False):
                assert _should_use_mlx(settings) is False

    def test_auto_on_non_apple(self):
        """Should not use MLX on non-Apple platforms."""
        settings = Settings(embedding_backend="auto")
        with patch("memory_mcp.embeddings.is_apple_silicon", return_value=False):
            with patch("memory_mcp.embeddings.is_mlx_available", return_value=True):
                assert _should_use_mlx(settings) is False

    def test_unknown_backend_falls_back_to_auto(self):
        """Unknown backend should fall back to auto-detection."""
        settings = Settings(embedding_backend="unknown")
        with patch("memory_mcp.embeddings.is_apple_silicon", return_value=True):
            with patch("memory_mcp.embeddings.is_mlx_available", return_value=True):
                assert _should_use_mlx(settings) is True


class TestMLXEmbeddingProvider:
    """Tests for MLXEmbeddingProvider class."""

    def test_provider_properties(self):
        """Provider should expose dimension and name."""
        provider = MLXEmbeddingProvider("mlx-community/test-model", 384)
        assert provider.dimension == 384
        assert provider.name == "mlx:mlx-community/test-model"

    def test_lazy_loading(self):
        """Model should not load until first use."""
        provider = MLXEmbeddingProvider("mlx-community/test-model", 384)
        assert provider._model is None
        assert provider._tokenizer is None

    def test_import_error_message(self):
        """Should provide helpful error when MLX not installed."""
        provider = MLXEmbeddingProvider("mlx-community/test-model", 384)

        with patch.dict("sys.modules", {"mlx_embeddings": None, "mlx_embeddings.utils": None}):
            with pytest.raises(ImportError, match="MLX dependencies not installed"):
                provider._get_model()


class TestCreateProviderWithMLX:
    """Tests for create_provider with MLX backend selection."""

    def test_creates_mlx_provider_when_forced(self):
        """Should create MLXEmbeddingProvider when backend is 'mlx'."""
        settings = Settings(embedding_backend="mlx")
        provider = create_provider(settings)
        assert isinstance(provider, MLXEmbeddingProvider)

    def test_creates_st_provider_when_forced(self):
        """Should create SentenceTransformerProvider when backend is 'st'."""
        settings = Settings(embedding_backend="st")
        provider = create_provider(settings)
        assert isinstance(provider, SentenceTransformerProvider)

    def test_creates_mlx_provider_on_apple_silicon(self):
        """Should create MLXEmbeddingProvider on Apple Silicon with auto."""
        settings = Settings(embedding_backend="auto")
        with patch("memory_mcp.embeddings.is_apple_silicon", return_value=True):
            with patch("memory_mcp.embeddings.is_mlx_available", return_value=True):
                provider = create_provider(settings)
                assert isinstance(provider, MLXEmbeddingProvider)

    def test_creates_st_provider_on_non_apple(self):
        """Should create SentenceTransformerProvider on non-Apple with auto."""
        settings = Settings(embedding_backend="auto")
        with patch("memory_mcp.embeddings.is_apple_silicon", return_value=False):
            provider = create_provider(settings)
            assert isinstance(provider, SentenceTransformerProvider)

    def test_mlx_model_mapping_applied(self):
        """Should map ST model names to MLX equivalents."""
        settings = Settings(
            embedding_backend="mlx",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        provider = create_provider(settings)
        assert isinstance(provider, MLXEmbeddingProvider)
        assert "mlx-community" in provider._model_name
