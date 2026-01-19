"""Embedding generation with pluggable provider interface.

Supports multiple embedding backends via the EmbeddingProvider protocol.
Currently implemented: SentenceTransformerProvider (default).
"""

import hashlib
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np

from memory_mcp.config import Settings, get_settings
from memory_mcp.logging import get_logger

log = get_logger("embeddings")


# ========== Provider Protocol ==========


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implementations must provide:
    - embed(text) -> np.ndarray: Single text embedding
    - embed_batch(texts) -> list[np.ndarray]: Batch embedding (for efficiency)
    - dimension: int: The embedding dimension
    - name: str: Provider identifier for logging/debugging
    """

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers with common utilities."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings.

        Since embeddings are normalized, dot product = cosine similarity.
        """
        return float(np.dot(embedding1, embedding2))


# ========== Mock Provider (for tests) ==========


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Fast mock embedding provider for testing.

    Generates deterministic embeddings that preserve word-level similarity.
    Each word gets a consistent random vector, and text embeddings are
    the normalized sum of word vectors. This means texts with shared words
    will have higher cosine similarity.

    No model loading, instant results, reproducible across runs.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._word_vectors: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "mock"

    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get or create a consistent random vector for a word."""
        word_lower = word.lower()
        if word_lower not in self._word_vectors:
            # Use word hash to seed random generator for reproducibility
            word_hash = hashlib.md5(word_lower.encode()).digest()
            seed = int.from_bytes(word_hash[:4], "little")
            rng = np.random.Generator(np.random.PCG64(seed))
            self._word_vectors[word_lower] = rng.standard_normal(self._dimension).astype(np.float32)
        return self._word_vectors[word_lower]

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from word vectors.

        Texts with shared words will have higher similarity.
        """
        # Tokenize: split on non-alphanumeric, keep words 2+ chars
        words = [w for w in re.split(r"[^a-zA-Z0-9]+", text) if len(w) >= 2]

        if not words:
            # Fallback for empty/punctuation-only text
            text_hash = hashlib.md5(text.encode()).digest()
            seed = int.from_bytes(text_hash[:4], "little")
            rng = np.random.Generator(np.random.PCG64(seed))
            embedding = rng.standard_normal(self._dimension).astype(np.float32)
        else:
            # Sum word vectors
            embedding = np.zeros(self._dimension, dtype=np.float32)
            for word in words:
                embedding += self._get_word_vector(word)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


# ========== Sentence Transformers Provider ==========


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding provider using sentence-transformers library.

    Default model: all-MiniLM-L6-v2 (384 dimensions, ~90MB).
    """

    def __init__(self, model_name: str, expected_dim: int):
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._model = None

    @property
    def dimension(self) -> int:
        return self._expected_dim

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    def _get_model(self):
        """Lazy-load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model: {}", self._model_name)
            self._model = SentenceTransformer(self._model_name)

            # Verify dimension matches
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self._expected_dim:
                log.warning(
                    "Model dimension {} != expected {}. Update MEMORY_MCP_EMBEDDING_DIM.",
                    actual_dim,
                    self._expected_dim,
                )
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [np.array(e, dtype=np.float32) for e in embeddings]


# ========== Cached Provider Wrapper ==========


class CachedEmbeddingProvider(BaseEmbeddingProvider):
    """Wrapper that adds LRU caching to any embedding provider.

    Caches embeddings by content hash to avoid redundant computation.
    """

    def __init__(self, provider: EmbeddingProvider, cache_size: int = 1000):
        self._provider = provider
        self._cache_size = cache_size
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []  # LRU tracking

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    @property
    def name(self) -> str:
        return f"cached:{self._provider.name}"

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self._cache_size and self._cache_order:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding with caching."""
        key = self._cache_key(text)

        if key in self._cache:
            # Move to end (most recently used)
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]

        # Compute and cache
        embedding = self._provider.embed(text)
        self._evict_if_needed()
        self._cache[key] = embedding
        self._cache_order.append(key)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings with caching for batch."""
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
                # Update LRU
                if key in self._cache_order:
                    self._cache_order.remove(key)
                self._cache_order.append(key)
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Batch compute uncached
        if uncached_texts:
            new_embeddings = self._provider.embed_batch(uncached_texts)
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                key = self._cache_key(text)
                self._evict_if_needed()
                self._cache[key] = embedding
                self._cache_order.append(key)
                results[idx] = embedding

        return results

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "provider": self._provider.name,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_order.clear()


# ========== Provider Factory ==========


def create_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Create an embedding provider based on settings.

    Currently supports:
    - sentence-transformers/* models (default)

    Future: OpenAI, LlamaCpp, etc.
    """
    settings = settings or get_settings()
    model_name = settings.embedding_model
    dimension = settings.embedding_dim

    if model_name.startswith("sentence-transformers/") or "/" in model_name:
        # Sentence transformers model
        return SentenceTransformerProvider(model_name, dimension)

    # Default to sentence transformers
    log.warning("Unknown model format '{}', using sentence-transformers", model_name)
    return SentenceTransformerProvider(model_name, dimension)


# ========== Legacy Compatibility ==========


class EmbeddingEngine:
    """Legacy wrapper for backwards compatibility.

    Wraps an EmbeddingProvider with caching.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        provider = create_provider(self.settings)
        self._provider = CachedEmbeddingProvider(provider, cache_size=1000)
        log.info("EmbeddingEngine initialized with provider: {}", self._provider.name)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._provider.dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return self._provider.embed_batch(texts)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return self._provider.similarity(embedding1, embedding2)

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        return self._provider.cache_stats()

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._provider.clear_cache()


# ========== Utilities ==========


def content_hash(content: str) -> str:
    """Generate SHA256 hash of content for O(1) exact lookup."""
    return hashlib.sha256(content.encode()).hexdigest()


@lru_cache(maxsize=1)
def get_embedding_engine() -> EmbeddingEngine:
    """Get singleton embedding engine."""
    return EmbeddingEngine()
