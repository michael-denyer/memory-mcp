"""Embedding generation using sentence-transformers."""

import hashlib
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from memory_mcp.config import Settings, get_settings


class EmbeddingEngine:
    """Manages embedding model and generation."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings.

        Since embeddings are normalized, dot product = cosine similarity.
        """
        return float(np.dot(embedding1, embedding2))


def content_hash(content: str) -> str:
    """Generate SHA256 hash of content for O(1) exact lookup."""
    return hashlib.sha256(content.encode()).hexdigest()


@lru_cache(maxsize=1)
def get_embedding_engine() -> EmbeddingEngine:
    """Get singleton embedding engine."""
    return EmbeddingEngine()
