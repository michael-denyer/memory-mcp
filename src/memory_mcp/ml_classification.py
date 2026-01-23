"""ML-enhanced memory classification using embeddings.

Uses few-shot learning with embedding prototypes instead of rigid regex patterns.
Categories are defined by example phrases that capture the semantic meaning.

IMPORTANT: Classification confidence is NOT a ranking signal.
-------------------------------------------------------------
The similarity score from classification reflects "fit to category prototype",
not memory utility or quality. High confidence means the content clearly matches
a category's semantic space, but says nothing about whether the memory is:
- Accurate or trustworthy (use trust_score for that)
- Useful to recall (use helpfulness/utility_score for that)
- Recently validated (use last_accessed_at for that)

Appropriate uses for ML confidence:
- Category assignment (threshold gate): OK
- Category-specific promotion gates (e.g., lower threshold for landmines): OK
- Tie-breaking within same category: Weak signal, OK

Inappropriate uses:
- General recall ranking: WRONG (use composite score instead)
- Hot cache eviction priority: WRONG (use hot_score instead)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from memory_mcp.logging import get_logger

if TYPE_CHECKING:
    from memory_mcp.embeddings import EmbeddingProvider

log = get_logger("ml_classification")


# Category prototypes: example phrases that define each category's semantic space
# These are embedded once and used for few-shot classification
CATEGORY_PROTOTYPES: dict[str, list[str]] = {
    "antipattern": [
        "Don't use this approach, it causes problems",
        "Never do this, it's an anti-pattern",
        "Avoid using this method because it's deprecated",
        "Bad practice: this will break in production",
        "Wrong way to handle this - use X instead",
        "This is a code smell, refactor to use Y",
    ],
    "landmine": [
        "Watch out - this can silently fail",
        "Be careful, there's a hidden gotcha here",
        "Warning: this breaks if you forget to check X",
        "Easy to miss but critical: always do Y first",
        "Subtle bug: this looks correct but fails on edge cases",
        "Pitfall: the API doesn't throw on invalid input",
    ],
    "decision": [
        "We chose X over Y because of performance",
        "Decision: use React instead of Vue for this project",
        "Rationale: picked PostgreSQL for its JSON support",
        "Trade-off: faster builds vs smaller bundle size",
        "Why we went with serverless architecture",
    ],
    "convention": [
        "Always name files in kebab-case",
        "The convention is to use TypeScript for all new code",
        "Standard: all API responses return JSON",
        "Rule: tests must be colocated with source files",
        "We always run lint before commit",
    ],
    "preference": [
        "I prefer using async/await over promises",
        "Recommend using pnpm instead of npm",
        "Best practice: keep functions under 20 lines",
        "Better to use composition over inheritance",
        "My style is to use early returns",
    ],
    "lesson": [
        "Learned that caching needs invalidation strategy",
        "Realized the bottleneck was database queries",
        "Discovered that the API rate limits at 100 req/s",
        "Turns out the library doesn't support ESM",
        "In hindsight, should have used a queue",
    ],
    "constraint": [
        "Must use Node 18 or higher for this feature",
        "Cannot deploy on weekends due to policy",
        "Blocked by waiting for API access approval",
        "Requires authentication for all endpoints",
        "Only works with the premium tier API",
    ],
    "architecture": [
        "The system uses a microservices architecture",
        "Data flows from API gateway through the message queue",
        "Components are organized by feature not layer",
        "The auth service handles all authentication",
        "Schema uses normalized tables with foreign keys",
    ],
    "context": [
        "For background: we migrated from MongoDB last year",
        "The current state is that deployment is manual",
        "Previously this was handled by a cron job",
        "The reason we need this is customer feedback",
    ],
    "bug": [
        "Fixed: null pointer when user has no profile",
        "Bug: the form doesn't validate email format",
        "Workaround: restart the service after config change",
        "Issue: memory leak in the websocket handler",
    ],
    "todo": [
        "TODO: add rate limiting to this endpoint",
        "Need to implement caching for this query",
        "Should add tests for the edge cases",
        "FIXME: this doesn't handle Unicode properly",
    ],
}


class EmbeddingClassifier:
    """Few-shot classifier using embedding similarity to prototypes.

    Instead of rigid regex patterns, computes cosine similarity between
    the memory's embedding and pre-computed category prototype embeddings.

    Benefits over regex:
    - Captures semantic meaning ("avoid" ≈ "don't use" ≈ "bad practice")
    - Handles paraphrasing and synonyms
    - More robust to natural language variation
    - Can use the same embedding already computed for semantic search
    """

    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider
        self._prototype_embeddings: dict[str, np.ndarray] | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize prototype embeddings."""
        if self._initialized:
            return

        log.info("Initializing category prototype embeddings...")
        self._prototype_embeddings = {}

        for category, prototypes in CATEGORY_PROTOTYPES.items():
            # Embed all prototypes and average them for the category centroid
            embeddings = self._provider.embed_batch(prototypes)
            centroid = np.mean(embeddings, axis=0)
            # Normalize the centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self._prototype_embeddings[category] = centroid

        self._initialized = True
        log.info("Initialized {} category prototypes", len(self._prototype_embeddings))

    @overload
    def classify(
        self,
        embedding: np.ndarray,
        threshold: float = ...,
        return_scores: Literal[False] = ...,
    ) -> str | None: ...

    @overload
    def classify(
        self,
        embedding: np.ndarray,
        threshold: float = ...,
        return_scores: Literal[True] = ...,
    ) -> tuple[str | None, dict[str, float]]: ...

    def classify(
        self,
        embedding: np.ndarray,
        threshold: float = 0.35,
        return_scores: bool = False,
    ) -> str | tuple[str | None, dict[str, float]] | None:
        """Classify a memory embedding into a category.

        Args:
            embedding: Pre-computed embedding vector (normalized)
            threshold: Minimum similarity to assign a category
            return_scores: If True, return (category, all_scores) tuple

        Returns:
            Category string, or None if below threshold.
            If return_scores=True, returns (category, scores_dict).
        """
        self._ensure_initialized()

        scores: dict[str, float] = {}
        for category, prototype in self._prototype_embeddings.items():
            # Cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(embedding, prototype))
            scores[category] = similarity

        # Find best match
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        result = best_category if best_score >= threshold else None

        if return_scores:
            return result, scores
        return result

    def classify_text(
        self,
        text: str,
        threshold: float = 0.35,
    ) -> str | None:
        """Convenience method to classify raw text.

        Args:
            text: Memory content to classify
            threshold: Minimum similarity to assign a category

        Returns:
            Category string, or None if below threshold.
        """
        embedding = self._provider.embed(text)
        # When return_scores=False, classify returns str | None directly
        return self.classify(embedding, threshold, return_scores=False)

    def get_category_scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Get similarity scores for all categories.

        Useful for debugging or showing category confidence.
        """
        _, scores = self.classify(embedding, threshold=0.0, return_scores=True)
        return scores


# Singleton classifier instance (lazy-loaded)
_classifier: EmbeddingClassifier | None = None


def get_classifier(provider: EmbeddingProvider) -> EmbeddingClassifier:
    """Get or create the singleton classifier."""
    global _classifier
    if _classifier is None:
        _classifier = EmbeddingClassifier(provider)
    return _classifier


def ml_classify_category(
    content: str,
    embedding: np.ndarray | None = None,
    provider: EmbeddingProvider | None = None,
    threshold: float = 0.35,
) -> str | None:
    """ML-based category classification (drop-in replacement for infer_category).

    Uses embedding similarity to category prototypes instead of regex patterns.

    Args:
        content: Memory content (used if embedding not provided)
        embedding: Pre-computed embedding (preferred, avoids recomputation)
        provider: Embedding provider (required if embedding not provided)
        threshold: Minimum similarity to assign category (0.35 = moderate confidence)

    Returns:
        Category string or None.
    """
    if embedding is None:
        if provider is None:
            # Fall back to rule-based if no embedding infrastructure
            from memory_mcp.helpers import infer_category

            return infer_category(content)
        embedding = provider.embed(content)

    classifier = get_classifier(provider)
    return classifier.classify(embedding, threshold, return_scores=False)


def hybrid_classify_category(
    content: str,
    embedding: np.ndarray | None = None,
    provider: EmbeddingProvider | None = None,
    ml_threshold: float = 0.40,
) -> str | None:
    """Hybrid classification: ML with regex fallback.

    Tries ML classification first (more nuanced), falls back to regex
    if ML confidence is low. This provides the best of both worlds:
    - ML captures semantic meaning for natural language
    - Regex catches explicit patterns like "TODO:" or "FIXME"

    Args:
        content: Memory content
        embedding: Pre-computed embedding (optional)
        provider: Embedding provider (optional)
        ml_threshold: ML confidence threshold before falling back to regex

    Returns:
        Category string or None.
    """
    from memory_mcp.helpers import infer_category

    # Try ML first if we have embedding infrastructure
    if embedding is not None or provider is not None:
        if embedding is None:
            embedding = provider.embed(content)

        classifier = get_classifier(provider)
        ml_category, scores = classifier.classify(
            embedding, threshold=ml_threshold, return_scores=True
        )

        if ml_category is not None:
            log.debug(
                "ML classified as '{}' (score={:.3f})",
                ml_category,
                scores[ml_category],
            )
            return ml_category

    # Fall back to regex for explicit patterns
    regex_category = infer_category(content)
    if regex_category:
        log.debug("Regex fallback classified as '{}'", regex_category)
    return regex_category
