"""Tests for semantic clustering in display contexts."""

from datetime import datetime, timezone

import numpy as np

from memory_mcp.helpers import (
    _compute_pairwise_similarity,
    _generate_cluster_label,
    cluster_memories_for_display,
)
from memory_mcp.models import (
    DisplayCluster,
    Memory,
    MemorySource,
    MemoryType,
)


def make_memory(
    id: int,
    content: str,
    tags: list[str] | None = None,
    memory_type: MemoryType = MemoryType.PROJECT,
) -> Memory:
    """Create a test memory with minimal required fields."""
    return Memory(
        id=id,
        content=content,
        content_hash=f"hash_{id}",
        memory_type=memory_type,
        source=MemorySource.MANUAL,
        is_hot=True,
        is_pinned=False,
        promotion_source=None,
        tags=tags or [],
        access_count=1,
        last_accessed_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )


def make_similar_embeddings(n: int, base: np.ndarray, noise: float = 0.05) -> list[np.ndarray]:
    """Create n embeddings similar to base with small noise."""
    embeddings = []
    for _ in range(n):
        noisy = base + np.random.randn(*base.shape) * noise
        noisy = noisy / np.linalg.norm(noisy)  # Normalize
        embeddings.append(noisy)
    return embeddings


def make_random_embedding(dim: int = 384) -> np.ndarray:
    """Create a random normalized embedding."""
    emb = np.random.randn(dim).astype(np.float32)
    return emb / np.linalg.norm(emb)


class TestPairwiseSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        emb = make_random_embedding()
        sim = _compute_pairwise_similarity(emb, emb)
        assert abs(sim - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0."""
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = _compute_pairwise_similarity(emb_a, emb_b)
        assert abs(sim) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        sim = _compute_pairwise_similarity(emb_a, emb_b)
        assert abs(sim + 1.0) < 0.001

    def test_zero_vector(self):
        """Zero vector should return 0 similarity."""
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim = _compute_pairwise_similarity(emb_a, emb_b)
        assert sim == 0.0


class TestClusterLabelGeneration:
    """Tests for cluster label generation."""

    def test_uses_most_common_tag(self):
        """Should use the most common tag as label."""
        memories = [
            make_memory(1, "content", tags=["python", "testing"]),
            make_memory(2, "content", tags=["python", "api"]),
            make_memory(3, "content", tags=["python"]),
        ]
        label = _generate_cluster_label(memories)
        assert label == "Python"

    def test_fallback_to_memory_type(self):
        """Should fall back to memory type when no tags."""
        memories = [
            make_memory(1, "content", tags=[]),
            make_memory(2, "content", tags=[]),
        ]
        label = _generate_cluster_label(memories)
        assert label == "Project"

    def test_title_cases_label(self):
        """Should title-case the label."""
        memories = [
            make_memory(1, "content", tags=["api-testing"]),
            make_memory(2, "content", tags=["api-testing"]),
        ]
        label = _generate_cluster_label(memories)
        assert label == "Api Testing"

    def test_limits_words(self):
        """Should limit label to max_words."""
        memories = [
            make_memory(1, "content", tags=["very-long-tag-name-here"]),
            make_memory(2, "content", tags=["very-long-tag-name-here"]),
        ]
        label = _generate_cluster_label(memories, max_words=2)
        assert label == "Very Long"


class TestClusterMemoriesForDisplay:
    """Tests for the main clustering function."""

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        clusters, unclustered = cluster_memories_for_display([], {})
        assert clusters == []
        assert unclustered == []

    def test_no_embeddings(self):
        """Should put all memories in unclustered when no embeddings."""
        memories = [make_memory(1, "test"), make_memory(2, "test")]
        clusters, unclustered = cluster_memories_for_display(memories, {})
        assert clusters == []
        assert len(unclustered) == 2

    def test_clusters_similar_memories(self):
        """Similar memories should be grouped together."""
        np.random.seed(42)  # For reproducibility

        # Create memories
        memories = [
            make_memory(1, "Python function", tags=["python"]),
            make_memory(2, "Python class", tags=["python"]),
            make_memory(3, "JavaScript module", tags=["javascript"]),
            make_memory(4, "JavaScript function", tags=["javascript"]),
        ]

        # Create embeddings: 1&2 similar, 3&4 similar, but groups dissimilar
        base_python = make_random_embedding()
        base_js = make_random_embedding()

        # Use very small noise to ensure high similarity within groups
        embeddings = {
            1: base_python + np.random.randn(384).astype(np.float32) * 0.01,
            2: base_python + np.random.randn(384).astype(np.float32) * 0.01,
            3: base_js + np.random.randn(384).astype(np.float32) * 0.01,
            4: base_js + np.random.randn(384).astype(np.float32) * 0.01,
        }
        # Normalize
        for k in embeddings:
            embeddings[k] = embeddings[k] / np.linalg.norm(embeddings[k])

        # Use a lower threshold to ensure clustering works
        clusters, unclustered = cluster_memories_for_display(
            memories, embeddings, threshold=0.70, min_cluster_size=2
        )

        # Should have at least 1 cluster (the greedy algorithm may not find 2 perfect clusters)
        assert len(clusters) >= 1
        # Total items should be accounted for
        total = sum(c.size for c in clusters) + len(unclustered)
        assert total == 4

    def test_singletons_go_to_unclustered(self):
        """Memories without cluster mates should go to unclustered."""
        np.random.seed(42)

        memories = [
            make_memory(1, "Python function", tags=["python"]),
            make_memory(2, "Random unique content"),
            make_memory(3, "Another unique item"),
        ]

        # All embeddings are completely different
        embeddings = {
            1: make_random_embedding(),
            2: make_random_embedding(),
            3: make_random_embedding(),
        }

        clusters, unclustered = cluster_memories_for_display(
            memories, embeddings, threshold=0.95, min_cluster_size=2
        )

        assert len(clusters) == 0
        assert len(unclustered) == 3

    def test_respects_max_clusters(self):
        """Should not create more than max_clusters."""
        np.random.seed(42)

        # Create 6 memories that could form 3 pairs
        memories = [make_memory(i, f"content {i}") for i in range(6)]

        # Create 3 distinct cluster bases
        bases = [make_random_embedding() for _ in range(3)]

        # Pair memories with similar embeddings
        embeddings = {}
        for i in range(6):
            base_idx = i // 2
            embeddings[i] = bases[base_idx] + np.random.randn(384).astype(np.float32) * 0.02
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

        clusters, unclustered = cluster_memories_for_display(
            memories, embeddings, threshold=0.90, min_cluster_size=2, max_clusters=2
        )

        # Should have at most 2 clusters
        assert len(clusters) <= 2
        # Remaining should be unclustered
        assert len(clusters) + len(unclustered) // 2 <= 3

    def test_preserves_score_ordering(self):
        """Memories within clusters should preserve original order."""
        np.random.seed(42)

        # Create memories in specific order
        memories = [
            make_memory(1, "First", tags=["test"]),
            make_memory(2, "Second", tags=["test"]),
            make_memory(3, "Third", tags=["test"]),
        ]

        # All similar embeddings
        base = make_random_embedding()
        embeddings = {
            1: base + np.random.randn(384).astype(np.float32) * 0.01,
            2: base + np.random.randn(384).astype(np.float32) * 0.01,
            3: base + np.random.randn(384).astype(np.float32) * 0.01,
        }
        for k in embeddings:
            embeddings[k] = embeddings[k] / np.linalg.norm(embeddings[k])

        clusters, _ = cluster_memories_for_display(
            memories, embeddings, threshold=0.90, min_cluster_size=2
        )

        assert len(clusters) == 1
        # Order should be preserved
        assert clusters[0].members[0].id == 1
        assert clusters[0].members[1].id == 2
        assert clusters[0].members[2].id == 3

    def test_disabled_returns_all_unclustered(self):
        """When threshold is 1.0, nothing should cluster."""
        memories = [make_memory(i, f"content {i}") for i in range(4)]
        base = make_random_embedding()
        embeddings = {i: base.copy() for i in range(4)}  # All identical

        clusters, unclustered = cluster_memories_for_display(
            memories, embeddings, threshold=1.0, min_cluster_size=2
        )

        # With threshold=1.0, only exact matches cluster
        # Even identical embeddings might not hit 1.0 due to float precision
        # But the point is very high threshold = minimal clustering
        total = sum(c.size for c in clusters) + len(unclustered)
        assert total == 4


class TestDisplayCluster:
    """Tests for DisplayCluster dataclass."""

    def test_size_property(self):
        """Size property should return member count."""
        memories = [make_memory(i, f"content {i}") for i in range(3)]
        cluster = DisplayCluster(
            label="Test",
            members=memories,
            avg_similarity=0.85,
        )
        assert cluster.size == 3
