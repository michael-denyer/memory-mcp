"""Contradiction detection mixin for Storage class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.models import (
    Memory,
    MemoryRelation,
    PotentialContradiction,
    RelationType,
    TrustReason,
)

if TYPE_CHECKING:
    pass

log = get_logger("storage.contradictions")


class ContradictionsMixin:
    """Mixin providing contradiction detection methods for Storage."""

    def find_contradictions(
        self,
        memory_id: int,
        similarity_threshold: float = 0.75,
        limit: int = 5,
    ) -> list[PotentialContradiction]:
        """Find memories that may contradict a given memory.

        Looks for memories that are semantically similar (same topic)
        but might contain conflicting information.

        Args:
            memory_id: The memory to check for contradictions
            similarity_threshold: Minimum similarity to consider (default 0.75)
            limit: Maximum contradictions to return

        Returns:
            List of potential contradictions sorted by similarity.
        """
        with self._connection() as conn:
            # Get the target memory and its embedding
            memory_row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not memory_row:
                return []

            embedding_row = conn.execute(
                "SELECT embedding FROM memory_vectors WHERE rowid = ?", (memory_id,)
            ).fetchone()
            if not embedding_row:
                return []

            # Find similar memories (excluding self)
            rows = conn.execute(
                """
                SELECT
                    m.id,
                    m.content,
                    m.content_hash,
                    m.memory_type,
                    m.source,
                    m.is_hot,
                    m.access_count,
                    m.last_accessed_at,
                    m.created_at,
                    m.trust_score,
                    vec_distance_cosine(v.embedding, ?) as distance
                FROM memory_vectors v
                JOIN memories m ON m.id = v.rowid
                WHERE m.id != ?
                ORDER BY distance ASC
                LIMIT ?
                """,
                (embedding_row["embedding"], memory_id, limit * 2),
            ).fetchall()

            # Check which are already linked as contradictions
            existing_contradictions = set()
            contradicts_type = RelationType.CONTRADICTS.value
            for row in conn.execute(
                """
                SELECT to_memory_id FROM memory_relationships
                WHERE from_memory_id = ? AND relation_type = ?
                UNION
                SELECT from_memory_id FROM memory_relationships
                WHERE to_memory_id = ? AND relation_type = ?
                """,
                (memory_id, contradicts_type, memory_id, contradicts_type),
            ):
                existing_contradictions.add(row[0])

            source_memory = self._row_to_memory(memory_row, conn)
            results: list[PotentialContradiction] = []

            for row in rows:
                similarity = 1 - row["distance"]
                if similarity < similarity_threshold:
                    continue

                other_memory = self._row_to_memory(row, conn)
                results.append(
                    PotentialContradiction(
                        memory_a=source_memory,
                        memory_b=other_memory,
                        similarity=similarity,
                        already_linked=other_memory.id in existing_contradictions,
                    )
                )

                if len(results) >= limit:
                    break

            return results

    def get_all_contradictions(self) -> list[tuple[Memory, Memory, MemoryRelation]]:
        """Get all memory pairs marked as contradictions.

        Returns:
            List of (memory_a, memory_b, relationship) tuples.
        """
        from datetime import datetime

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, from_memory_id, to_memory_id, relation_type, created_at
                FROM memory_relationships
                WHERE relation_type = ?
                ORDER BY created_at DESC
                """,
                (RelationType.CONTRADICTS.value,),
            ).fetchall()

            results: list[tuple[Memory, Memory, MemoryRelation]] = []
            for row in rows:
                relation = MemoryRelation(
                    id=row["id"],
                    from_memory_id=row["from_memory_id"],
                    to_memory_id=row["to_memory_id"],
                    relation_type=RelationType(row["relation_type"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )

                m1 = self.get_memory(row["from_memory_id"])
                m2 = self.get_memory(row["to_memory_id"])

                if m1 and m2:
                    results.append((m1, m2, relation))

            return results

    def mark_contradiction(
        self,
        memory_id_a: int,
        memory_id_b: int,
    ) -> MemoryRelation | None:
        """Mark two memories as contradicting each other.

        Creates a CONTRADICTS relationship between the memories.

        Args:
            memory_id_a: First memory ID
            memory_id_b: Second memory ID

        Returns:
            The created relationship, or None if already exists or memories don't exist.
        """
        return self.link_memories(memory_id_a, memory_id_b, RelationType.CONTRADICTS)

    def resolve_contradiction(
        self,
        memory_id_a: int,
        memory_id_b: int,
        keep_id: int,
        resolution: str = "supersedes",
    ) -> bool:
        """Resolve a contradiction by keeping one memory and handling the other.

        Args:
            memory_id_a: First memory in contradiction
            memory_id_b: Second memory in contradiction
            keep_id: Which memory to keep (must be one of the two)
            resolution: How to handle the discarded memory:
                - "supersedes": Keep memory supersedes the other (default)
                - "delete": Delete the other memory
                - "weaken": Weaken trust in the other memory

        Returns:
            True if resolution succeeded.
        """
        if keep_id not in (memory_id_a, memory_id_b):
            log.warning("keep_id must be one of the contradicting memories")
            return False

        discard_id = memory_id_b if keep_id == memory_id_a else memory_id_a

        # Remove the contradiction relationship
        self.unlink_memories(memory_id_a, memory_id_b, RelationType.CONTRADICTS)
        self.unlink_memories(memory_id_b, memory_id_a, RelationType.CONTRADICTS)

        if resolution == "delete":
            return self.delete_memory(discard_id)
        elif resolution == "supersedes":
            self.link_memories(keep_id, discard_id, RelationType.SUPERSEDES)
            # Weaken trust in superseded memory
            self.weaken_trust(discard_id, 0.2, TrustReason.CONTRADICTION_RESOLVED)
            return True
        elif resolution == "weaken":
            self.weaken_trust(discard_id, 0.3, TrustReason.CONTRADICTION_RESOLVED)
            return True
        else:
            log.warning("Unknown resolution type: {}", resolution)
            return False
