"""Memory relationships (knowledge graph) mixin for Storage class."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.models import (
    AuditOperation,
    Memory,
    MemoryRelation,
    RelationType,
)

if TYPE_CHECKING:
    pass

log = get_logger("storage.relationships")


class RelationshipsMixin:
    """Mixin providing memory relationship methods for Storage."""

    def link_memories(
        self,
        from_memory_id: int,
        to_memory_id: int,
        relation_type: RelationType,
    ) -> MemoryRelation | None:
        """Create a typed relationship between two memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation_type: Type of relationship

        Returns:
            MemoryRelation if created, None if memories don't exist or already linked.
        """
        if from_memory_id == to_memory_id:
            log.warning("Cannot link memory to itself: id={}", from_memory_id)
            return None

        with self.transaction() as conn:
            # Verify both memories exist
            from_exists = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?", (from_memory_id,)
            ).fetchone()
            to_exists = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?", (to_memory_id,)
            ).fetchone()

            if not from_exists or not to_exists:
                log.warning(
                    "Cannot link: memory {} or {} does not exist",
                    from_memory_id,
                    to_memory_id,
                )
                return None

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO memory_relationships (from_memory_id, to_memory_id, relation_type)
                    VALUES (?, ?, ?)
                    RETURNING id, created_at
                    """,
                    (from_memory_id, to_memory_id, relation_type.value),
                )
                row = cursor.fetchone()
                log.info(
                    "Linked memories: {} -[{}]-> {}",
                    from_memory_id,
                    relation_type.value,
                    to_memory_id,
                )
                return MemoryRelation(
                    id=row[0],
                    from_memory_id=from_memory_id,
                    to_memory_id=to_memory_id,
                    relation_type=relation_type,
                    created_at=datetime.fromisoformat(row[1]),
                )
            except sqlite3.IntegrityError:
                # Already linked with this relation type
                log.debug(
                    "Relationship already exists: {} -[{}]-> {}",
                    from_memory_id,
                    relation_type.value,
                    to_memory_id,
                )
                return None

    def unlink_memories(
        self,
        from_memory_id: int,
        to_memory_id: int,
        relation_type: RelationType | None = None,
    ) -> int:
        """Remove relationship(s) between memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation_type: If specified, only remove this type. Otherwise remove all.

        Returns:
            Number of relationships removed.
        """
        with self.transaction() as conn:
            if relation_type:
                cursor = conn.execute(
                    """
                    DELETE FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ? AND relation_type = ?
                    """,
                    (from_memory_id, to_memory_id, relation_type.value),
                )
            else:
                cursor = conn.execute(
                    """
                    DELETE FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ?
                    """,
                    (from_memory_id, to_memory_id),
                )

            count = cursor.rowcount
            if count > 0:
                self._record_audit(
                    conn,
                    AuditOperation.UNLINK_MEMORIES,
                    target_type="relationship",
                    details=json.dumps(
                        {
                            "from_memory_id": from_memory_id,
                            "to_memory_id": to_memory_id,
                            "relation_type": relation_type.value if relation_type else None,
                            "count_removed": count,
                        }
                    ),
                )
                log.info(
                    "Unlinked {} relationship(s): {} -> {}",
                    count,
                    from_memory_id,
                    to_memory_id,
                )
            return count

    def get_related(
        self,
        memory_id: int,
        relation_type: RelationType | None = None,
        direction: str = "both",
    ) -> list[tuple[Memory, MemoryRelation]]:
        """Get memories related to a given memory.

        Args:
            memory_id: The memory to find relationships for
            relation_type: Filter by relationship type (optional)
            direction: "outgoing" (from this memory), "incoming" (to this memory), or "both"

        Returns:
            List of (related_memory, relationship) tuples.
        """
        results: list[tuple[Memory, MemoryRelation]] = []

        with self._connection() as conn:
            queries: list[tuple[str, tuple[int, ...] | tuple[int, str], str]] = []

            if direction in ("outgoing", "both"):
                # Relationships FROM this memory
                if relation_type:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'outgoing' as direction
                            FROM memory_relationships r
                            WHERE r.from_memory_id = ? AND r.relation_type = ?
                            """,
                            (memory_id, relation_type.value),
                            "to_memory_id",
                        )
                    )
                else:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'outgoing' as direction
                            FROM memory_relationships r
                            WHERE r.from_memory_id = ?
                            """,
                            (memory_id,),
                            "to_memory_id",
                        )
                    )

            if direction in ("incoming", "both"):
                # Relationships TO this memory
                if relation_type:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'incoming' as direction
                            FROM memory_relationships r
                            WHERE r.to_memory_id = ? AND r.relation_type = ?
                            """,
                            (memory_id, relation_type.value),
                            "from_memory_id",
                        )
                    )
                else:
                    queries.append(
                        (
                            """
                            SELECT r.*, 'incoming' as direction
                            FROM memory_relationships r
                            WHERE r.to_memory_id = ?
                            """,
                            (memory_id,),
                            "from_memory_id",
                        )
                    )

            for query, params, related_id_col in queries:
                rows = conn.execute(query, params).fetchall()
                for row in rows:
                    related_id = row[related_id_col]
                    relation = MemoryRelation(
                        id=row["id"],
                        from_memory_id=row["from_memory_id"],
                        to_memory_id=row["to_memory_id"],
                        relation_type=RelationType(row["relation_type"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                    # Fetch the related memory
                    memory = self.get_memory(related_id)
                    if memory:
                        results.append((memory, relation))

        return results

    def get_relationship(
        self,
        from_memory_id: int,
        to_memory_id: int,
        relation_type: RelationType | None = None,
    ) -> list[MemoryRelation]:
        """Get specific relationship(s) between two memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation_type: Filter by type (optional)

        Returns:
            List of matching relationships.
        """
        with self._connection() as conn:
            if relation_type:
                rows = conn.execute(
                    """
                    SELECT * FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ? AND relation_type = ?
                    """,
                    (from_memory_id, to_memory_id, relation_type.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memory_relationships
                    WHERE from_memory_id = ? AND to_memory_id = ?
                    """,
                    (from_memory_id, to_memory_id),
                ).fetchall()

            return [
                MemoryRelation(
                    id=row["id"],
                    from_memory_id=row["from_memory_id"],
                    to_memory_id=row["to_memory_id"],
                    relation_type=RelationType(row["relation_type"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    def get_relationship_stats(self) -> dict:
        """Get statistics about memory relationships."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memory_relationships").fetchone()[0]
            by_type = {
                row["relation_type"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT relation_type, COUNT(*) as count
                    FROM memory_relationships
                    GROUP BY relation_type
                    """
                )
            }
            # Count memories with at least one relationship
            linked_memories = conn.execute(
                """
                SELECT COUNT(DISTINCT memory_id) FROM (
                    SELECT from_memory_id as memory_id FROM memory_relationships
                    UNION
                    SELECT to_memory_id as memory_id FROM memory_relationships
                )
                """
            ).fetchone()[0]

            return {
                "total_relationships": total,
                "by_type": by_type,
                "linked_memories": linked_memories,
            }
