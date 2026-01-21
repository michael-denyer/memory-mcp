"""Memory consolidation (MemoryBank-inspired) mixin for Storage class."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from memory_mcp.logging import get_logger
from memory_mcp.models import AuditOperation, ConsolidationCluster, MemoryType

if TYPE_CHECKING:
    pass

log = get_logger("storage.consolidation")


class ConsolidationMixin:
    """Mixin providing memory consolidation methods for Storage."""

    def find_consolidation_clusters(
        self,
        memory_type: MemoryType | None = None,
        threshold: float | None = None,
        min_cluster_size: int | None = None,
    ) -> list[ConsolidationCluster]:
        """Find clusters of similar memories that could be consolidated.

        Uses vector similarity to group near-duplicates.

        Args:
            memory_type: Optional filter by memory type
            threshold: Similarity threshold (default from settings)
            min_cluster_size: Minimum memories to form cluster (default from settings)

        Returns:
            List of ConsolidationCluster objects
        """
        threshold = threshold or self.settings.consolidation_threshold
        min_size = min_cluster_size or self.settings.consolidation_min_cluster_size

        # Get all memories (optionally filtered)
        with self._connection() as conn:
            if memory_type:
                rows = conn.execute(
                    """
                    SELECT id, content, access_count, is_hot, is_pinned
                    FROM memories
                    WHERE memory_type = ?
                    ORDER BY access_count DESC
                    """,
                    (memory_type.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, content, access_count, is_hot, is_pinned
                    FROM memories
                    ORDER BY access_count DESC
                    """
                ).fetchall()

        if len(rows) < min_size:
            return []

        # Build similarity matrix using embeddings
        memory_ids = [r["id"] for r in rows]
        contents = [r["content"] for r in rows]
        access_counts = {r["id"]: r["access_count"] for r in rows}
        is_hot = {r["id"]: r["is_hot"] for r in rows}
        is_pinned = {r["id"]: r["is_pinned"] for r in rows}

        # Get embeddings for all memories
        embeddings = []
        for content in contents:
            emb = self._embedding_engine.embed(content)
            embeddings.append(emb)

        embeddings_array = np.array(embeddings)

        # Find clusters using greedy approach
        clusters: list[ConsolidationCluster] = []
        assigned: set[int] = set()

        for i, mem_id in enumerate(memory_ids):
            if mem_id in assigned:
                continue

            # Skip pinned memories as cluster seeds
            if is_pinned.get(mem_id):
                continue

            # Find similar memories
            cluster_members = [mem_id]
            similarities = []

            for j, other_id in enumerate(memory_ids):
                if i == j or other_id in assigned:
                    continue

                # Compute cosine similarity
                norm_i = np.linalg.norm(embeddings_array[i])
                norm_j = np.linalg.norm(embeddings_array[j])
                if norm_i > 0 and norm_j > 0:
                    sim = float(
                        np.dot(embeddings_array[i], embeddings_array[j]) / (norm_i * norm_j)
                    )
                else:
                    sim = 0.0

                if sim >= threshold:
                    cluster_members.append(other_id)
                    similarities.append(sim)

            if len(cluster_members) >= min_size:
                # Get tags for all members
                all_tags: set[str] = set()
                for mid in cluster_members:
                    memory = self.get_memory(mid)
                    if memory:
                        all_tags.update(memory.tags)

                # Choose representative: prefer hot, then highest access count
                representative = max(
                    cluster_members,
                    key=lambda mid: (is_hot.get(mid, 0), access_counts.get(mid, 0)),
                )

                avg_sim = sum(similarities) / len(similarities) if similarities else 1.0
                clusters.append(
                    ConsolidationCluster(
                        representative_id=representative,
                        member_ids=cluster_members,
                        avg_similarity=avg_sim,
                        total_access_count=sum(access_counts.get(m, 0) for m in cluster_members),
                        combined_tags=sorted(all_tags),
                    )
                )

                assigned.update(cluster_members)

        log.info(
            "Found {} consolidation clusters from {} memories",
            len(clusters),
            len(memory_ids),
        )
        return clusters

    def consolidate_cluster(
        self,
        cluster: ConsolidationCluster,
    ) -> dict:
        """Consolidate a cluster by merging members into representative.

        Args:
            cluster: The cluster to consolidate

        Returns:
            Dict with consolidation results
        """
        if len(cluster.member_ids) < 2:
            return {"success": False, "error": "Cluster too small"}

        # Get representative memory
        representative = self.get_memory(cluster.representative_id)
        if not representative:
            return {"success": False, "error": "Representative memory not found"}

        deleted_ids = []
        new_tags: set[str] = set()

        with self.transaction() as conn:
            # Update representative with combined tags
            existing_tags = set(representative.tags)
            new_tags = set(cluster.combined_tags) - existing_tags

            for tag in new_tags:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (cluster.representative_id, tag),
                )

            # Update access count to reflect combined usage
            conn.execute(
                """
                UPDATE memories
                SET access_count = ?,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (cluster.total_access_count, cluster.representative_id),
            )

            # Delete non-representative members
            for member_id in cluster.member_ids:
                if member_id != cluster.representative_id:
                    # Delete vectors
                    conn.execute(
                        "DELETE FROM memory_vectors WHERE rowid = ?",
                        (member_id,),
                    )
                    # Delete memory
                    conn.execute(
                        "DELETE FROM memories WHERE id = ?",
                        (member_id,),
                    )
                    deleted_ids.append(member_id)

            # Record in audit log
            self._record_audit(
                conn,
                AuditOperation.CLEANUP_MEMORIES,
                target_type="consolidation",
                target_id=cluster.representative_id,
                details=json.dumps(
                    {
                        "merged_count": len(deleted_ids),
                        "deleted_ids": deleted_ids,
                        "avg_similarity": cluster.avg_similarity,
                    }
                ),
            )

        log.info(
            "Consolidated cluster: kept id={}, deleted {} members",
            cluster.representative_id,
            len(deleted_ids),
        )

        return {
            "success": True,
            "representative_id": cluster.representative_id,
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "tags_added": list(new_tags),
        }

    def preview_consolidation(
        self,
        memory_type: MemoryType | None = None,
    ) -> dict:
        """Preview what consolidation would do without making changes.

        Args:
            memory_type: Optional filter by memory type

        Returns:
            Preview of consolidation results
        """
        clusters = self.find_consolidation_clusters(memory_type=memory_type)

        total_memories = sum(len(c.member_ids) for c in clusters)
        memories_to_delete = sum(len(c.member_ids) - 1 for c in clusters)

        pct = memories_to_delete / total_memories * 100 if total_memories > 0 else 0

        return {
            "cluster_count": len(clusters),
            "total_memories_in_clusters": total_memories,
            "memories_to_delete": memories_to_delete,
            "space_savings_pct": round(pct, 1),
            "clusters": [
                {
                    "representative_id": c.representative_id,
                    "member_count": len(c.member_ids),
                    "avg_similarity": round(c.avg_similarity, 3),
                    "total_access_count": c.total_access_count,
                }
                for c in clusters
            ],
        }

    def run_consolidation(
        self,
        memory_type: MemoryType | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Run consolidation on all eligible clusters.

        Args:
            memory_type: Optional filter by memory type
            dry_run: If True, only preview without making changes

        Returns:
            Consolidation results
        """
        if dry_run:
            return self.preview_consolidation(memory_type=memory_type)

        clusters = self.find_consolidation_clusters(memory_type=memory_type)

        results: dict = {
            "clusters_processed": 0,
            "memories_deleted": 0,
            "errors": [],
        }

        for cluster in clusters:
            result = self.consolidate_cluster(cluster)
            if result.get("success"):
                results["clusters_processed"] += 1
                results["memories_deleted"] += result.get("deleted_count", 0)
            else:
                results["errors"].append(result.get("error"))

        log.info(
            "Consolidation complete: {} clusters, {} memories deleted",
            results["clusters_processed"],
            results["memories_deleted"],
        )

        return results
