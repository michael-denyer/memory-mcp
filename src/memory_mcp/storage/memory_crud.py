"""Memory CRUD operations mixin for Storage class."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from memory_mcp.embeddings import content_hash
from memory_mcp.logging import get_logger
from memory_mcp.models import (
    AuditOperation,
    Memory,
    MemorySource,
    MemoryType,
    PromotionSource,
)

if TYPE_CHECKING:
    pass

log = get_logger("storage.memory_crud")


class ValidationError(ValueError):
    """Raised when input validation fails."""


class MemoryCrudMixin:
    """Mixin providing memory CRUD methods for Storage."""

    def _validate_content(self, content: str, field_name: str = "content") -> None:
        """Validate content length against settings.

        Raises:
            ValidationError: If content is empty or exceeds max length.
        """
        if not content or not content.strip():
            raise ValidationError(f"{field_name} cannot be empty")
        if len(content) > self.settings.max_content_length:
            raise ValidationError(
                f"{field_name} too long: {len(content)} chars "
                f"(max: {self.settings.max_content_length})"
            )

    def _validate_tags(self, tags: list[str] | None) -> list[str]:
        """Validate and normalize tags.

        Returns:
            Normalized tag list (stripped, non-empty).

        Raises:
            ValidationError: If too many tags provided.
        """
        if not tags:
            return []
        # Strip whitespace and filter empty
        normalized = [t.strip() for t in tags if t and t.strip()]
        if len(normalized) > self.settings.max_tags:
            raise ValidationError(
                f"Too many tags: {len(normalized)} (max: {self.settings.max_tags})"
            )
        return normalized

    def _find_semantic_duplicate(
        self,
        conn: sqlite3.Connection,
        embedding: np.ndarray,
        threshold: float,
        project_id: str | None = None,
    ) -> tuple[int, float] | None:
        """Find the most similar existing memory above threshold.

        Args:
            conn: Database connection
            embedding: Embedding of new content
            threshold: Minimum similarity to consider a duplicate
            project_id: Project ID for project-scoped dedup (None = global)

        Returns:
            Tuple of (memory_id, similarity) if found, None otherwise.
        """
        # Project-scoped dedup when project awareness is enabled
        if self.settings.project_awareness_enabled and project_id:
            row = conn.execute(
                """
                SELECT m.id, vec_distance_cosine(v.embedding, ?) as distance
                FROM memory_vectors v
                JOIN memories m ON m.id = v.rowid
                WHERE m.project_id = ?
                ORDER BY distance ASC
                LIMIT 1
                """,
                (embedding.tobytes(), project_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT m.id, vec_distance_cosine(v.embedding, ?) as distance
                FROM memory_vectors v
                JOIN memories m ON m.id = v.rowid
                ORDER BY distance ASC
                LIMIT 1
                """,
                (embedding.tobytes(),),
            ).fetchone()

        if row is None:
            return None

        similarity = 1 - row["distance"]
        if similarity >= threshold:
            return row["id"], similarity
        return None

    def _get_memory_by_id(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
    ) -> Memory | None:
        """Get a memory by ID using an existing connection."""
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row, conn)

    def _merge_with_existing(
        self,
        conn: sqlite3.Connection,
        new_content: str,
        existing: Memory,
        new_tags: list[str],
        similarity: float,
    ) -> tuple[int, bool]:
        """Merge new content with an existing similar memory.

        Strategy:
        - If new content is longer/richer: update content and embedding
        - Always merge tags
        - Increment access count

        Returns:
            Tuple of (memory_id, False) - False because not a new memory.
        """
        new_is_longer = len(new_content) > len(existing.content)

        if new_is_longer:
            # New content is richer - update content and embedding
            new_embedding = self._embedding_engine.embed(new_content)
            new_hash = content_hash(new_content)
            conn.execute(
                """
                UPDATE memories
                SET content = ?,
                    content_hash = ?,
                    access_count = access_count + 1,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_content, new_hash, existing.id),
            )
            conn.execute(
                "UPDATE memory_vectors SET embedding = ? WHERE rowid = ?",
                (new_embedding.tobytes(), existing.id),
            )
            log.info(
                "Merged memory #{}: updated content (similarity={:.2f}, {} -> {} chars)",
                existing.id,
                similarity,
                len(existing.content),
                len(new_content),
            )
        else:
            # Existing content is richer - just increment access
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (existing.id,),
            )
            log.info(
                "Merged memory #{}: kept existing content (similarity={:.2f})",
                existing.id,
                similarity,
            )

        # Merge tags
        if new_tags:
            conn.executemany(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                [(existing.id, tag) for tag in new_tags],
            )

        return existing.id, False

    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source: MemorySource = MemorySource.MANUAL,
        tags: list[str] | None = None,
        is_hot: bool = False,
        source_log_id: int | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[int, bool]:
        """Store a new memory with embedding.

        Args:
            content: The memory content
            memory_type: Type of memory (project, pattern, etc.)
            source: How memory was created (manual, mined)
            tags: Optional tags for categorization
            is_hot: Whether to add to hot cache immediately
            source_log_id: For mined memories, the originating output_log ID
            session_id: Conversation session ID for provenance tracking
            project_id: Project ID for project-aware filtering (e.g., "github/owner/repo")

        Returns:
            Tuple of (memory_id, is_new) where is_new indicates if a new
            memory was created (False means existing memory was updated).

        Raises:
            ValidationError: If content too long or too many tags.

        On duplicate content:
            - Increments access_count and updates last_accessed_at
            - Merges new tags with existing tags
            - Does NOT change memory_type or source (first write wins)
        """
        # Defense-in-depth validation
        self._validate_content(content)
        tags = self._validate_tags(tags)

        # Project-scoped hash when project awareness is enabled
        # This allows same content to exist in different projects
        if self.settings.project_awareness_enabled and project_id:
            hash_val = content_hash(f"{project_id}:{content}")
        else:
            hash_val = content_hash(content)

        # Trust score based on source type
        trust_scores = {
            MemorySource.MANUAL: self.settings.trust_score_manual,
            MemorySource.MINED: self.settings.trust_score_mined,
        }
        trust_score = trust_scores.get(source, self.settings.trust_score_mined)

        # Compute importance score at admission time (MemGPT-inspired)
        importance_score = 0.5  # Default
        if self.settings.importance_scoring_enabled:
            from memory_mcp.helpers import compute_importance_score

            importance_score = compute_importance_score(
                content,
                self.settings.importance_length_weight,
                self.settings.importance_code_weight,
                self.settings.importance_entity_weight,
            )

        with self.transaction() as conn:
            # Check if memory already exists (project-scoped when filtering enabled)
            if self.settings.project_awareness_enabled and project_id:
                # Project-scoped dedup: same content in different projects = different memories
                existing = conn.execute(
                    "SELECT id FROM memories WHERE content_hash = ? AND project_id = ?",
                    (hash_val, project_id),
                ).fetchone()
            else:
                # Global dedup: same content anywhere = same memory
                existing = conn.execute(
                    "SELECT id FROM memories WHERE content_hash = ?", (hash_val,)
                ).fetchone()

            if existing:
                # Update existing memory - merge tags, increment access
                memory_id = existing[0]
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (memory_id,),
                )

                # Merge new tags (INSERT OR IGNORE handles duplicates)
                if tags:
                    conn.executemany(
                        "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                        [(memory_id, tag) for tag in tags],
                    )
                    log.debug(
                        "Updated existing memory id={} (merged {} tags)",
                        memory_id,
                        len(tags),
                    )
                else:
                    log.debug("Updated existing memory id={}", memory_id)

                return memory_id, False

            # Compute embedding for new content (needed for dedup check and insert)
            embedding = self._embedding_engine.embed(content)

            # Semantic deduplication: check for very similar existing memories
            if self.settings.semantic_dedup_enabled:
                similar = self._find_semantic_duplicate(
                    conn, embedding, self.settings.semantic_dedup_threshold, project_id
                )
                if similar:
                    existing_id, similarity = similar
                    existing_memory = self._get_memory_by_id(conn, existing_id)
                    if existing_memory:
                        return self._merge_with_existing(
                            conn, content, existing_memory, tags, similarity
                        )

            # No duplicate found - insert new memory
            extracted_at = datetime.now().isoformat() if source == MemorySource.MINED else None
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    content, content_hash, memory_type, source, is_hot,
                    trust_score, importance_score, source_log_id, extracted_at,
                    session_id, project_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    content,
                    hash_val,
                    memory_type.value,
                    source.value,
                    int(is_hot),
                    trust_score,
                    importance_score,
                    source_log_id,
                    extracted_at,
                    session_id,
                    project_id,
                ),
            )
            memory_id = cursor.fetchone()[0]

            # Update session memory count if session provided
            if session_id:
                self._update_session_activity(conn, session_id, memory_delta=1)

            # Track project for project-aware features
            if project_id:
                self._track_project(conn, project_id)

            # Insert embedding only for new memories
            conn.execute(
                "INSERT INTO memory_vectors (rowid, embedding) VALUES (?, ?)",
                (memory_id, embedding.tobytes()),
            )

            # Insert tags only for new memories
            if tags:
                conn.executemany(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    [(memory_id, tag) for tag in tags],
                )

            log.info(
                "Stored new memory id={} type={} trust={} project={}",
                memory_id,
                memory_type.value,
                trust_score,
                project_id,
            )

            return memory_id, True

    def _compute_hot_score(self, access_count: int, last_accessed_at: datetime | None) -> float:
        """Compute hot cache score for LRU ranking.

        Score = (access_count * access_weight) + (recency_boost * recency_weight)

        Recency boost uses exponential decay with configurable half-life.
        """
        access_weight = self.settings.hot_score_access_weight
        recency_weight = self.settings.hot_score_recency_weight
        halflife_days = self.settings.hot_score_recency_halflife_days

        # Access count component
        access_score = access_count * access_weight

        # Recency component with exponential decay
        recency_boost = 0.0
        if last_accessed_at:
            days_since_access = (datetime.now() - last_accessed_at).total_seconds() / 86400
            # Exponential decay: 1.0 at time 0, 0.5 at half-life, approaches 0
            recency_boost = 2 ** (-days_since_access / halflife_days) * recency_weight

        return access_score + recency_boost

    def _compute_salience_score(
        self,
        importance_score: float,
        trust_score: float,
        access_count: int,
        last_accessed_at: datetime | None,
    ) -> float:
        """Compute unified salience score for promotion/eviction decisions.

        Combines importance, trust, access frequency, and recency into a single
        metric (Engram-inspired). Used for smarter hot cache promotion than
        simple access count threshold.
        """
        from memory_mcp.helpers import compute_salience_score

        return compute_salience_score(
            importance_score=importance_score,
            trust_score=trust_score,
            access_count=access_count,
            last_accessed_at=last_accessed_at,
            importance_weight=self.settings.salience_importance_weight,
            trust_weight=self.settings.salience_trust_weight,
            access_weight=self.settings.salience_access_weight,
            recency_weight=self.settings.salience_recency_weight,
            recency_halflife_days=self.settings.salience_recency_halflife_days,
            max_access_count=self.settings.hot_cache_max_items,  # Normalize against cache size
        )

    def _get_row_value(self, row: sqlite3.Row, column: str, default=None):
        """Get column value from row, returning default if column doesn't exist."""
        return row[column] if column in row.keys() else default

    def _row_to_memory(
        self,
        row: sqlite3.Row,
        conn: sqlite3.Connection,
        tags: list[str] | None = None,
        similarity: float | None = None,
    ) -> Memory:
        """Convert a database row to a Memory object. Must be called with lock held."""
        if tags is None:
            tags = [
                r["tag"]
                for r in conn.execute(
                    "SELECT tag FROM memory_tags WHERE memory_id = ?", (row["id"],)
                )
            ]

        last_accessed = row["last_accessed_at"]
        last_accessed_dt = datetime.fromisoformat(last_accessed) if last_accessed else None

        # Parse optional columns (may not exist before migrations)
        promo_source_str = self._get_row_value(row, "promotion_source")
        promotion_source = PromotionSource(promo_source_str) if promo_source_str else None

        is_pinned = bool(self._get_row_value(row, "is_pinned", False))
        trust_score_val = self._get_row_value(row, "trust_score", 1.0)
        trust_score = trust_score_val if trust_score_val is not None else 1.0
        importance_score_val = self._get_row_value(row, "importance_score", 0.5)
        importance_score = importance_score_val if importance_score_val is not None else 0.5
        source_log_id = self._get_row_value(row, "source_log_id")
        extracted_at_str = self._get_row_value(row, "extracted_at")
        extracted_at = datetime.fromisoformat(extracted_at_str) if extracted_at_str else None
        session_id = self._get_row_value(row, "session_id")
        project_id = self._get_row_value(row, "project_id")

        hot_score = self._compute_hot_score(row["access_count"], last_accessed_dt)
        salience_score = self._compute_salience_score(
            importance_score, trust_score, row["access_count"], last_accessed_dt
        )

        return Memory(
            id=row["id"],
            content=row["content"],
            content_hash=row["content_hash"],
            memory_type=MemoryType(row["memory_type"]),
            source=MemorySource(row["source"]),
            is_hot=bool(row["is_hot"]),
            is_pinned=is_pinned,
            promotion_source=promotion_source,
            tags=tags,
            access_count=row["access_count"],
            last_accessed_at=last_accessed_dt,
            created_at=datetime.fromisoformat(row["created_at"]),
            trust_score=trust_score,
            importance_score=importance_score,
            source_log_id=source_log_id,
            extracted_at=extracted_at,
            session_id=session_id,
            project_id=project_id,
            similarity=similarity,
            hot_score=hot_score,
            salience_score=salience_score,
        )

    def get_memory(self, memory_id: int) -> Memory | None:
        """Get a memory by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not row:
                return None
            return self._row_to_memory(row, conn)

    def get_memory_by_hash(self, hash_val: str) -> Memory | None:
        """O(1) exact lookup by content hash."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (hash_val,)
            ).fetchone()
            if not row:
                return None
            # Re-fetch with full data using get_memory (RLock allows nested calls)
            return self.get_memory(row["id"])

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory."""
        # Capture content summary for audit before deletion
        memory = self.get_memory(memory_id)
        content_preview = memory.content[:100] if memory else None

        with self.transaction() as conn:
            conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (memory_id,))
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                self._record_audit(
                    conn,
                    AuditOperation.DELETE_MEMORY,
                    target_type="memory",
                    target_id=memory_id,
                    details=json.dumps({"content_preview": content_preview}),
                )
                log.info("Deleted memory id={}", memory_id)
            return deleted

    def update_access(self, memory_id: int) -> None:
        """Update access count and timestamp."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (memory_id,),
            )

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories with optional type filter and pagination."""
        with self._connection() as conn:
            if memory_type:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE memory_type = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (memory_type.value, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()

            return [self._row_to_memory(row, conn) for row in rows]

    def get_stats(self) -> dict:
        """Get memory statistics."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            hot = conn.execute("SELECT COUNT(*) FROM memories WHERE is_hot = 1").fetchone()[0]
            by_type = {
                row["memory_type"]: row["count"]
                for row in conn.execute(
                    "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
                )
            }
            by_source = {
                row["source"]: row["count"]
                for row in conn.execute(
                    "SELECT source, COUNT(*) as count FROM memories GROUP BY source"
                )
            }
            return {
                "total": total,
                "hot": hot,
                "by_type": by_type,
                "by_source": by_source,
            }
