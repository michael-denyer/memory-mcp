"""Session (conversation provenance) mixin for Storage class."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.models import Memory, MemoryType, PromotionSource, Session

if TYPE_CHECKING:
    pass

log = get_logger("storage.sessions")


class SessionsMixin:
    """Mixin providing session management methods for Storage."""

    def _update_session_activity(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        memory_delta: int = 0,
        log_delta: int = 0,
    ) -> None:
        """Update session activity counters. Creates session if needed."""
        # Upsert: insert or update session
        conn.execute(
            """
            INSERT INTO sessions (id, memory_count, log_count)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_activity_at = CURRENT_TIMESTAMP,
                memory_count = memory_count + excluded.memory_count,
                log_count = log_count + excluded.log_count
            """,
            (session_id, memory_delta, log_delta),
        )

    def _track_project(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        name: str | None = None,
        path: str | None = None,
    ) -> None:
        """Track a project for project awareness. Creates or updates last_accessed_at."""
        conn.execute(
            """
            INSERT INTO projects (id, name, path)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_accessed_at = CURRENT_TIMESTAMP,
                name = COALESCE(excluded.name, projects.name),
                path = COALESCE(excluded.path, projects.path)
            """,
            (project_id, name, path),
        )

    def create_or_get_session(
        self,
        session_id: str,
        topic: str | None = None,
        project_path: str | None = None,
    ) -> Session:
        """Create a new session or return existing one.

        Args:
            session_id: Unique session identifier (UUID or transcript hash)
            topic: Optional topic description
            project_path: Working directory for the session

        Returns:
            Session object (created or existing)
        """
        with self.transaction() as conn:
            # Try to get existing
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()

            if row:
                return self._row_to_session(row)

            # Create new session
            conn.execute(
                """
                INSERT INTO sessions (id, topic, project_path)
                VALUES (?, ?, ?)
                """,
                (session_id, topic, project_path),
            )

            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            log.info("Created session id={} topic={}", session_id, topic)
            return self._row_to_session(row)

    def update_session_topic(self, session_id: str, topic: str) -> bool:
        """Update the topic for a session."""
        with self.transaction() as conn:
            cursor = conn.execute("UPDATE sessions SET topic = ? WHERE id = ?", (topic, session_id))
            return cursor.rowcount > 0

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return self._row_to_session(row) if row else None

    def get_sessions(
        self,
        limit: int = 20,
        project_path: str | None = None,
    ) -> list[Session]:
        """Get recent sessions, optionally filtered by project.

        Args:
            limit: Maximum sessions to return
            project_path: Filter to sessions from this project

        Returns:
            List of sessions ordered by last activity (most recent first)
        """
        with self._connection() as conn:
            if project_path:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE project_path = ?
                    ORDER BY last_activity_at DESC
                    LIMIT ?
                    """,
                    (project_path, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    ORDER BY last_activity_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [self._row_to_session(row) for row in rows]

    def get_session_memories(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories from a specific session.

        Args:
            session_id: Session to get memories from
            limit: Maximum memories to return

        Returns:
            List of memories from the session, ordered by creation time
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT m.*, GROUP_CONCAT(t.tag, ',') as tags_str
                FROM memories m
                LEFT JOIN memory_tags t ON m.id = t.memory_id
                WHERE m.session_id = ?
                GROUP BY m.id
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

            memories = []
            for row in rows:
                tags_str = row["tags_str"]
                tags = tags_str.split(",") if tags_str else []
                memories.append(self._row_to_memory(row, conn, tags=tags))

            return memories

    def get_cross_session_patterns(self, min_sessions: int = 2) -> list[dict]:
        """Find content patterns that appear across multiple sessions.

        Useful for identifying frequently-discussed topics that might
        warrant promotion to hot cache.

        Args:
            min_sessions: Minimum sessions a pattern must appear in

        Returns:
            List of dicts with pattern info and session counts
        """
        with self._connection() as conn:
            # Find memories that appear in multiple sessions via similar content
            # This uses a simple approach: group by content_hash
            rows = conn.execute(
                """
                SELECT
                    content,
                    memory_type,
                    COUNT(DISTINCT session_id) as session_count,
                    SUM(access_count) as total_accesses,
                    GROUP_CONCAT(DISTINCT session_id) as sessions
                FROM memories
                WHERE session_id IS NOT NULL
                GROUP BY content_hash
                HAVING COUNT(DISTINCT session_id) >= ?
                ORDER BY session_count DESC, total_accesses DESC
                LIMIT 50
                """,
                (min_sessions,),
            ).fetchall()

            return [
                {
                    "content": row["content"][:200],  # Truncate for display
                    "memory_type": row["memory_type"],
                    "session_count": row["session_count"],
                    "total_accesses": row["total_accesses"],
                    "sessions": row["sessions"].split(",") if row["sessions"] else [],
                }
                for row in rows
            ]

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert a database row to a Session object."""
        return Session(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            last_activity_at=datetime.fromisoformat(row["last_activity_at"]),
            topic=row["topic"],
            project_path=row["project_path"],
            memory_count=row["memory_count"],
            log_count=row["log_count"],
        )

    # ========== Episodic Memory ==========

    def _compute_memory_salience(self, memory: Memory) -> float:
        """Compute and store salience score for a memory.

        Args:
            memory: Memory to compute salience for

        Returns:
            Computed salience score
        """
        salience = self._compute_salience_score(
            importance_score=memory.importance_score,
            trust_score=memory.trust_score,
            access_count=memory.access_count,
            last_accessed_at=memory.last_accessed_at,
        )
        memory.salience_score = salience
        return salience

    def summarize_session(self, session_id: str) -> dict:
        """Summarize a session's key decisions, insights, and action items.

        Groups session memories by semantic category to extract:
        - Decisions: Choices made and their rationale
        - Insights: Lessons learned, antipatterns, landmines, constraints
        - Action Items: TODOs, bugs, tasks to complete
        - Context: Background info, conventions, preferences, architecture

        This can be called before end_session() to review what will be promoted,
        or anytime to get a structured view of the conversation.

        Args:
            session_id: Session to summarize

        Returns:
            Dict with categorized memories and counts
        """
        session = self.get_session(session_id)
        if session is None:
            return {"success": False, "error": f"Session not found: {session_id}"}

        memories = self.get_session_memories(session_id, limit=500)

        # Category groupings
        decision_categories = {"decision"}
        insight_categories = {"lesson", "antipattern", "landmine", "constraint"}
        action_categories = {"todo", "bug"}

        decisions: list[dict] = []
        insights: list[dict] = []
        action_items: list[dict] = []
        context: list[dict] = []

        for memory in memories:
            entry = {
                "id": memory.id,
                "content": memory.content[:300],  # Truncate for display
                "category": memory.category,
                "memory_type": memory.memory_type.value if memory.memory_type else None,
                "importance": round(memory.importance_score or 0.5, 2),
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
            }

            category = memory.category or ""
            if category in decision_categories:
                decisions.append(entry)
            elif category in insight_categories:
                insights.append(entry)
            elif category in action_categories:
                action_items.append(entry)
            else:
                context.append(entry)

        # Sort each group by importance (descending)
        decisions.sort(key=lambda x: x["importance"], reverse=True)
        insights.sort(key=lambda x: x["importance"], reverse=True)
        action_items.sort(key=lambda x: x["importance"], reverse=True)
        context.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "success": True,
            "session_id": session_id,
            "topic": session.topic,
            "total_memories": len(memories),
            "decisions": decisions[:20],  # Top 20 of each
            "insights": insights[:20],
            "action_items": action_items[:20],
            "context": context[:20],
            "summary": {
                "decisions_count": len(decisions),
                "insights_count": len(insights),
                "action_items_count": len(action_items),
                "context_count": len(context),
            },
        }

    def end_session(
        self,
        session_id: str,
        promote_top: bool = True,
        promote_type: MemoryType = MemoryType.PROJECT,
    ) -> dict:
        """End a session and optionally promote top episodic memories.

        Mirrors human memory consolidation: episodic (short-term) memories
        that prove valuable get promoted to semantic (long-term) storage.

        Args:
            session_id: Session to end
            promote_top: Whether to promote top memories to long-term storage
            promote_type: Memory type for promoted memories (default: PROJECT)

        Returns:
            Dict with session summary and promotion results
        """
        session = self.get_session(session_id)
        if session is None:
            return {"success": False, "error": f"Session not found: {session_id}"}

        # Get and rank episodic memories by salience
        session_memories = self.get_session_memories(session_id)
        episodic_memories = [m for m in session_memories if m.memory_type == MemoryType.EPISODIC]

        for memory in episodic_memories:
            self._compute_memory_salience(memory)

        episodic_memories.sort(key=lambda m: m.salience_score or 0, reverse=True)

        # Promote top memories above threshold
        promoted_ids: list[int] = []
        if promote_top and episodic_memories:
            threshold = self.settings.episodic_promote_threshold
            candidates = [
                m
                for m in episodic_memories[: self.settings.episodic_promote_top_n]
                if (m.salience_score or 0) >= threshold
            ]

            if candidates:
                # Batch update memory types in single transaction
                with self.transaction() as conn:
                    for memory in candidates:
                        conn.execute(
                            "UPDATE memories SET memory_type = ? WHERE id = ?",
                            (promote_type.value, memory.id),
                        )
                        promoted_ids.append(memory.id)

                # Promote to hot cache after transaction commits
                for memory in candidates:
                    if not memory.is_hot:
                        self.promote_to_hot(memory.id, PromotionSource.SESSION_END)
                    log.info(
                        "Promoted episodic memory id={} to {} (salience={:.2f})",
                        memory.id,
                        promote_type.value,
                        memory.salience_score or 0,
                    )

        log.info(
            "Ended session {} with {} episodic memories, {} promoted",
            session_id[:8],
            len(episodic_memories),
            len(promoted_ids),
        )

        return {
            "success": True,
            "session_id": session_id,
            "episodic_count": len(episodic_memories),
            "promoted_count": len(promoted_ids),
            "promoted_ids": promoted_ids,
            "top_memories": [
                {
                    "id": m.id,
                    "content": m.content[:100],
                    "salience": round(m.salience_score or 0, 3),
                }
                for m in episodic_memories[:5]
            ],
        }
