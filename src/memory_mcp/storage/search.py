"""Vector search and recall mixin for Storage class."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from memory_mcp.logging import get_logger
from memory_mcp.models import (
    Memory,
    MemoryType,
    RecallMode,
    RecallModeConfig,
    RecallResult,
    RelationType,
    ScoreBreakdown,
    TrustReason,
)

log = get_logger("storage.search")


class SearchMixin:
    """Mixin providing vector search and recall methods for Storage."""

    def get_recall_mode_config(self, mode: RecallMode) -> RecallModeConfig:
        """Get configuration for a recall mode preset."""
        if mode == RecallMode.PRECISION:
            return RecallModeConfig(
                threshold=self.settings.precision_threshold,
                limit=self.settings.precision_limit,
                similarity_weight=self.settings.precision_similarity_weight,
                recency_weight=self.settings.precision_recency_weight,
                access_weight=self.settings.precision_access_weight,
            )
        elif mode == RecallMode.EXPLORATORY:
            return RecallModeConfig(
                threshold=self.settings.exploratory_threshold,
                limit=self.settings.exploratory_limit,
                similarity_weight=self.settings.exploratory_similarity_weight,
                recency_weight=self.settings.exploratory_recency_weight,
                access_weight=self.settings.exploratory_access_weight,
            )
        else:  # BALANCED (default)
            return RecallModeConfig(
                threshold=self.settings.default_confidence_threshold,
                limit=self.settings.default_recall_limit,
                similarity_weight=self.settings.recall_similarity_weight,
                recency_weight=self.settings.recall_recency_weight,
                access_weight=self.settings.recall_access_weight,
            )

    def _compute_recency_score(
        self, created_at: datetime, last_accessed_at: datetime | None = None
    ) -> float:
        """Compute recency score (0-1) with exponential decay.

        Uses last_accessed_at when available, falling back to created_at.
        This ensures frequently-used memories maintain high recency scores,
        even if created long ago.

        Returns 1.0 for just-accessed/created items, decaying to 0.5 at half-life.
        """
        halflife_days = self.settings.recall_recency_halflife_days
        reference_time = last_accessed_at if last_accessed_at else created_at
        days_old = (datetime.now() - reference_time).total_seconds() / 86400
        return 2 ** (-days_old / halflife_days)

    def _compute_trust_decay(
        self,
        base_trust: float,
        created_at: datetime,
        last_accessed_at: datetime | None = None,
        memory_type: MemoryType | None = None,
        access_count: int = 0,
    ) -> float:
        """Compute time-decayed trust score with usage-aware adjustment.

        Trust decays based on time since last meaningful interaction:
        - If recently accessed, decay is based on last_accessed_at (refresh on use)
        - Otherwise, decay is based on created_at
        - Decay rate varies by memory type (project decays slowest, conversation fastest)
        - Frequently-used memories decay slower (usage multiplier)

        This means memories that are actively used maintain their trust,
        while unused memories slowly decay - aligning with Engram's principle
        that frequently-used patterns should remain reliable.

        Args:
            base_trust: Initial trust (1.0 for manual, 0.7 for mined by default)
            created_at: When the memory was created
            last_accessed_at: When the memory was last accessed (optional)
            memory_type: Type of memory for per-type decay rate
            access_count: Number of times memory has been accessed (for usage-aware decay)

        Returns:
            Trust score with exponential decay applied based on staleness,
            adjusted for usage frequency.
        """
        import math

        halflife_days = self._get_trust_decay_halflife(memory_type)
        reference_time = last_accessed_at if last_accessed_at else created_at
        days_since_activity = (datetime.now() - reference_time).total_seconds() / 86400

        # Base exponential decay
        base_decay = 2 ** (-days_since_activity / halflife_days)

        # Usage multiplier: frequently-used memories decay slower
        # log(1) = 0, log(10) ≈ 2.3, log(100) ≈ 4.6
        # Multiplier ranges from 1.0 (unused) to 1.5 (heavily used)
        usage_factor = min(1.5, 1.0 + 0.1 * math.log(max(1, access_count + 1)))

        # Apply usage factor only if there's actual decay happening
        # (avoids inflating trust above base for fresh memories)
        adjusted_decay = base_decay * usage_factor if base_decay < 1.0 else base_decay

        return base_trust * min(1.0, adjusted_decay)

    def _compute_access_score(self, access_count: int, max_access: int) -> float:
        """Normalize access count to 0-1 range."""
        if max_access <= 0:
            return 0.0
        return min(1.0, access_count / max_access)

    def _get_fts_matches(
        self, query: str, conn: sqlite3.Connection, limit: int = 50
    ) -> dict[int, float]:
        """Get keyword matches from FTS5 with BM25 scores.

        Returns dict mapping memory_id to normalized keyword score (0-1).
        Used for hybrid search to boost semantically weak but keyword-relevant results.
        """
        # Check if FTS table exists
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_fts'"
        ).fetchone()
        if not table_exists:
            return {}

        # Escape FTS5 special characters and prepare query
        # Note: We use exact word matching (no * suffix) for precision
        # FTS5's Porter stemmer already handles word variations
        words = query.split()
        if not words:
            return {}

        # Build query with OR logic (any word match)
        # Escape double quotes in the query
        escaped_words = [w.replace('"', '""') for w in words if len(w) >= 2]
        if not escaped_words:
            return {}

        # Use OR matching - any keyword match is useful
        fts_query = " OR ".join(f'"{w}"' for w in escaped_words)

        try:
            rows = conn.execute(
                """
                SELECT rowid, bm25(memory_fts) as score
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY bm25(memory_fts)
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS match failed (malformed query, etc)
            return {}

        if not rows:
            return {}

        # BM25 scores are negative (closer to 0 is better)
        # Normalize to 0-1 range (best match = 1.0)
        min_score = min(row["score"] for row in rows)
        max_score = max(row["score"] for row in rows)
        score_range = max_score - min_score if max_score != min_score else 1.0

        return {
            row["rowid"]: 1.0 - (row["score"] - min_score) / score_range if score_range else 1.0
            for row in rows
        }

    def _generate_recall_guidance(
        self,
        confidence: str,
        result_count: int,
        gated_count: int,
        mode: RecallMode,
    ) -> str:
        """Generate hallucination prevention guidance based on recall results.

        Provides explicit instructions on how to use (or not use) the results.
        """
        if confidence == "high" and result_count > 0:
            return (
                "HIGH CONFIDENCE: Use these memories directly. "
                "The top result closely matches your query."
            )
        elif confidence == "medium" and result_count > 0:
            return (
                "MEDIUM CONFIDENCE: Verify these memories apply to current context. "
                "Results are relevant but may need validation."
            )
        elif result_count == 0 and gated_count > 0:
            # Had results but all were below threshold
            return (
                f"NO CONFIDENT MATCH: {gated_count} memories were found but filtered "
                "due to low similarity. Reason from first principles or try a "
                "different query. Do NOT guess or hallucinate information."
            )
        elif result_count == 0:
            # No results at all
            suggestions = []
            if mode == RecallMode.PRECISION:
                suggestions.append("try 'exploratory' mode for broader search")
            suggestions.append("try rephrasing your query")
            suggestions.append("store relevant information with 'remember' first")

            return (
                "NO MATCH FOUND: No relevant memories exist for this query. "
                f"Suggestions: {'; '.join(suggestions)}. "
                "Do NOT fabricate or guess information."
            )
        else:
            # Low confidence with some results
            return (
                "LOW CONFIDENCE: Results have weak similarity to your query. "
                "Use with caution and verify independently. Consider that the "
                "information you need may not be stored yet."
            )

    def _compute_composite_score(
        self,
        similarity: float,
        recency_score: float,
        access_score: float,
        trust_score: float = 1.0,
        helpfulness_score: float = 0.25,
        weights: RecallModeConfig | None = None,
    ) -> ScoreBreakdown:
        """Compute weighted composite score for ranking.

        Combines semantic similarity with recency, access frequency, trust, and helpfulness.
        Trust and helpfulness weights are optional (default 0 and 0.05 respectively).

        Args:
            similarity: Semantic similarity score (0-1)
            recency_score: Time-decayed recency score (0-1)
            access_score: Normalized access count (0-1)
            trust_score: Trust score with decay (0-1)
            helpfulness_score: Bayesian utility_score (0-1), default 0.25 cold start
            weights: Optional custom weights from recall mode preset

        Returns:
            ScoreBreakdown with total and individual weighted components.
        """
        if weights:
            sim_weight = weights.similarity_weight
            rec_weight = weights.recency_weight
            acc_weight = weights.access_weight
        else:
            sim_weight = self.settings.recall_similarity_weight
            rec_weight = self.settings.recall_recency_weight
            acc_weight = self.settings.recall_access_weight

        trust_weight = self.settings.recall_trust_weight
        helpfulness_weight = self.settings.recall_helpfulness_weight

        sim_component = similarity * sim_weight
        rec_component = recency_score * rec_weight
        acc_component = access_score * acc_weight
        trust_component = trust_score * trust_weight
        helpfulness_component = helpfulness_score * helpfulness_weight

        return ScoreBreakdown(
            total=sim_component
            + rec_component
            + acc_component
            + trust_component
            + helpfulness_component,
            similarity_component=sim_component,
            recency_component=rec_component,
            access_component=acc_component,
            trust_component=trust_component,
            helpfulness_component=helpfulness_component,
        )

    def recall(
        self,
        query: str,
        limit: int | None = None,
        threshold: float | None = None,
        mode: RecallMode | None = None,
        memory_types: list[MemoryType] | None = None,
        expand_relations: bool | None = None,
        project_id: str | None = None,
    ) -> RecallResult:
        """Semantic search with confidence gating and composite ranking.

        Args:
            query: Search query for semantic similarity
            limit: Maximum results (overrides mode preset if set)
            threshold: Minimum similarity (overrides mode preset if set)
            mode: Recall mode preset (precision, balanced, exploratory)
            memory_types: Filter to specific memory types
            expand_relations: Expand results via knowledge graph (default from config)
            project_id: Filter to specific project (also includes global memories
                if project_include_global is enabled)

        Results are ranked by composite score combining:
        - Semantic similarity (weight varies by mode)
        - Recency with exponential decay
        - Access frequency
        - Trust score with decay (optional)
        """
        # Get mode config (or default balanced)
        mode_config = self.get_recall_mode_config(mode or RecallMode.BALANCED)

        # Allow explicit overrides
        effective_limit = limit if limit is not None else mode_config.limit
        effective_threshold = threshold if threshold is not None else mode_config.threshold

        # Infer query intent for category-based ranking boost
        from memory_mcp.helpers import infer_query_intent

        intent_boosts = infer_query_intent(query)

        query_embedding = self._embedding_engine.embed(query)

        with self._connection() as conn:
            # Build query with optional type and project filters
            # Include all Memory fields for accurate response mapping
            base_select = """
                SELECT
                    m.id,
                    m.content,
                    m.content_hash,
                    m.memory_type,
                    m.source,
                    m.is_hot,
                    m.is_pinned,
                    m.promotion_source,
                    m.access_count,
                    m.last_accessed_at,
                    m.created_at,
                    m.trust_score,
                    m.source_log_id,
                    m.extracted_at,
                    m.session_id,
                    m.project_id,
                    m.utility_score,
                    m.category,
                    m.last_used_at,
                    vec_distance_cosine(v.embedding, ?) as distance
                FROM memory_vectors v
                JOIN memories m ON m.id = v.rowid
            """

            # Build WHERE conditions
            conditions = []
            params: list = [query_embedding.tobytes()]

            if memory_types:
                type_placeholders = ",".join("?" * len(memory_types))
                conditions.append(f"m.memory_type IN ({type_placeholders})")
                params.extend([t.value for t in memory_types])

            # Project filtering: include project-specific + global (NULL project_id)
            if project_id:
                if self.settings.project_include_global:
                    conditions.append("(m.project_id = ? OR m.project_id IS NULL)")
                else:
                    conditions.append("m.project_id = ?")
                params.append(project_id)

            # Build final query
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query_sql = f"""
                {base_select}
                {where_clause}
                ORDER BY distance ASC
                LIMIT ?
            """
            params.append(effective_limit * 3)

            rows = conn.execute(query_sql, params).fetchall()

            # Get FTS keyword matches for hybrid search
            fts_matches: dict[int, float] = {}
            if self.settings.hybrid_search_enabled:
                fts_matches = self._get_fts_matches(query, conn, limit=effective_limit * 3)

            # Find max access count for normalization
            max_access = max((row["access_count"] for row in rows), default=1)

            # Convert distance to similarity and compute scores
            candidates = []
            gated_count = 0
            keyword_weight = self.settings.hybrid_keyword_weight
            keyword_boost_threshold = self.settings.hybrid_keyword_boost_threshold

            for row in rows:
                similarity = 1 - row["distance"]  # cosine distance to similarity
                memory_id = row["id"]

                # Hybrid scoring: boost low-similarity results if they have keyword matches
                # This helps catch cases like "FastAPI" matching "framework" queries
                keyword_score = fts_matches.get(memory_id, 0.0)
                effective_similarity = similarity

                if self.settings.hybrid_search_enabled and keyword_score > 0:
                    # Apply keyword boost inversely proportional to semantic similarity
                    # Low semantic = more keyword influence, high semantic = less keyword influence
                    if similarity < keyword_boost_threshold:
                        # For low semantic matches, keyword score can significantly boost
                        boost_factor = 1 - (similarity / keyword_boost_threshold)
                        effective_similarity = (
                            similarity * (1 - keyword_weight * boost_factor)
                            + keyword_score * keyword_weight * boost_factor
                        )
                    else:
                        # For high semantic matches, small keyword bonus
                        effective_similarity = (
                            similarity * (1 - keyword_weight * 0.3)
                            + keyword_score * keyword_weight * 0.3
                        )

                if effective_similarity >= effective_threshold:
                    created_at = datetime.fromisoformat(row["created_at"])
                    last_accessed_str = row["last_accessed_at"]
                    last_accessed_at = (
                        datetime.fromisoformat(last_accessed_str) if last_accessed_str else None
                    )
                    recency_score = self._compute_recency_score(created_at, last_accessed_at)
                    access_score = self._compute_access_score(row["access_count"], max_access)

                    # Compute trust with time decay (refreshed by access, per-type rate)
                    # Usage-aware: frequently-used memories decay slower
                    memory_type_enum = (
                        MemoryType(row["memory_type"]) if row["memory_type"] else None
                    )
                    base_trust = row["trust_score"] if row["trust_score"] is not None else 1.0
                    trust_decayed = self._compute_trust_decay(
                        base_trust,
                        created_at,
                        last_accessed_at,
                        memory_type_enum,
                        row["access_count"] or 0,
                    )

                    # Get Bayesian helpfulness with category-aware recency decay
                    from memory_mcp.helpers import compute_helpfulness_with_decay

                    utility_score = row["utility_score"]
                    base_helpfulness = utility_score if utility_score is not None else 0.25
                    category = row["category"]
                    last_used_str = row["last_used_at"]
                    last_used_at = datetime.fromisoformat(last_used_str) if last_used_str else None
                    helpfulness = compute_helpfulness_with_decay(
                        base_helpfulness, last_used_at, category
                    )

                    score_breakdown = self._compute_composite_score(
                        effective_similarity,
                        recency_score,
                        access_score,
                        trust_decayed,
                        helpfulness,
                        weights=mode_config,
                    )

                    # Apply intent-based category boost
                    from memory_mcp.helpers import compute_intent_boost

                    intent_boost = compute_intent_boost(category, intent_boosts)
                    final_score = score_breakdown.total + intent_boost

                    memory = self._row_to_memory(row, conn, similarity=similarity)
                    memory.recency_score = recency_score
                    memory.trust_score_decayed = trust_decayed
                    memory.composite_score = final_score
                    # Populate weighted components for transparency
                    memory.similarity_component = score_breakdown.similarity_component
                    memory.recency_component = score_breakdown.recency_component
                    memory.access_component = score_breakdown.access_component
                    memory.trust_component = score_breakdown.trust_component
                    memory.helpfulness_component = score_breakdown.helpfulness_component
                    # Store keyword score for debugging/transparency
                    memory.keyword_score = keyword_score if keyword_score > 0 else None
                    # Store intent boost for debugging/transparency
                    memory.intent_boost = intent_boost if intent_boost > 0 else None
                    candidates.append(memory)
                else:
                    gated_count += 1

            # Re-rank by composite score
            candidates.sort(key=lambda m: m.composite_score or 0, reverse=True)

            # Take top results and update access counts and retrieved_count
            memories = candidates[:effective_limit]
            memory_ids_to_check = []
            for memory in memories:
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        retrieved_count = retrieved_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (memory.id,),
                )
                # Track for auto-promotion check (if not already hot)
                if not memory.is_hot:
                    memory_ids_to_check.append(memory.id)
            conn.commit()

        # Check for auto-promotion outside the transaction
        for memory_id in memory_ids_to_check:
            self.check_auto_promote(memory_id)

        # Record access sequences for predictive cache (if enabled)
        if self.settings.predictive_cache_enabled and len(memories) >= 2:
            memory_ids = [m.id for m in memories]
            for i in range(len(memory_ids) - 1):
                self.record_access_sequence(memory_ids[i], memory_ids[i + 1])

        # Auto-warm hot cache with predicted next memories (based on top result)
        if self.settings.predictive_cache_enabled and memories:
            self.warm_predicted_cache(memories[0].id)

        # Auto-strengthen trust for high-similarity matches (confidence-weighted)
        if self.settings.trust_auto_strengthen_on_recall:
            for memory in memories:
                if (
                    memory.similarity
                    and memory.similarity >= self.settings.trust_high_similarity_threshold
                ):
                    # Confidence-weighted: similarity scales the boost
                    self.adjust_trust(
                        memory.id,
                        reason=TrustReason.HIGH_SIMILARITY_HIT,
                        delta=self.settings.trust_high_similarity_boost,
                        similarity=memory.similarity,
                    )

        # Determine confidence level based on top result's similarity
        if not memories:
            confidence = "low"
        elif (
            memories[0].similarity
            and memories[0].similarity > self.settings.high_confidence_threshold
        ):
            confidence = "high"
        else:
            confidence = "medium"

        # Generate hallucination prevention guidance
        effective_mode = mode or RecallMode.BALANCED
        guidance = self._generate_recall_guidance(
            confidence, len(memories), gated_count, effective_mode
        )

        # Expand via knowledge graph if enabled
        should_expand = (
            expand_relations
            if expand_relations is not None
            else self.settings.recall_expand_relations
        )
        if should_expand and memories:
            memories = self.expand_via_relations(memories)

        log.debug(
            "Recall query='{}' mode={} returned {} results (confidence={}, gated={})",
            query[:50],
            effective_mode.value,
            len(memories),
            confidence,
            gated_count,
        )

        return RecallResult(
            memories=memories,
            confidence=confidence,
            gated_count=gated_count,
            mode=effective_mode,
            guidance=guidance,
        )

    def _get_memories_by_ids(self, ids: list[int]) -> list[Memory]:
        """Fetch multiple memories by ID, filtering out None results."""
        return [m for mid in ids if (m := self.get_memory(mid)) is not None]

    def expand_via_relations(
        self,
        memories: list[Memory],
        max_per_memory: int | None = None,
        decay_factor: float | None = None,
    ) -> list[Memory]:
        """Expand recall results by traversing knowledge graph relations.

        For each memory, finds related memories via the knowledge graph and adds
        them with a decayed score. This implements Engram-style associative recall
        where one memory activates related memories.

        Relation handling:
        - relates_to, depends_on, elaborates: Add related memory
        - contradicts: Add with flag for user awareness
        - supersedes: Prefer the superseding (newer) memory

        Args:
            memories: Initial recall results to expand
            max_per_memory: Max related memories per source (default from config)
            decay_factor: Score decay for expanded results (default from config)

        Returns:
            Expanded list with related memories appended (deduplicated).
        """
        max_expansion = max_per_memory or self.settings.recall_max_expansion
        decay = decay_factor or self.settings.recall_expansion_decay

        # Track already-included memory IDs to avoid duplicates
        seen_ids = {m.id for m in memories}
        expanded: list[Memory] = []

        for source_memory in memories:
            # Get related memories (1-hop only)
            related = self.get_related(source_memory.id, direction="both")

            added_count = 0
            for related_memory, relation in related:
                if added_count >= max_expansion:
                    break
                if related_memory.id in seen_ids:
                    continue

                # Handle supersedes specially - if this memory supersedes another,
                # we already have the newer version, skip the old one
                if relation.relation_type == RelationType.SUPERSEDES:
                    if relation.from_memory_id == source_memory.id:
                        # source supersedes related - skip related (it's outdated)
                        continue

                # Apply score decay from parent
                if source_memory.composite_score is not None:
                    related_memory.composite_score = source_memory.composite_score * decay
                if source_memory.similarity is not None:
                    related_memory.similarity = source_memory.similarity * decay

                expanded.append(related_memory)
                seen_ids.add(related_memory.id)
                added_count += 1

        return memories + expanded

    def recall_with_fallback(
        self,
        query: str,
        fallback_types: list[list[MemoryType]] | None = None,
        mode: RecallMode | None = None,
        min_results: int = 1,
    ) -> RecallResult:
        """Recall with multi-query fallback through different memory type filters.

        Tries each type filter in sequence until min_results are found or
        all fallbacks exhausted. Default fallback order:
        1. patterns only (code snippets)
        2. project facts
        3. all types (no filter)

        Args:
            query: Search query
            fallback_types: List of type filters to try in order
            mode: Recall mode preset
            min_results: Minimum results needed before stopping fallback

        Returns:
            RecallResult from first successful search, or last attempt
        """
        if fallback_types is None:
            # Default fallback: patterns -> project -> all
            fallback_types = [
                [MemoryType.PATTERN],
                [MemoryType.PROJECT],
                None,  # All types
            ]

        best_result: RecallResult | None = None

        for type_filter in fallback_types:
            result = self.recall(
                query=query,
                mode=mode,
                memory_types=type_filter,
            )

            # Track best result (most memories found)
            if best_result is None or len(result.memories) > len(best_result.memories):
                best_result = result

            # Stop if we have enough high-quality results
            if len(result.memories) >= min_results and result.confidence != "low":
                log.debug(
                    "Fallback succeeded with type_filter={} ({} results)",
                    [t.value for t in type_filter] if type_filter else "all",
                    len(result.memories),
                )
                return result

        # Return best result found (or empty)
        log.debug(
            "Fallback exhausted, returning best result ({} memories)",
            len(best_result.memories) if best_result else 0,
        )
        return best_result or RecallResult(memories=[], confidence="low", gated_count=0)

    def recall_by_tag(self, tag: str, limit: int | None = None) -> list[Memory]:
        """Get memories by tag."""
        limit = limit or self.settings.default_recall_limit

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT m.id FROM memories m
                JOIN memory_tags t ON t.memory_id = m.id
                WHERE t.tag = ?
                ORDER BY m.access_count DESC, m.created_at DESC
                LIMIT ?
                """,
                (tag, limit),
            ).fetchall()
            ids = [row["id"] for row in rows]

        return self._get_memories_by_ids(ids)
