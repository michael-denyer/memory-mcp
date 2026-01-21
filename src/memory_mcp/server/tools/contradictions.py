"""Contradiction detection tools: find_contradictions, get_contradictions, mark/resolve."""

from typing import Annotated

from pydantic import Field

from memory_mcp.responses import (
    ContradictionPairResponse,
    ContradictionResponse,
    RelationshipResponse,
    error_response,
    memory_to_response,
    relation_to_response,
    success_response,
)
from memory_mcp.server.app import mcp, storage


@mcp.tool
def find_contradictions(
    memory_id: Annotated[int, Field(description="Memory ID to check for contradictions")],
    similarity_threshold: Annotated[
        float,
        Field(description="Minimum similarity to consider (0.75 = same topic)"),
    ] = 0.75,
    limit: Annotated[int, Field(description="Maximum contradictions to return")] = 5,
) -> list[ContradictionResponse]:
    """Find memories that may contradict a given memory.

    Searches for semantically similar memories that could contain
    conflicting information. High similarity means same topic area,
    which is where contradictions are likely.

    Use this after storing a new memory or when validating existing ones.
    """
    contradictions = storage.find_contradictions(
        memory_id=memory_id,
        similarity_threshold=similarity_threshold,
        limit=limit,
    )
    return [
        ContradictionResponse(
            memory_a=memory_to_response(c.memory_a),
            memory_b=memory_to_response(c.memory_b),
            similarity=c.similarity,
            already_linked=c.already_linked,
        )
        for c in contradictions
    ]


@mcp.tool
def get_contradictions() -> list[ContradictionPairResponse]:
    """Get all memory pairs marked as contradictions.

    Returns pairs that have been flagged as containing conflicting
    information. Use resolve_contradiction to handle these.
    """
    contradictions = storage.get_all_contradictions()
    return [
        ContradictionPairResponse(
            memory_a=memory_to_response(m1),
            memory_b=memory_to_response(m2),
            relationship=relation_to_response(rel),
        )
        for m1, m2, rel in contradictions
    ]


@mcp.tool
def mark_contradiction(
    memory_id_a: Annotated[int, Field(description="First memory ID")],
    memory_id_b: Annotated[int, Field(description="Second memory ID")],
) -> RelationshipResponse | dict:
    """Mark two memories as contradicting each other.

    Creates a CONTRADICTS relationship. Use this when you discover
    that two memories contain conflicting information about the same topic.
    """
    relation = storage.mark_contradiction(memory_id_a, memory_id_b)
    if relation is None:
        return error_response(
            f"Failed to mark contradiction between #{memory_id_a} and #{memory_id_b}. "
            "Check that both memories exist and aren't already marked as contradicting."
        )
    return relation_to_response(relation)


@mcp.tool
def resolve_contradiction(
    memory_id_a: Annotated[int, Field(description="First memory in contradiction")],
    memory_id_b: Annotated[int, Field(description="Second memory in contradiction")],
    keep_id: Annotated[int, Field(description="ID of memory to keep (must be one of the two)")],
    resolution: Annotated[
        str,
        Field(
            description=(
                "How to handle the discarded memory: "
                "'supersedes' (kept memory replaces other), "
                "'delete' (remove the other memory), "
                "'weaken' (reduce trust in other memory)"
            )
        ),
    ] = "supersedes",
) -> dict:
    """Resolve a contradiction by keeping one memory and handling the other.

    After resolving:
    - 'supersedes': Creates SUPERSEDES relationship, weakens trust in discarded
    - 'delete': Removes the discarded memory entirely
    - 'weaken': Reduces trust in discarded memory but keeps it
    """
    if resolution not in ("supersedes", "delete", "weaken"):
        return error_response(
            f"Invalid resolution '{resolution}'. Use: supersedes, delete, or weaken"
        )

    success = storage.resolve_contradiction(memory_id_a, memory_id_b, keep_id, resolution)
    if not success:
        return error_response(
            "Failed to resolve contradiction. Ensure keep_id is one of the two memories."
        )

    other_id = memory_id_b if keep_id == memory_id_a else memory_id_a
    return success_response(f"Resolved contradiction: kept #{keep_id}, {resolution} #{other_id}")
