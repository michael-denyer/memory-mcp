"""Relationship tools: link_memories, unlink_memories, get_related_memories, relationship_stats."""

from typing import Annotated

from pydantic import Field

from memory_mcp.responses import (
    RelatedMemoryResponse,
    RelationshipResponse,
    RelationshipStatsResponse,
    error_response,
    memory_to_response,
    relation_to_response,
    success_response,
)
from memory_mcp.server.app import mcp, storage
from memory_mcp.storage import RelationType


def parse_relation_type(relation_type: str) -> RelationType | None:
    """Parse relation type string, returning None if invalid."""
    try:
        return RelationType(relation_type)
    except ValueError:
        return None


def invalid_relation_type_error() -> dict:
    """Return error for invalid relation type."""
    return error_response(f"Invalid relation_type. Use: {[t.value for t in RelationType]}")


@mcp.tool
def link_memories(
    from_memory_id: Annotated[int, Field(description="Source memory ID")],
    to_memory_id: Annotated[int, Field(description="Target memory ID")],
    relation_type: Annotated[
        str,
        Field(
            description=(
                "Relationship type: 'relates_to' (general), 'depends_on' (prerequisite), "
                "'supersedes' (replaces), 'refines' (more specific), 'contradicts' (conflict), "
                "'elaborates' (more detail), 'mentions' (references entity)"
            )
        ),
    ],
) -> RelationshipResponse | dict:
    """Create a typed relationship between two memories.

    Use this to build a knowledge graph connecting related concepts.
    Relationships are directional: from_memory -[relation_type]-> to_memory.

    Examples:
    - "Python 3.12 features" -[supersedes]-> "Python 3.11 features"
    - "Auth implementation" -[depends_on]-> "Database schema"
    - "API endpoint details" -[elaborates]-> "API overview"
    - "Project notes" -[mentions]-> "PostgreSQL" (entity extraction)
    """
    rel_type = parse_relation_type(relation_type)
    if rel_type is None:
        return invalid_relation_type_error()

    relation = storage.link_memories(from_memory_id, to_memory_id, rel_type)
    if relation is None:
        return error_response(
            f"Failed to link memories #{from_memory_id} -> #{to_memory_id}. "
            "Check that both memories exist and aren't already linked with this type."
        )

    return relation_to_response(relation)


@mcp.tool
def unlink_memories(
    from_memory_id: Annotated[int, Field(description="Source memory ID")],
    to_memory_id: Annotated[int, Field(description="Target memory ID")],
    relation_type: Annotated[
        str | None,
        Field(description="Specific relation type to remove, or None to remove all"),
    ] = None,
) -> dict:
    """Remove relationship(s) between two memories.

    If relation_type is specified, only removes that specific relationship.
    If not specified, removes all relationships between the two memories.
    """
    rel_type = None
    if relation_type:
        rel_type = parse_relation_type(relation_type)
        if rel_type is None:
            return invalid_relation_type_error()

    count = storage.unlink_memories(from_memory_id, to_memory_id, rel_type)
    if count == 0:
        return error_response(
            f"No relationships found between #{from_memory_id} and #{to_memory_id}"
        )

    return success_response(
        f"Removed {count} relationship(s) between #{from_memory_id} and #{to_memory_id}",
        removed_count=count,
    )


@mcp.tool
def get_related_memories(
    memory_id: Annotated[int, Field(description="Memory ID to find relationships for")],
    relation_type: Annotated[
        str | None,
        Field(description="Filter by relation type"),
    ] = None,
    direction: Annotated[
        str,
        Field(
            description=(
                "Direction: 'outgoing' (from this memory), "
                "'incoming' (to this memory), or 'both' (default)"
            )
        ),
    ] = "both",
) -> list[RelatedMemoryResponse] | dict:
    """Get memories related to a given memory.

    Returns related memories along with their relationship information.
    Use this to explore the knowledge graph around a specific concept.
    """
    if direction not in ("outgoing", "incoming", "both"):
        return error_response("Invalid direction. Use: 'outgoing', 'incoming', or 'both'")

    rel_type = None
    if relation_type:
        rel_type = parse_relation_type(relation_type)
        if rel_type is None:
            return invalid_relation_type_error()

    related = storage.get_related(memory_id, rel_type, direction)

    return [
        RelatedMemoryResponse(
            memory=memory_to_response(memory),
            relationship=relation_to_response(relation),
        )
        for memory, relation in related
    ]


@mcp.tool
def relationship_stats() -> RelationshipStatsResponse:
    """Get statistics about memory relationships in the knowledge graph."""
    stats = storage.get_relationship_stats()
    return RelationshipStatsResponse(**stats)
