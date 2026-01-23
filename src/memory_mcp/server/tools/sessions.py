"""Session tools: get_sessions, get_session, get_session_memories, etc."""

from typing import Annotated

from pydantic import Field

from memory_mcp.helpers import parse_memory_type
from memory_mcp.responses import (
    CrossSessionPatternResponse,
    MemoryResponse,
    SessionResponse,
    error_response,
    memory_to_response,
    session_to_response,
    success_response,
)
from memory_mcp.server.app import mcp, storage
from memory_mcp.storage import MemoryType


@mcp.tool
def get_sessions(
    limit: Annotated[int, Field(description="Maximum sessions to return")] = 20,
    project_path: Annotated[
        str | None, Field(description="Filter to sessions from this project path")
    ] = None,
) -> list[SessionResponse]:
    """Get recent conversation sessions.

    Sessions track which conversations memories originated from.
    Use this to see conversation history and navigate to specific sessions.
    """
    sessions = storage.get_sessions(limit=limit, project_path=project_path)
    return [session_to_response(s) for s in sessions]


@mcp.tool
def get_session(
    session_id: Annotated[str, Field(description="Session ID to retrieve")],
) -> SessionResponse | dict:
    """Get details for a specific session."""
    session = storage.get_session(session_id)
    if session is None:
        return error_response(f"Session not found: {session_id}")
    return session_to_response(session)


@mcp.tool
def get_session_memories(
    session_id: Annotated[str, Field(description="Session ID to get memories from")],
    limit: Annotated[int, Field(description="Maximum memories to return")] = 100,
) -> list[MemoryResponse] | dict:
    """Get all memories from a specific conversation session.

    Use this to explore what was learned during a particular conversation.
    """
    session = storage.get_session(session_id)
    if session is None:
        return error_response(f"Session not found: {session_id}")

    memories = storage.get_session_memories(session_id, limit=limit)
    return [memory_to_response(m) for m in memories]


@mcp.tool
def cross_session_patterns(
    min_sessions: Annotated[
        int, Field(description="Minimum sessions a pattern must appear in")
    ] = 2,
) -> list[CrossSessionPatternResponse]:
    """Find content patterns appearing across multiple conversation sessions.

    Useful for identifying frequently-discussed topics that might warrant
    promotion to hot cache. Patterns appearing in many sessions are likely
    important project knowledge.

    Returns patterns sorted by session count and total accesses.
    """
    patterns = storage.get_cross_session_patterns(min_sessions=min_sessions)
    return [CrossSessionPatternResponse(**p) for p in patterns]


@mcp.tool
def set_session_topic(
    session_id: Annotated[str, Field(description="Session ID to update")],
    topic: Annotated[str, Field(description="Topic description for the session")],
) -> dict:
    """Set or update the topic for a conversation session.

    Topics help identify what conversations were about when reviewing
    session history. Can be auto-detected or manually set.
    """
    if storage.update_session_topic(session_id, topic):
        return success_response(f"Updated topic for session {session_id}", topic=topic)
    return error_response(f"Session not found: {session_id}")


@mcp.tool
def summarize_session(
    session_id: Annotated[str, Field(description="Session ID to summarize")],
) -> dict:
    """Summarize a session's key decisions, insights, and action items.

    Groups session memories by semantic category to extract structured knowledge:
    - Decisions: Choices made and their rationale
    - Insights: Lessons learned, antipatterns, landmines, constraints
    - Action Items: TODOs, bugs, tasks to complete
    - Context: Background info, conventions, preferences, architecture

    Use this before end_session() to review what was captured, or anytime
    to get a structured view of a conversation's key takeaways.
    """
    return storage.summarize_session(session_id)


@mcp.tool
def end_session(
    session_id: Annotated[str, Field(description="Session ID to end")],
    promote_top: Annotated[
        bool,
        Field(description="Promote top episodic memories to long-term storage (default: true)"),
    ] = True,
    promote_type: Annotated[
        str | None,
        Field(
            description=(
                "Memory type for promoted memories: 'project' or 'pattern' (default: project)"
            )
        ),
    ] = None,
) -> dict:
    """End a session and consolidate episodic memories.

    Mirrors human memory consolidation: short-term (episodic) memories that
    prove valuable get promoted to long-term (semantic) storage.

    Top episodic memories are selected by salience score (combining importance,
    trust, access count, and recency). Only memories above the threshold are
    promoted.

    Use this at the end of a conversation to preserve valuable learnings.
    """
    ptype = MemoryType.PROJECT
    if promote_type:
        ptype = parse_memory_type(promote_type)
        if ptype is None:
            return error_response(
                f"Invalid promote_type '{promote_type}'. "
                "Use: project, pattern, reference, conversation"
            )

    result = storage.end_session(
        session_id=session_id,
        promote_top=promote_top,
        promote_type=ptype,
    )

    if not result.get("success"):
        return error_response(result.get("error", "Unknown error"))

    return result
