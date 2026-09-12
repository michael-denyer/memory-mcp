"""Session tools: end_session."""

from typing import Annotated

from pydantic import Field

from memory_mcp.helpers import parse_memory_type
from memory_mcp.responses import (
    error_response,
)
from memory_mcp.server.app import mcp, storage
from memory_mcp.storage import MemoryType


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
