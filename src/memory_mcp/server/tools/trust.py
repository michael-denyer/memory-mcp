"""Trust management tools: validate_memory, invalidate_memory, get_trust_history."""

from typing import Annotated

from pydantic import Field

from memory_mcp.responses import (
    TrustHistoryEntry,
    TrustHistoryResponse,
    TrustResponse,
    error_response,
)
from memory_mcp.server.app import mcp, storage
from memory_mcp.storage import TRUST_REASON_DEFAULTS, TrustReason

STRENGTHENING_REASONS = ["used_correctly", "explicitly_confirmed", "cross_validated"]
WEAKENING_REASONS = [
    "outdated",
    "partially_incorrect",
    "factually_wrong",
    "superseded",
    "low_utility",
]


def _parse_trust_reason(
    reason: str | None, expected_positive: bool
) -> tuple[TrustReason | None, str | None]:
    """Parse and validate a trust reason string.

    Args:
        reason: The reason string to parse
        expected_positive: True for strengthening reasons, False for weakening

    Returns:
        Tuple of (TrustReason or None, error message or None)
    """
    if not reason:
        return None, None

    try:
        trust_reason = TrustReason(reason)
        default_delta = TRUST_REASON_DEFAULTS.get(trust_reason, 0)

        if expected_positive and default_delta < 0:
            return None, f"Reason '{reason}' is a weakening reason. Use invalidate_memory instead."
        if not expected_positive and default_delta > 0:
            return (
                None,
                f"Reason '{reason}' is a strengthening reason. Use validate_memory instead.",
            )

        return trust_reason, None
    except ValueError:
        valid = STRENGTHENING_REASONS if expected_positive else WEAKENING_REASONS
        action = "strengthening" if expected_positive else "weakening"
        return None, f"Invalid reason '{reason}'. Valid {action} reasons: {valid}"


@mcp.tool
def validate_memory(
    memory_id: Annotated[int, Field(description="ID of memory to validate")],
    reason: Annotated[
        str | None,
        Field(
            description=(
                "Reason for validation: 'used_correctly' (applied and worked), "
                "'explicitly_confirmed' (user verified), "
                "'cross_validated' (multiple sources agree). "
                "If not specified, uses default boost."
            )
        ),
    ] = None,
    boost: Annotated[
        float | None,
        Field(
            description=(
                "Custom trust boost (overrides reason default). If None, uses reason's default."
            )
        ),
    ] = None,
    note: Annotated[
        str | None,
        Field(description="Optional note explaining the validation context"),
    ] = None,
) -> TrustResponse | dict:
    """Mark a memory as validated/confirmed useful.

    Increases the memory's trust score and records the reason in the trust history.
    Use this when you verify that a recalled memory is still accurate and helpful.

    Reasons and their default boosts:
    - used_correctly: +0.05 (memory was applied successfully)
    - explicitly_confirmed: +0.15 (user explicitly confirmed accuracy)
    - cross_validated: +0.20 (corroborated by multiple sources)

    Trust score is capped at 1.0.
    """
    memory = storage.get_memory(memory_id)
    if not memory:
        return error_response(f"Memory #{memory_id} not found")

    old_trust = memory.trust_score

    trust_reason, error = _parse_trust_reason(reason, expected_positive=True)
    if error:
        return error_response(error)

    if trust_reason:
        new_trust = storage.adjust_trust(memory_id, reason=trust_reason, delta=boost, note=note)
    else:
        new_trust = storage.strengthen_trust(memory_id, boost=boost or 0.1)

    if new_trust is None:
        return error_response(f"Failed to validate memory #{memory_id}")

    reason_msg = f" (reason: {reason})" if reason else ""
    return TrustResponse(
        memory_id=memory_id,
        old_trust=old_trust,
        new_trust=new_trust,
        message=f"Trust increased: {old_trust:.2f} -> {new_trust:.2f}{reason_msg}",
    )


@mcp.tool
def invalidate_memory(
    memory_id: Annotated[int, Field(description="ID of memory found to be incorrect/outdated")],
    reason: Annotated[
        str | None,
        Field(
            description=(
                "Reason for invalidation: 'outdated' (info is stale), "
                "'partially_incorrect' (some details wrong), "
                "'factually_wrong' (fundamentally incorrect), "
                "'superseded' (replaced by newer info), 'low_utility' (not useful). "
                "If not specified, uses default penalty."
            )
        ),
    ] = None,
    penalty: Annotated[
        float | None,
        Field(
            description=(
                "Custom trust penalty (overrides reason default). If None, uses reason's default."
            )
        ),
    ] = None,
    note: Annotated[
        str | None,
        Field(description="Optional note explaining the invalidation context"),
    ] = None,
) -> TrustResponse | dict:
    """Mark a memory as incorrect or outdated.

    Decreases the memory's trust score and records the reason in the trust history.
    Use this when you discover that a recalled memory contains inaccurate or outdated information.

    Reasons and their default penalties:
    - outdated: -0.10 (information is stale but was once correct)
    - partially_incorrect: -0.15 (some details are wrong)
    - factually_wrong: -0.30 (fundamentally incorrect)
    - superseded: -0.05 (replaced by newer information)
    - low_utility: -0.05 (not useful in practice)

    Trust score is floored at 0.0. Memories with very low trust will
    rank lower in recall results due to trust-weighted scoring.
    """
    memory = storage.get_memory(memory_id)
    if not memory:
        return error_response(f"Memory #{memory_id} not found")

    old_trust = memory.trust_score

    trust_reason, error = _parse_trust_reason(reason, expected_positive=False)
    if error:
        return error_response(error)

    if trust_reason:
        new_trust = storage.adjust_trust(memory_id, reason=trust_reason, delta=penalty, note=note)
    else:
        new_trust = storage.weaken_trust(memory_id, penalty=penalty or 0.1)

    if new_trust is None:
        return error_response(f"Failed to invalidate memory #{memory_id}")

    reason_msg = f" (reason: {reason})" if reason else ""
    return TrustResponse(
        memory_id=memory_id,
        old_trust=old_trust,
        new_trust=new_trust,
        message=f"Trust decreased: {old_trust:.2f} -> {new_trust:.2f}{reason_msg}",
    )


@mcp.tool
def get_trust_history(
    memory_id: Annotated[int, Field(description="ID of memory to get trust history for")],
    limit: Annotated[
        int, Field(description="Maximum number of history entries to return (default 20)")
    ] = 20,
) -> TrustHistoryResponse | dict:
    """Get the trust adjustment history for a memory.

    Shows all trust changes with reasons, timestamps, and context.
    Useful for understanding why a memory's trust score evolved.
    """
    memory = storage.get_memory(memory_id)
    if not memory:
        return error_response(f"Memory #{memory_id} not found")

    events = storage.get_trust_history(memory_id, limit=limit)

    entries = [
        TrustHistoryEntry(
            id=e.id,
            memory_id=e.memory_id,
            reason=e.reason.value,
            old_trust=e.old_trust,
            new_trust=e.new_trust,
            delta=e.delta,
            similarity=e.similarity,
            note=e.note,
            created_at=e.created_at.isoformat(),
        )
        for e in events
    ]

    return TrustHistoryResponse(
        memory_id=memory_id,
        entries=entries,
        current_trust=memory.trust_score,
        total_changes=len(entries),
    )
