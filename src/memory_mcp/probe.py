"""Learning-loop round-trip probe.

Runs the real logging -> mining -> storage pipeline end-to-end with a
disposable sentinel, then hard-deletes every artifact it created. A probe
failure means a stage of the pipeline is actually broken (a "wiring bug"),
not that data is stale - `run_mining` with `record_run=False` never writes
a `mining_runs` row, so probing never resets the staleness clock used by
`storage.get_loop_health`.

This module is intentionally CLI-free; a future task wires `run_probe` into
the hook-check CLI command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from memory_mcp.mining import run_mining
from memory_mcp.storage import Storage
from memory_mcp.storage.mining_runs import PROBE_SESSION_ID


@dataclass
class ProbeResult:
    """Outcome of a single round-trip probe run.

    Attributes:
        ok: True if every stage completed and cleanup left zero residue.
        stage: Empty string on success; otherwise the stage that failed -
            one of "log", "mine", "assert", "cleanup".
        error: `f"{type(e).__name__}: {e}"` for the exception that caused
            the failure, or None on success.
        details: Diagnostic counts. On success, cleanup counts under the
            keys "memories_deleted", "patterns_deleted", "outputs_deleted".
    """

    ok: bool
    stage: str
    error: str | None = None
    details: dict = field(default_factory=dict)


def _cleanup(storage: Storage, nonce: str) -> dict:
    """Delete every probe artifact matching `nonce` and verify zero residue.

    Memories are removed via `storage.delete_memory` (not raw SQL) so the
    vector and tag cascades run. `mined_patterns` and `output_log` rows have
    no dependents, so they're removed directly.

    Args:
        storage: Storage instance to clean up.
        nonce: The probe's content nonce, used to scope pattern/memory
            deletes via `LIKE '%' || nonce || '%'`.

    Returns:
        Dict with keys `memories_deleted`, `patterns_deleted`,
        `outputs_deleted` giving the counts removed.

    Raises:
        AssertionError: If residue remains after cleanup.
    """
    like_nonce = f"%{nonce}%"

    with storage._connection() as conn:
        memory_ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM memories WHERE content LIKE ?", (like_nonce,)
            ).fetchall()
        ]
    memories_deleted = sum(1 for mid in memory_ids if storage.delete_memory(mid))

    with storage.transaction() as conn:
        patterns_deleted = conn.execute(
            "DELETE FROM mined_patterns WHERE pattern LIKE ?", (like_nonce,)
        ).rowcount
        outputs_deleted = conn.execute(
            "DELETE FROM output_log WHERE session_id = ?", (PROBE_SESSION_ID,)
        ).rowcount

    with storage._connection() as conn:
        residue = (
            conn.execute(
                "SELECT COUNT(*) FROM output_log WHERE session_id = ?", (PROBE_SESSION_ID,)
            ).fetchone()[0],
            conn.execute(
                "SELECT COUNT(*) FROM mined_patterns WHERE pattern LIKE ?", (like_nonce,)
            ).fetchone()[0],
            conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content LIKE ?", (like_nonce,)
            ).fetchone()[0],
        )
    assert residue == (0, 0, 0), f"probe cleanup left residue: {residue}"

    return {
        "memories_deleted": memories_deleted,
        "patterns_deleted": patterns_deleted,
        "outputs_deleted": outputs_deleted,
    }


def run_probe(storage: Storage) -> ProbeResult:
    """Run the learning-loop round trip and report whether wiring is intact.

    Logs a disposable, import-shaped sentinel, mines it with the real
    pipeline, asserts it was captured as a mined pattern, then deletes every
    artifact it created. A clean pre-sweep runs first so leftovers from a
    prior failed probe never accumulate or get mistaken for real data.

    Args:
        storage: Storage instance to probe.

    Returns:
        A `ProbeResult`. On success, `ok=True`, `stage=""`, and `details`
        holds the cleanup counts. On failure, `ok=False`, `stage` names the
        failing stage ("log", "mine", "assert", or "cleanup" - pre-sweep
        failures also report as "cleanup"), and `error` describes the
        exception.
    """
    nonce = uuid4().hex[:8]

    try:
        _cleanup(storage, nonce)
    except Exception as e:  # noqa: BLE001 - reported via ProbeResult, not raised
        return ProbeResult(ok=False, stage="cleanup", error=f"{type(e).__name__}: {e}")

    try:
        # Constraint: the sentinel must stay >= mining_min_pattern_length
        # (default 30) characters, or the extractor's length filter in
        # run_mining drops the pattern before mined_patterns and the assert
        # stage can never pass. This sentinel is 37 chars (8-char nonce). If
        # a user raises the threshold past the sentinel length, the probe
        # fails at the assert stage and hook-check reports it.
        sentinel = f"import memory_probe_sentinel_{nonce}"
        storage.log_output(sentinel, session_id=PROBE_SESSION_ID)
    except Exception as e:  # noqa: BLE001
        return ProbeResult(ok=False, stage="log", error=f"{type(e).__name__}: {e}")

    try:
        run_mining(storage, hours=1, session_id=PROBE_SESSION_ID, record_run=False)
    except Exception as e:  # noqa: BLE001
        return ProbeResult(ok=False, stage="mine", error=f"{type(e).__name__}: {e}")

    try:
        with storage._connection() as conn:
            pattern_row = conn.execute(
                "SELECT COUNT(*) FROM mined_patterns WHERE pattern LIKE ?", (f"%{nonce}%",)
            ).fetchone()
        assert pattern_row[0] > 0, f"no mined_patterns row for nonce {nonce!r}"
    except Exception as e:  # noqa: BLE001
        return ProbeResult(ok=False, stage="assert", error=f"{type(e).__name__}: {e}")

    try:
        details = _cleanup(storage, nonce)
    except Exception as e:  # noqa: BLE001
        return ProbeResult(ok=False, stage="cleanup", error=f"{type(e).__name__}: {e}")

    return ProbeResult(ok=True, stage="", details=details)
