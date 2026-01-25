"""Analysis module for handoff parsing and hotspot detection.

Provides tools for analyzing structured handoff memories to detect
code hotspots, cluster similar issues, and identify patterns.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any


class HandoffType(str, Enum):
    """Types of handoff memories."""

    GOTCHA = "GOTCHA"
    BUG = "BUG"
    DECISION = "DECISION"
    TASK = "TASK"
    CONSTRAINT = "CONSTRAINT"
    PATTERN = "PATTERN"


@dataclass
class Handoff:
    """Parsed handoff memory with typed fields."""

    type: HandoffType

    # GOTCHA fields
    problem: str | None = None
    fix: str | None = None
    file: str | None = None

    # BUG fields (reuses fix and file from GOTCHA)
    symptom: str | None = None
    root_cause: str | None = None
    severity: str | None = None

    # DECISION fields
    decision: str | None = None
    reason: str | None = None
    alternative: str | None = None

    # TASK fields
    task: str | None = None
    status: str | None = None
    next_action: str | None = None
    blocker: str | None = None

    # CONSTRAINT fields
    constraint: str | None = None
    scope: str | None = None
    workaround: str | None = None

    # PATTERN fields
    pattern: str | None = None
    use: str | None = None
    example: str | None = None

    # Metadata
    raw_content: str | None = None
    memory_id: int | None = None
    created_at: str | None = None


@dataclass
class Hotspot:
    """A file identified as a code hotspot."""

    file: str
    total_mentions: int
    gotcha_count: int = 0
    bug_count: int = 0
    score: float = 0.0
    issues: list[Handoff] = field(default_factory=list)


@dataclass
class Cluster:
    """A cluster of similar handoffs."""

    representative: str
    handoffs: list[Handoff] = field(default_factory=list)
    count: int = 0


# Field mappings for each handoff type
FIELD_MAPPINGS: dict[HandoffType, dict[str, str]] = {
    HandoffType.GOTCHA: {
        "GOTCHA": "problem",
        "FIX": "fix",
        "FILE": "file",
    },
    HandoffType.BUG: {
        "BUG": "symptom",
        "ROOT_CAUSE": "root_cause",
        "FIX": "fix",
        "FILE": "file",
        "SEVERITY": "severity",
    },
    HandoffType.DECISION: {
        "DECISION": "decision",
        "REASON": "reason",
        "ALTERNATIVE": "alternative",
    },
    HandoffType.TASK: {
        "TASK": "task",
        "STATUS": "status",
        "NEXT": "next_action",
        "BLOCKER": "blocker",
    },
    HandoffType.CONSTRAINT: {
        "CONSTRAINT": "constraint",
        "SCOPE": "scope",
        "WORKAROUND": "workaround",
    },
    HandoffType.PATTERN: {
        "PATTERN": "pattern",
        "USE": "use",
        "EXAMPLE": "example",
    },
}


def parse_handoff(content: str) -> Handoff | None:
    """Parse pipe-delimited handoff content into a Handoff object.

    Args:
        content: Raw handoff content like "GOTCHA: problem | FIX: solution | FILE: path"

    Returns:
        Parsed Handoff object or None if content cannot be parsed
    """
    if not content or not content.strip():
        return None

    content = content.strip()

    # Determine the handoff type from the first field
    handoff_type = None
    for htype in HandoffType:
        if content.startswith(f"{htype.value}:"):
            handoff_type = htype
            break

    if handoff_type is None:
        return None

    # Split by pipe and parse key-value pairs
    parts = content.split("|")
    fields: dict[str, Any] = {"type": handoff_type, "raw_content": content}
    field_mapping = FIELD_MAPPINGS.get(handoff_type, {})

    for part in parts:
        part = part.strip()
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        key = key.strip().upper()
        value = value.strip()

        # Map the key to the dataclass field name
        if key in field_mapping:
            field_name = field_mapping[key]
            fields[field_name] = value

    return Handoff(**fields)


def extract_files(handoffs: list[Handoff]) -> Counter[str]:
    """Extract and count file mentions across handoffs.

    Args:
        handoffs: List of parsed handoffs

    Returns:
        Counter of file paths to mention counts
    """
    file_counts: Counter[str] = Counter()

    for handoff in handoffs:
        if handoff.file:
            file_counts[handoff.file] += 1

    return file_counts


def detect_hotspots(
    handoffs: list[Handoff],
    min_mentions: int = 2,
    bug_weight: float = 2.0,
    gotcha_weight: float = 1.0,
) -> list[Hotspot]:
    """Detect files with multiple issues (hotspots).

    Hotspots are files that appear frequently in GOTCHAs and BUGs,
    indicating problematic or brittle code.

    Args:
        handoffs: List of parsed handoffs
        min_mentions: Minimum mentions required to be considered a hotspot
        bug_weight: Weight for BUG mentions in score calculation
        gotcha_weight: Weight for GOTCHA mentions in score calculation

    Returns:
        List of Hotspot objects sorted by score (descending)
    """
    if not handoffs:
        return []

    # Group handoffs by file
    file_handoffs: dict[str, list[Handoff]] = {}
    for handoff in handoffs:
        if handoff.file:
            if handoff.file not in file_handoffs:
                file_handoffs[handoff.file] = []
            file_handoffs[handoff.file].append(handoff)

    # Build hotspots
    hotspots = []
    for file_path, file_issues in file_handoffs.items():
        total = len(file_issues)
        if total < min_mentions:
            continue

        gotcha_count = sum(1 for h in file_issues if h.type == HandoffType.GOTCHA)
        bug_count = sum(1 for h in file_issues if h.type == HandoffType.BUG)

        # Score: weighted sum of issues
        score = (gotcha_count * gotcha_weight) + (bug_count * bug_weight)

        hotspots.append(
            Hotspot(
                file=file_path,
                total_mentions=total,
                gotcha_count=gotcha_count,
                bug_count=bug_count,
                score=score,
                issues=file_issues,
            )
        )

    # Sort by score descending
    hotspots.sort(key=lambda h: h.score, reverse=True)
    return hotspots


def _similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def cluster_by_field(
    handoffs: list[Handoff],
    field_name: str,
    similarity_threshold: float = 0.6,
) -> list[Cluster]:
    """Cluster handoffs by similarity of a specific field.

    Uses string similarity to group handoffs with similar field values.
    Useful for finding common root causes or recurring problems.

    Args:
        handoffs: List of parsed handoffs
        field_name: Name of the field to cluster by (e.g., "root_cause", "problem")
        similarity_threshold: Minimum similarity ratio (0-1) to consider a match

    Returns:
        List of Cluster objects
    """
    if not handoffs:
        return []

    # Filter handoffs that have the specified field
    valid_handoffs = []
    for h in handoffs:
        value = getattr(h, field_name, None)
        if value:
            valid_handoffs.append((h, value))

    if not valid_handoffs:
        return []

    # Simple greedy clustering
    clusters: list[Cluster] = []
    used = set()

    for i, (handoff, value) in enumerate(valid_handoffs):
        if i in used:
            continue

        # Start a new cluster
        cluster = Cluster(representative=value, handoffs=[handoff], count=1)
        used.add(i)

        # Find similar items
        for j, (other_handoff, other_value) in enumerate(valid_handoffs):
            if j in used:
                continue

            if _similarity(value, other_value) >= similarity_threshold:
                cluster.handoffs.append(other_handoff)
                cluster.count += 1
                used.add(j)

        clusters.append(cluster)

    return clusters
