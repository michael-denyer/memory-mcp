"""Pattern mining from output logs."""

import re
from dataclasses import dataclass
from enum import Enum

from memory_mcp.embeddings import content_hash
from memory_mcp.storage import Storage


class PatternType(str, Enum):
    """Types of patterns that can be mined."""

    IMPORT = "import"  # Import statements
    FACT = "fact"  # "This project uses X" statements
    COMMAND = "command"  # Shell commands
    CODE = "code"  # Code snippets


# Common CLI tool prefixes for command extraction
COMMAND_PREFIXES = (
    "npm",
    "yarn",
    "pnpm",
    "uv",
    "pip",
    "python",
    "node",
    "git",
    "docker",
    "make",
    "cargo",
    "go",
)


@dataclass
class ExtractedPattern:
    """A pattern extracted from output."""

    pattern: str
    pattern_type: PatternType


# ========== Pattern Extractors ==========


def extract_imports(text: str) -> list[ExtractedPattern]:
    """Extract Python import statements."""
    patterns = []

    # Python imports: import X, from X import Y
    import_re = re.compile(
        r"^(?:from\s+[\w.]+\s+import\s+[\w,\s*]+|import\s+[\w,\s.]+)$",
        re.MULTILINE,
    )

    for match in import_re.findall(text):
        # Normalize whitespace
        normalized = " ".join(match.split())
        if len(normalized) > 10:  # Skip trivial imports
            patterns.append(ExtractedPattern(normalized, PatternType.IMPORT))

    return patterns


def extract_facts(text: str) -> list[ExtractedPattern]:
    """Extract factual statements about the project."""
    patterns = []

    # Common fact patterns
    fact_patterns = [
        r"[Tt]his project uses\s+[\w\s,]+",
        r"[Ww]e use\s+[\w\s,]+(?:for|to)\s+[\w\s,]+",
        r"[Tt]he (?:API|database|server|client) (?:is|uses|runs)\s+[\w\s,]+",
        r"[Aa]uthentication (?:uses|is handled by)\s+[\w\s,]+",
        r"[Tt]ests (?:use|are run with)\s+[\w\s,]+",
    ]

    for pattern in fact_patterns:
        fact_re = re.compile(pattern)
        for match in fact_re.findall(text):
            normalized = match.strip()
            if 10 < len(normalized) < 200:  # Reasonable length
                patterns.append(ExtractedPattern(normalized, PatternType.FACT))

    return patterns


def extract_commands(text: str) -> list[ExtractedPattern]:
    """Extract shell commands."""
    patterns = []

    # Common command patterns
    command_re = re.compile(
        r"(?:^|\n)[$>]\s*(.+?)(?:\n|$)|"  # $ or > prompts
        r"`([^`]+)`|"  # Backtick commands
        r"(?:run|execute|use):\s*`([^`]+)`",  # "run: `command`"
        re.MULTILINE,
    )

    for match in command_re.findall(text):
        cmd = next((m for m in match if m), None)
        if not cmd:
            continue

        normalized = cmd.strip()
        is_known_command = normalized.startswith(COMMAND_PREFIXES)
        has_valid_length = 5 < len(normalized) < 200

        if is_known_command and has_valid_length:
            patterns.append(ExtractedPattern(normalized, PatternType.COMMAND))

    return patterns


def extract_code_patterns(text: str) -> list[ExtractedPattern]:
    """Extract notable code patterns."""
    patterns = []

    # Function definitions
    func_re = re.compile(
        r"(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->[\w\[\],\s|]+)?:",
        re.MULTILINE,
    )

    for match in func_re.findall(text):
        # Get the full line for context
        full_pattern = f"def {match}(...)"
        if not match.startswith("_"):  # Skip private functions
            patterns.append(ExtractedPattern(full_pattern, PatternType.CODE))

    # Class definitions
    class_re = re.compile(r"class\s+(\w+)\s*(?:\([^)]*\))?:", re.MULTILINE)
    for match in class_re.findall(text):
        patterns.append(ExtractedPattern(f"class {match}", PatternType.CODE))

    return patterns


# ========== Main Mining Function ==========


PATTERN_EXTRACTORS = [
    extract_imports,
    extract_facts,
    extract_commands,
    extract_code_patterns,
]


def extract_patterns(text: str) -> list[ExtractedPattern]:
    """Extract all patterns from text."""
    all_patterns = [pattern for extractor in PATTERN_EXTRACTORS for pattern in extractor(text)]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for p in all_patterns:
        if p.pattern not in seen:
            seen.add(p.pattern)
            unique.append(p)

    return unique


def run_mining(storage: Storage, hours: int = 24) -> dict:
    """Run pattern mining on recent outputs.

    Returns statistics about patterns found and updated.
    """
    outputs = storage.get_recent_outputs(hours=hours)

    total_patterns = 0
    new_patterns = 0
    updated_patterns = 0

    for _, content, _ in outputs:
        patterns = extract_patterns(content)
        total_patterns += len(patterns)

        for pattern in patterns:
            hash_val = content_hash(pattern.pattern)
            is_existing = storage.mined_pattern_exists(hash_val)

            if is_existing:
                updated_patterns += 1
            else:
                new_patterns += 1

            storage.upsert_mined_pattern(pattern.pattern, pattern.pattern_type.value)

    return {
        "outputs_processed": len(outputs),
        "patterns_found": total_patterns,
        "new_patterns": new_patterns,
        "updated_patterns": updated_patterns,
    }
