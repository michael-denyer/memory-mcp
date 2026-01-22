"""Seeding tools: seed_from_text, seed_from_file, bootstrap_project."""

from typing import Annotated

from pydantic import Field

from memory_mcp.config import find_bootstrap_files
from memory_mcp.helpers import parse_memory_type
from memory_mcp.responses import BootstrapResponse, SeedResult
from memory_mcp.server.app import get_auto_project_id, log, mcp, settings, storage
from memory_mcp.storage import MemorySource, MemoryType
from memory_mcp.text_parsing import parse_content_into_chunks


def _seed_from_text_impl(
    content: str,
    memory_type: str = "project",
    promote_to_hot: bool = False,
) -> SeedResult:
    """Implementation for seeding memories from text content."""
    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return SeedResult(memories_created=0, memories_skipped=0, errors=["Invalid memory_type"])

    chunks = parse_content_into_chunks(content)
    created, skipped, errors = 0, 0, []

    # Use project_id for project-aware memory
    project_id = get_auto_project_id()

    for chunk in chunks:
        if len(chunk) > settings.max_content_length:
            errors.append(f"Chunk too long ({len(chunk)} chars), skipped")
            continue

        memory_id, is_new = storage.store_memory(
            content=chunk,
            memory_type=mem_type,
            source=MemorySource.MANUAL,
            project_id=project_id,
        )
        if is_new:
            created += 1
            if promote_to_hot:
                storage.promote_to_hot(memory_id)
        else:
            skipped += 1

    log.info("seed_from_text: created={} skipped={} errors={}", created, skipped, len(errors))
    return SeedResult(memories_created=created, memories_skipped=skipped, errors=errors)


@mcp.tool
def seed_from_text(
    content: Annotated[str, Field(description="Text content to parse and seed memories from")],
    memory_type: Annotated[
        str, Field(description="Memory type for all extracted items")
    ] = "project",
    promote_to_hot: Annotated[bool, Field(description="Promote all to hot cache")] = False,
) -> SeedResult:
    """Seed memories from text content.

    Parses the content into individual memories (one per paragraph or list item)
    and stores them. Useful for initial setup or bulk import.

    Content is split on:
    - Double newlines (paragraphs)
    - Lines starting with '- ' or '* ' (list items)
    - Lines starting with numbers like '1. ' (numbered lists)
    """
    return _seed_from_text_impl(content, memory_type, promote_to_hot)


@mcp.tool
def seed_from_file(
    file_path: Annotated[str, Field(description="Path to file to import")],
    memory_type: Annotated[str, Field(description="Memory type for content")] = "project",
    promote_to_hot: Annotated[bool, Field(description="Promote to hot cache")] = False,
) -> SeedResult:
    """Seed memories from a file.

    Reads the file and extracts memories based on content structure.
    Supports markdown files (splits on headers and lists) and plain text.

    Common use: Import from project CLAUDE.md or documentation files.
    """
    from pathlib import Path

    path = Path(file_path).expanduser()
    if not path.exists():
        return SeedResult(
            memories_created=0, memories_skipped=0, errors=[f"File not found: {file_path}"]
        )

    if not path.is_file():
        return SeedResult(
            memories_created=0, memories_skipped=0, errors=[f"Not a file: {file_path}"]
        )

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return SeedResult(memories_created=0, memories_skipped=0, errors=[f"Read error: {e}"])

    return _seed_from_text_impl(
        content=content, memory_type=memory_type, promote_to_hot=promote_to_hot
    )


def _empty_bootstrap_response(
    message: str,
    errors: list[str] | None = None,
    success: bool = True,
) -> BootstrapResponse:
    """Create a BootstrapResponse for early-exit cases (no files processed)."""
    return BootstrapResponse(
        success=success,
        message=message,
        errors=errors or [],
    )


@mcp.tool
def bootstrap_project(
    root_path: Annotated[
        str,
        Field(description="Project root directory (default: current directory)"),
    ] = ".",
    file_patterns: Annotated[
        list[str] | None,
        Field(
            description=(
                "Specific files to seed. If not provided, auto-detects: "
                "CLAUDE.md, README.md, CONTRIBUTING.md, ARCHITECTURE.md"
            )
        ),
    ] = None,
    promote_to_hot: Annotated[
        bool,
        Field(description="Promote all bootstrapped memories to hot cache"),
    ] = True,
    memory_type: Annotated[
        str,
        Field(description="Memory type for all content"),
    ] = "project",
    tags: Annotated[
        list[str] | None,
        Field(description="Tags to apply to all memories"),
    ] = None,
) -> BootstrapResponse:
    """Bootstrap hot cache from project documentation files.

    Scans for common project documentation files (README.md, CLAUDE.md, etc.),
    parses them into memories, and optionally promotes to hot cache.

    This is ideal for quickly populating the hot cache when starting work
    on a new codebase.

    Edge cases handled gracefully:
    - Empty repo: Returns success with files_found=0 and helpful message
    - No markdown files: Returns success with message
    - File read errors: Logged in errors list, continues with other files
    - Empty files: Skipped silently
    - Binary files: Skipped with warning
    - All content already exists: Returns memories_skipped count
    """
    from pathlib import Path

    root = Path(root_path).expanduser().resolve()

    if not root.exists():
        return _empty_bootstrap_response(
            "Root path does not exist.",
            errors=[f"Root path not found: {root_path}"],
        )

    if not root.is_dir():
        return _empty_bootstrap_response(
            "Root path is not a directory.",
            errors=[f"Not a directory: {root_path}"],
        )

    # Determine files to process
    if file_patterns:
        file_paths = [root / f for f in file_patterns]
    else:
        file_paths = find_bootstrap_files(root)

    if not file_paths:
        return _empty_bootstrap_response(
            "No documentation files found. Create README.md or CLAUDE.md to bootstrap."
        )

    # Validate memory type
    mem_type = parse_memory_type(memory_type)
    if mem_type is None:
        return _empty_bootstrap_response(
            "Invalid memory type specified.",
            errors=[f"Invalid memory_type. Use: {[t.value for t in MemoryType]}"],
            success=False,
        )

    result = storage.bootstrap_from_files(
        file_paths=file_paths,
        memory_type=mem_type,
        promote_to_hot=promote_to_hot,
        tags=tags,
    )

    return BootstrapResponse(**result)
