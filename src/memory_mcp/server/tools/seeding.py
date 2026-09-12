"""Seeding tools: bootstrap_project."""

from typing import Annotated

from pydantic import Field

from memory_mcp.config import find_bootstrap_files
from memory_mcp.helpers import parse_memory_type
from memory_mcp.responses import BootstrapResponse
from memory_mcp.server.app import (
    mcp,
    storage,
)
from memory_mcp.storage import MemoryType


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
