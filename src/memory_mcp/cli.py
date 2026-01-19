"""CLI commands for memory-mcp.

These commands can be called from shell scripts and Claude Code hooks.
"""

import json
import sys
from pathlib import Path

import click

from memory_mcp.config import find_bootstrap_files, get_settings
from memory_mcp.storage import MemorySource, MemoryType, Storage
from memory_mcp.text_parsing import parse_content_into_chunks


@click.group()
@click.option("--json", "use_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def cli(ctx: click.Context, use_json: bool) -> None:
    """CLI commands for memory-mcp."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = use_json


@cli.command("log-output")
@click.option("-c", "--content", help="Content to log (or use stdin)")
@click.option(
    "-f", "--file", "filepath", type=click.Path(exists=True), help="Read content from file"
)
@click.pass_context
def log_output(ctx: click.Context, content: str | None, filepath: str | None) -> None:
    """Log output content for pattern mining."""
    settings = get_settings()
    use_json = ctx.obj["json"]

    if not settings.mining_enabled:
        click.echo("Mining is disabled", err=True)
        raise SystemExit(1)

    # Read content from file or stdin
    if filepath:
        content = Path(filepath).read_text(encoding="utf-8")
    elif content is None:
        content = sys.stdin.read()

    if not content.strip():
        click.echo("No content to log", err=True)
        raise SystemExit(1)

    if len(content) > settings.max_content_length:
        click.echo(
            f"Content too long ({len(content)} chars). Max: {settings.max_content_length}",
            err=True,
        )
        raise SystemExit(1)

    storage = Storage(settings)
    try:
        log_id = storage.log_output(content)
        if use_json:
            click.echo(json.dumps({"success": True, "log_id": log_id}))
        else:
            click.echo(f"Logged output (id={log_id})")
    finally:
        storage.close()


@cli.command("run-mining")
@click.option("--hours", default=24, help="Hours of logs to process")
@click.pass_context
def run_mining(ctx: click.Context, hours: int) -> None:
    """Run pattern mining on logged outputs."""
    settings = get_settings()
    use_json = ctx.obj["json"]

    if not settings.mining_enabled:
        click.echo("Mining is disabled", err=True)
        raise SystemExit(1)

    from memory_mcp.mining import run_mining as do_mining

    storage = Storage(settings)
    try:
        result = do_mining(storage, hours=hours)
        if use_json:
            click.echo(json.dumps(result))
        else:
            click.echo(f"Processed {result['outputs_processed']} outputs")
            click.echo(f"Found {result['patterns_found']} patterns")
    finally:
        storage.close()


@cli.command("seed")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "-t",
    "--type",
    "memory_type",
    default="project",
    type=click.Choice(["project", "pattern", "reference", "conversation"]),
    help="Memory type",
)
@click.option("--promote", is_flag=True, help="Promote all seeded memories to hot cache")
@click.pass_context
def seed(ctx: click.Context, file: str, memory_type: str, promote: bool) -> None:
    """Seed memories from a file (e.g., CLAUDE.md)."""
    use_json = ctx.obj["json"]
    path = Path(file).expanduser()

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        click.echo(f"Read error: {e}", err=True)
        raise SystemExit(1)

    mem_type = MemoryType(memory_type)
    settings = get_settings()
    chunks = parse_content_into_chunks(content)
    created, skipped, errors = 0, 0, []

    storage = Storage(settings)
    try:
        for chunk in chunks:
            if len(chunk) > settings.max_content_length:
                errors.append(f"Chunk too long ({len(chunk)} chars)")
                continue

            memory_id, is_new = storage.store_memory(
                content=chunk,
                memory_type=mem_type,
                source=MemorySource.MANUAL,
            )
            if is_new:
                created += 1
                if promote:
                    storage.promote_to_hot(memory_id)
            else:
                skipped += 1
    finally:
        storage.close()

    if use_json:
        click.echo(
            json.dumps(
                {
                    "memories_created": created,
                    "memories_skipped": skipped,
                    "errors": errors,
                }
            )
        )
    else:
        click.echo(f"Created {created} memories, skipped {skipped} duplicates")
        if errors:
            click.echo(f"Errors: {len(errors)}")


@cli.command("bootstrap")
@click.option(
    "-r",
    "--root",
    "root_path",
    type=click.Path(exists=True),
    default=".",
    help="Project root directory",
)
@click.option(
    "-f",
    "--files",
    multiple=True,
    help="Specific files to seed (default: auto-detect)",
)
@click.option(
    "-t",
    "--type",
    "memory_type",
    default="project",
    type=click.Choice(["project", "pattern", "reference", "conversation"]),
    help="Memory type for all content",
)
@click.option(
    "--promote/--no-promote",
    default=True,
    help="Promote to hot cache (default: yes)",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Tags to apply to all memories",
)
@click.pass_context
def bootstrap(
    ctx: click.Context,
    root_path: str,
    files: tuple[str, ...],
    memory_type: str,
    promote: bool,
    tags: tuple[str, ...],
) -> None:
    """Bootstrap hot cache from project documentation files.

    Scans for common documentation files (README.md, CLAUDE.md, etc.),
    parses them into memories, and promotes to hot cache.

    Examples:

        # Auto-detect and bootstrap from current directory
        memory-mcp-cli bootstrap

        # Bootstrap from specific project root
        memory-mcp-cli bootstrap -r /path/to/project

        # Bootstrap specific files only
        memory-mcp-cli bootstrap -f README.md -f ARCHITECTURE.md

        # Bootstrap without promoting to hot cache
        memory-mcp-cli bootstrap --no-promote

        # JSON output for scripting
        memory-mcp-cli --json bootstrap
    """
    use_json = ctx.obj["json"]
    root = Path(root_path).expanduser().resolve()

    # Determine files to process
    if files:
        file_paths = [root / f for f in files]
    else:
        file_paths = find_bootstrap_files(root)

    # Handle empty repo case
    if not file_paths:
        result = {
            "success": True,
            "files_found": 0,
            "files_processed": 0,
            "memories_created": 0,
            "memories_skipped": 0,
            "hot_cache_promoted": 0,
            "errors": [],
            "message": (
                "No documentation files found. Create README.md or CLAUDE.md to bootstrap."
            ),
        }
        if use_json:
            click.echo(json.dumps(result))
        else:
            click.echo(result["message"])
        return

    mem_type = MemoryType(memory_type)
    tag_list = list(tags) if tags else None
    settings = get_settings()

    storage = Storage(settings)
    try:
        result = storage.bootstrap_from_files(
            file_paths=file_paths,
            memory_type=mem_type,
            promote_to_hot=promote,
            tags=tag_list,
        )
    finally:
        storage.close()

    if use_json:
        click.echo(json.dumps(result))
    else:
        click.echo(result["message"])
        errors = result.get("errors")
        if isinstance(errors, list):
            for err in errors:
                click.echo(f"  Warning: {err}", err=True)


@cli.command("db-rebuild-vectors")
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="Memories to embed per batch (default 100)",
)
@click.option(
    "--clear-only",
    is_flag=True,
    help="Only clear vectors, don't re-embed",
)
@click.pass_context
def db_rebuild_vectors(ctx: click.Context, batch_size: int, clear_only: bool) -> None:
    """Rebuild all memory vectors with the current embedding model.

    Use this to fix dimension mismatch errors or when switching models.
    Memories are preserved - only the vector embeddings are rebuilt.

    Examples:

        # Rebuild all vectors
        memory-mcp-cli db-rebuild-vectors

        # Just clear vectors (faster, but recall won't work)
        memory-mcp-cli db-rebuild-vectors --clear-only

        # JSON output for scripting
        memory-mcp-cli --json db-rebuild-vectors
    """
    use_json = ctx.obj["json"]
    settings = get_settings()

    storage = Storage(settings)
    try:
        if clear_only:
            clear_result = storage.clear_vectors()
            result = {
                **clear_result,
                "memories_total": 0,
                "memories_embedded": 0,
                "memories_failed": 0,
            }
            msg = f"Cleared {result['vectors_cleared']} vectors"
        else:
            result = storage.rebuild_vectors(batch_size=batch_size)
            msg = (
                f"Rebuilt {result['memories_embedded']}/{result['memories_total']} "
                f"vectors with {result['new_model']} (dim={result['new_dimension']})"
            )

        if use_json:
            click.echo(json.dumps({"success": True, **result}))
        else:
            click.echo(msg)
            if result.get("memories_failed", 0) > 0:
                click.echo(f"  Failed: {result['memories_failed']}", err=True)
    except Exception as e:
        if use_json:
            click.echo(json.dumps({"success": False, "error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    finally:
        storage.close()


def main() -> int:
    """Main CLI entry point."""
    try:
        cli(standalone_mode=False)
        return 0
    except click.ClickException as e:
        e.show()
        return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
