"""CLI commands for memory-mcp.

These commands can be called from shell scripts and Claude Code hooks.
"""

import json
import sys
import time
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from memory_mcp.config import Settings, find_bootstrap_files, get_settings
from memory_mcp.logging import get_logger
from memory_mcp.probe import run_probe
from memory_mcp.project import get_current_project_id
from memory_mcp.storage import MemorySource, MemoryType, Storage
from memory_mcp.text_parsing import parse_content_into_chunks

console = Console()
log = get_logger("cli")

LOOP_WARNING_STAMP_TTL_SECONDS = 24 * 60 * 60


def _loop_warning_line(settings: Settings) -> str | None:
    """Build a one-line staleness/error warning for the learning loop, if due.

    Rate-limited to once per 24h via a stamp file next to the database, so
    that repeated `bootstrap` invocations (e.g. one per Claude Code session)
    don't spam the same warning. The stamp is only touched when a warning is
    actually emitted, so the first stale/erroring day always fires
    immediately once the rate limit from the prior warning has expired.

    Args:
        settings: Active configuration, used to check the feature flag and
            locate the database (and therefore the stamp file).

    Returns:
        The warning line to print, or None if the loop is healthy, warnings
        are disabled, the rate limit hasn't elapsed, or any internal error
        occurred while computing loop health. A broken warning system must
        never break session start, so all errors degrade to silence.
    """
    try:
        if not settings.loop_warnings_enabled:
            return None

        stamp_path = Path(settings.db_path).parent / "loop-warning.stamp"
        if stamp_path.exists():
            age_seconds = time.time() - stamp_path.stat().st_mtime
            if age_seconds < LOOP_WARNING_STAMP_TTL_SECONDS:
                return None

        storage = Storage(settings)
        try:
            health = storage.get_loop_health()
        finally:
            storage.close()

        state = health.get("state")
        if state == "red":
            line = "memory loop is erroring (last 3 runs failed) — run `memory-mcp-cli hook-check`"
        elif state == "amber":
            days = health.get("days_since_success")
            if days is None:
                line = "memory loop has never produced — run `memory-mcp-cli hook-check`"
            else:
                line = (
                    f"memory loop hasn't produced in {days} days — run `memory-mcp-cli hook-check`"
                )
        else:
            return None

        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.touch()
        return line
    except Exception:
        return None


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
@click.option("-p", "--project-id", help="Project ID override (default: derived from cwd)")
@click.option("-s", "--session-id", help="Session ID for provenance tracking")
@click.pass_context
def log_output(
    ctx: click.Context,
    content: str | None,
    filepath: str | None,
    project_id: str | None,
    session_id: str | None,
) -> None:
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

    # Use explicit project_id or derive from cwd
    if project_id is None and settings.project_awareness_enabled:
        project_id = get_current_project_id()

    storage = Storage(settings)
    try:
        log_id = storage.log_output(content, project_id=project_id, session_id=session_id)
        if use_json:
            click.echo(json.dumps({"success": True, "log_id": log_id}))
        else:
            click.echo(f"Logged output (id={log_id})")
    finally:
        storage.close()


@cli.command("log-response")
@click.pass_context
def log_response(ctx: click.Context) -> None:
    """Log Claude's response from hook input for pattern mining.

    This is called by Claude Code's Stop hook. It reads the hook input from stdin,
    extracts the transcript path, and logs the assistant's last response.

    The hook input JSON should contain either:
    - transcript_path: Direct path to the transcript file
    - session_id + project_path: To derive the transcript location
    """
    import subprocess

    settings = get_settings()

    if not settings.mining_enabled:
        return  # Silent exit if mining disabled

    # Read hook input from stdin
    hook_input = sys.stdin.read().strip()
    if not hook_input:
        return

    try:
        data = json.loads(hook_input)
    except json.JSONDecodeError:
        return

    # Session provenance for the logged output; mining inherits the session
    # from the source log, so losing it here breaks session linking downstream
    session_id = data.get("session_id") or data.get("sessionId")

    # Find transcript path (multiple formats supported)
    transcript_path = (
        data.get("transcript_path")
        or data.get("transcriptPath")
        or data.get("transcript", {}).get("path")  # Nested format
    )

    if not transcript_path or not Path(transcript_path).exists():
        # Try to derive from session_id
        project_path = (
            data.get("project_path")
            or data.get("projectPath")
            or data.get("cwd")
            or data.get("workspace_path")
        )

        if session_id and project_path:
            project_slug = project_path.replace("/", "-")
            candidate = Path.home() / ".claude" / "projects" / project_slug / f"{session_id}.jsonl"
            if candidate.exists():
                transcript_path = str(candidate)

    if not transcript_path or not Path(transcript_path).exists():
        return

    # Read last 200 lines of transcript (JSONL format)
    try:
        result = subprocess.run(
            ["tail", "-200", transcript_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        transcript_tail = result.stdout
    except Exception:
        return

    if not transcript_tail:
        return

    # Extract last assistant message
    last_response = None
    last_user_msg = None

    for line in reversed(transcript_tail.strip().split("\n")):
        try:
            entry = json.loads(line)
            msg = entry.get("message", {})
            role = msg.get("role")
            content = msg.get("content", [])

            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            text = "\n".join(text_parts)

            if role == "assistant" and text and last_response is None:
                last_response = text
            elif role == "user" and text and last_user_msg is None:
                last_user_msg = text[:500]  # Truncate user message

            if last_response and last_user_msg:
                break
        except json.JSONDecodeError:
            continue

    if not last_response:
        return

    # Combine user message with response for richer context
    if last_user_msg:
        content = f"USER: {last_user_msg}\n\nASSISTANT: {last_response}"
    else:
        content = last_response

    # Skip if too short
    if len(content) < 20:
        return

    # Truncate if too long
    if len(content) > settings.max_content_length:
        content = content[: settings.max_content_length]

    # Log the content
    project_id = get_current_project_id() if settings.project_awareness_enabled else None

    storage = Storage(settings)
    try:
        storage.log_output(content, project_id=project_id, session_id=session_id)

        try:
            marked = storage.mark_used_memories(last_response)
            if marked:
                log.info(f"auto-marked {marked} injected memories as used")
        except Exception as e:
            log.warning(f"auto-mark failed (non-fatal): {e}")
    finally:
        storage.close()

    # Spawn async mining (doesn't block the hook)
    try:
        mining_args = ["memory-mcp-cli", "run-mining", "--hours", "1"]
        if project_id:
            mining_args.extend(["--project-id", project_id])
        subprocess.Popen(
            mining_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process
        )
    except Exception:
        pass  # Mining is optional


@cli.command("pre-compact")
@click.option("--skip-mining", is_flag=True, help="Skip background mining")
@click.pass_context
def pre_compact(ctx: click.Context, skip_mining: bool) -> None:
    """Consolidate session memories before conversation compaction.

    Called by Claude Code's PreCompact hook. Reads hook input from stdin,
    extracts session info, and runs end_session() to promote top episodic
    memories to long-term storage.

    Also spawns background mining to extract patterns from output logs.
    Mining runs async and doesn't block compaction.

    Designed to be quiet - exits 0 even on errors to not block compaction.
    """
    import subprocess

    settings = get_settings()
    use_json = ctx.obj["json"]

    # Read hook input from stdin
    hook_input = sys.stdin.read().strip()
    if not hook_input:
        if use_json:
            click.echo(json.dumps({"success": True, "action": "skipped", "reason": "no_input"}))
        return

    try:
        data = json.loads(hook_input)
    except json.JSONDecodeError:
        if use_json:
            click.echo(json.dumps({"success": True, "action": "skipped", "reason": "invalid_json"}))
        return

    # Extract session_id from various possible field names
    session_id = (
        data.get("session_id") or data.get("sessionId") or data.get("session", {}).get("id")
    )

    if not session_id:
        if use_json:
            click.echo(
                json.dumps({"success": True, "action": "skipped", "reason": "no_session_id"})
            )
        return

    storage = Storage(settings)
    mining_started = False
    try:
        # Run end_session to promote episodic memories to long-term storage
        result = storage.end_session(
            session_id=session_id,
            promote_top=True,
            promote_type=MemoryType.PROJECT,
        )

        # Spawn background mining (async, doesn't block)
        if not skip_mining:
            try:
                subprocess.Popen(
                    ["memory-mcp-cli", "run-mining", "--hours", "24"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,  # Detach from parent process
                )
                mining_started = True
            except Exception:
                pass  # Mining is optional, don't fail if it can't start

        if use_json:
            click.echo(
                json.dumps(
                    {
                        "success": True,
                        "action": "consolidated",
                        "session_id": session_id,
                        "promoted_count": result.get("promoted_count", 0),
                        "top_memories": result.get("top_memories", []),
                        "mining_started": mining_started,
                    }
                )
            )
        else:
            promoted = result.get("promoted_count", 0)
            if promoted > 0:
                click.echo(f"Pre-compact: promoted {promoted} memories from session")
            if mining_started:
                click.echo("Pre-compact: mining started in background")
    except Exception as e:
        # Silent failure for hooks - don't block compaction
        if use_json:
            click.echo(json.dumps({"success": False, "error": str(e)}))
        # Always exit 0 to not block compaction
    finally:
        storage.close()


@cli.command("run-mining")
@click.option("--hours", default=24, help="Hours of logs to process")
@click.option("-p", "--project-id", help="Project ID override (default: derived from cwd)")
@click.pass_context
def run_mining(ctx: click.Context, hours: int, project_id: str | None) -> None:
    """Run pattern mining on logged outputs."""
    settings = get_settings()
    use_json = ctx.obj["json"]

    if not settings.mining_enabled:
        click.echo("Mining is disabled", err=True)
        raise SystemExit(1)

    from memory_mcp.mining import run_mining as do_mining

    storage = Storage(settings)
    try:
        # Use explicit project_id or derive from cwd
        if project_id is None and settings.project_awareness_enabled:
            project_id = get_current_project_id()

        result = do_mining(storage, hours=hours, project_id=project_id)
        if use_json:
            click.echo(json.dumps(result))
        else:
            console.print("[bold]Mining Results[/bold]")
            console.print(f"  Outputs processed: [cyan]{result['outputs_processed']}[/cyan]")
            console.print(f"  Patterns found: [cyan]{result['patterns_found']}[/cyan]")
            console.print(f"  New memories: [green]{result['new_memories']}[/green]")
            console.print(f"  Updated patterns: [yellow]{result['updated_patterns']}[/yellow]")
            promoted = result.get("promoted_to_hot", 0)
            if promoted > 0:
                console.print(f"  Promoted to hot: [magenta]{promoted}[/magenta]")
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

    # Get project_id if project awareness is enabled
    project_id = None
    if settings.project_awareness_enabled:
        project_id = get_current_project_id()

    storage = Storage(settings)
    # Create a session for this CLI invocation
    session_id = str(uuid.uuid4())
    storage.create_or_get_session(session_id, topic="CLI seed", project_path=str(path.parent))

    try:
        for chunk in chunks:
            if len(chunk) > settings.max_content_length:
                errors.append(f"Chunk too long ({len(chunk)} chars)")
                continue

            memory_id, is_new = storage.store_memory(
                content=chunk,
                memory_type=mem_type,
                source=MemorySource.MANUAL,
                project_id=project_id,
                session_id=session_id,
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
        console.print("[bold]Seed Results[/bold]")
        console.print(f"  Created: [green]{created}[/green] memories")
        console.print(f"  Skipped: [yellow]{skipped}[/yellow] duplicates")
        if errors:
            console.print(f"  [red]Errors: {len(errors)}[/red]")


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
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress output (for hooks)",
)
@click.pass_context
def bootstrap(
    ctx: click.Context,
    root_path: str,
    files: tuple[str, ...],
    memory_type: str,
    promote: bool,
    tags: tuple[str, ...],
    quiet: bool,
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
    # Loop staleness warning: computed first (before the empty-repo early
    # return and before the quiet gate) so the rate-limit stamp is touched
    # on every invocation that surfaces a warning, regardless of output
    # mode. Plain mode echoes it immediately, ahead of the payload, so it
    # reaches hook stdout (the injected session context) even under `-q`.
    # JSON mode never echoes it as bare text - doing so would prepend
    # unparseable text before the JSON payload, breaking `| jq` consumers -
    # instead it's folded into the payload's "loop_warning" key below.
    settings = get_settings()
    warning = _loop_warning_line(settings)
    use_json = ctx.obj["json"]
    if warning and not use_json:
        click.echo(warning)

    root = Path(root_path).expanduser().resolve()

    # Determine files to process
    if files:
        file_paths = [root / f for f in files]
    else:
        file_paths = find_bootstrap_files(root)

    # Handle empty repo case
    if not file_paths:
        message = "No documentation files found. Create README.md or CLAUDE.md to bootstrap."
        if quiet:
            return
        if use_json:
            click.echo(
                json.dumps(
                    {
                        "success": True,
                        "files_found": 0,
                        "files_processed": 0,
                        "memories_created": 0,
                        "memories_skipped": 0,
                        "hot_cache_promoted": 0,
                        "errors": [],
                        "message": message,
                        "loop_warning": warning,
                    }
                )
            )
        else:
            click.echo(message)
        return

    mem_type = MemoryType(memory_type)
    tag_list = list(tags) if tags else None

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

    if quiet:
        return

    if use_json:
        click.echo(json.dumps({**result, "loop_warning": warning}))
    else:
        console.print("[bold]Bootstrap Results[/bold]")
        console.print(f"  Files processed: [cyan]{result.get('files_processed', 0)}[/cyan]")
        console.print(f"  Memories created: [green]{result.get('memories_created', 0)}[/green]")
        console.print(f"  Memories skipped: [yellow]{result.get('memories_skipped', 0)}[/yellow]")
        if promote:
            console.print(
                f"  Hot cache promoted: [magenta]{result.get('hot_cache_promoted', 0)}[/magenta]"
            )
        errors = result.get("errors")
        if isinstance(errors, list) and errors:
            console.print(f"  [red]Warnings: {len(errors)}[/red]")
            for err in errors:
                console.print(f"    [dim]{err}[/dim]")


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
        else:
            result = storage.rebuild_vectors(batch_size=batch_size)

        if use_json:
            click.echo(json.dumps({"success": True, **result}))
        else:
            console.print("[bold]Vector Rebuild Results[/bold]")
            console.print(f"  Vectors cleared: [yellow]{result['vectors_cleared']}[/yellow]")
            if not clear_only:
                embedded = result["memories_embedded"]
                total = result["memories_total"]
                console.print(f"  Memories embedded: [green]{embedded}[/green]/{total}")
                failed = result.get("memories_failed", 0)
                if failed > 0:
                    console.print(f"  [red]Failed: {failed}[/red]")
            console.print(f"  New model: [cyan]{result['new_model']}[/cyan]")
            console.print(f"  New dimension: [cyan]{result['new_dimension']}[/cyan]")
    except Exception as e:
        if use_json:
            click.echo(json.dumps({"success": False, "error": str(e)}))
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    finally:
        storage.close()


def _display_consolidation_preview(result: dict) -> None:
    """Display dry-run consolidation preview."""
    console.print("[bold]Consolidation Preview (dry run)[/bold]")
    clusters = result.get("clusters", [])

    if not clusters:
        console.print("[dim]No clusters found - nothing to consolidate[/dim]")
        return

    console.print(f"  Clusters found: [cyan]{result.get('cluster_count', 0)}[/cyan]")
    console.print(
        f"  Memories in clusters: [cyan]{result.get('total_memories_in_clusters', 0)}[/cyan]"
    )
    console.print(f"  Would delete: [yellow]{result.get('memories_to_delete', 0)}[/yellow]")
    console.print(f"  Space savings: [green]{result.get('space_savings_pct', 0)}%[/green]")

    console.print("\n[bold]Clusters:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rep. ID", width=8)
    table.add_column("Members", width=8)
    table.add_column("Similarity", width=10)
    table.add_column("Access Count", width=12)

    max_display = 10
    for cluster in clusters[:max_display]:
        table.add_row(
            str(cluster.get("representative_id", "")),
            str(cluster.get("member_count", "")),
            f"{cluster.get('avg_similarity', 0):.3f}",
            str(cluster.get("total_access_count", "")),
        )
    console.print(table)

    remaining = len(clusters) - max_display
    if remaining > 0:
        console.print(f"  [dim]... and {remaining} more clusters[/dim]")

    console.print("\n[dim]Run without --dry-run to apply changes[/dim]")


def _display_consolidation_results(result: dict) -> None:
    """Display actual consolidation results."""
    console.print("[bold]Consolidation Results[/bold]")
    console.print(f"  Clusters processed: [cyan]{result.get('clusters_processed', 0)}[/cyan]")
    console.print(f"  Memories deleted: [yellow]{result.get('memories_deleted', 0)}[/yellow]")

    errors = result.get("errors", [])
    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")
        for err in errors:
            console.print(f"    [dim]{err}[/dim]")


@cli.command("consolidate")
@click.option(
    "-t",
    "--type",
    "memory_type",
    default=None,
    type=click.Choice(["project", "pattern", "reference", "conversation"]),
    help="Only consolidate memories of this type",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview consolidation without making changes",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Similarity threshold for clustering (default: 0.85)",
)
@click.pass_context
def consolidate(
    ctx: click.Context,
    memory_type: str | None,
    dry_run: bool,
    threshold: float | None,
) -> None:
    """Consolidate similar memories to reduce redundancy.

    Finds clusters of semantically similar memories and merges them,
    keeping the best representative from each cluster.

    Examples:

        # Preview what would be consolidated (dry run)
        memory-mcp-cli consolidate --dry-run

        # Run consolidation
        memory-mcp-cli consolidate

        # Only consolidate pattern memories
        memory-mcp-cli consolidate -t pattern

        # Use stricter similarity threshold
        memory-mcp-cli consolidate --threshold 0.9

        # JSON output for scripting
        memory-mcp-cli --json consolidate --dry-run
    """
    use_json = ctx.obj["json"]
    mem_type = MemoryType(memory_type) if memory_type else None
    settings = get_settings()

    if threshold is not None:
        settings.consolidation_threshold = threshold

    storage = Storage(settings)
    try:
        result = storage.run_consolidation(memory_type=mem_type, dry_run=dry_run)

        if use_json:
            click.echo(json.dumps({"success": True, "dry_run": dry_run, **result}))
        elif dry_run:
            _display_consolidation_preview(result)
        else:
            _display_consolidation_results(result)
    finally:
        storage.close()


@cli.command("dashboard")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8765, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def dashboard(host: str, port: int, reload: bool) -> None:
    """Launch the web dashboard for Memory MCP.

    Opens a browser-based interface for viewing and managing memories.

    Examples:

        # Start dashboard on default port
        memory-mcp-cli dashboard

        # Use a different port
        memory-mcp-cli dashboard --port 9000

        # Enable auto-reload for development
        memory-mcp-cli dashboard --reload
    """
    from memory_mcp.dashboard import run_dashboard

    console.print("[bold]Starting Memory MCP Dashboard[/bold]")
    console.print(f"  URL: [cyan]http://{host}:{port}[/cyan]")
    console.print("  Press Ctrl+C to stop\n")

    run_dashboard(host=host, port=port, reload=reload)


@cli.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show memory system status with hot cache contents."""
    use_json = ctx.obj["json"]
    settings = get_settings()

    storage = Storage(settings)
    try:
        stats = storage.get_hot_cache_stats()
        hot_memories = storage.get_hot_memories()
        metrics = storage.get_hot_cache_metrics()
        memory_stats = storage.get_stats()
        health = storage.get_loop_health()

        if use_json:
            click.echo(
                json.dumps(
                    {
                        "memory_stats": memory_stats,
                        "hot_cache": stats,
                        "metrics": metrics.to_dict(),
                        "hot_memories": [
                            {"id": m.id, "content": m.content[:100], "type": m.memory_type.value}
                            for m in hot_memories
                        ],
                        "learning_loop": health,
                    }
                )
            )
            return

        # Header
        console.print("\n[bold cyan]Memory MCP Status[/bold cyan]")
        console.print(f"Database: {settings.db_path}")

        # Memory overview
        console.print("\n[bold]Memory Overview:[/bold]")
        overview_table = Table(show_header=False, box=None)
        overview_table.add_column("Metric", style="dim")
        overview_table.add_column("Value", style="bold")
        overview_table.add_row("Total memories", str(memory_stats["total_memories"]))
        overview_table.add_row(
            "Hot cache", f"{memory_stats['hot_cache_count']}/{stats['max_items']}"
        )

        # Type breakdown
        by_type = memory_stats.get("by_type", {})
        if by_type:
            type_str = ", ".join(f"{t}: {c}" for t, c in sorted(by_type.items()))
            overview_table.add_row("By type", type_str)

        # Source breakdown
        by_source = memory_stats.get("by_source", {})
        if by_source:
            source_str = ", ".join(f"{s}: {c}" for s, c in sorted(by_source.items()))
            overview_table.add_row("By source", source_str)

        console.print(overview_table)

        # Learning loop health
        console.print("\n[bold]Learning Loop:[/bold]")
        loop_table = Table(show_header=False, box=None)
        loop_table.add_column("Metric", style="dim")
        loop_table.add_column("Value", style="bold")

        state_color = {"green": "green", "amber": "yellow", "red": "red"}.get(
            health["state"], "white"
        )
        loop_table.add_row("State", f"[{state_color}]{health['state']}[/{state_color}]")
        loop_table.add_row("Outputs (24h/7d)", f"{health['outputs_24h']}/{health['outputs_7d']}")
        loop_table.add_row("Patterns mined (7d)", str(health["patterns_7d"]))
        loop_table.add_row("Memories created (7d)", str(health["memories_7d"]))
        loop_table.add_row("Last successful run", health["last_success_at"] or "never")

        console.print(loop_table)

        # Hot cache stats
        console.print("\n[bold]Hot Cache Metrics:[/bold]")
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", style="bold")

        stats_table.add_row("Cache hits", str(metrics.hits))
        stats_table.add_row("Cache misses", str(metrics.misses))
        stats_table.add_row("Promotions", str(metrics.promotions))
        stats_table.add_row("Evictions", str(metrics.evictions))

        total = metrics.hits + metrics.misses
        if total > 0:
            hit_rate = metrics.hits / total * 100
            stats_table.add_row("Hit rate", f"{hit_rate:.1f}%")

        console.print(stats_table)

        # Hot memories table
        if not hot_memories:
            console.print("\n[dim]Hot cache is empty[/dim]")
        else:
            console.print("\n[bold]Hot Cache Contents:[/bold]")
            mem_table = Table(show_header=True, header_style="bold magenta")
            mem_table.add_column("ID", style="dim", width=6)
            mem_table.add_column("Type", width=12)
            mem_table.add_column("Content", width=60)
            mem_table.add_column("Pinned", width=6)

            max_display = 10
            for mem in hot_memories[:max_display]:
                content = mem.content.replace("\n", " ")
                preview = content[:57] + "..." if len(content) > 60 else content
                pinned = "[pin]" if mem.is_pinned else ""
                mem_table.add_row(str(mem.id), mem.memory_type.value, preview, pinned)

            console.print(mem_table)

            remaining = len(hot_memories) - max_display
            if remaining > 0:
                console.print(f"  ... and {remaining} more")

    finally:
        storage.close()


@cli.command("hook-check")
@click.option(
    "--no-probe",
    is_flag=True,
    help="Skip the learning-loop round-trip probe",
)
@click.pass_context
def hook_check(ctx: click.Context, no_probe: bool) -> None:
    """Check hook dependencies and database connectivity.

    Validates that the memory-mcp hook can run successfully:
    - uv command is available
    - jq command is available
    - Database is accessible and writable
    - Hook script exists
    - Learning-loop round trip works (log -> mine -> storage)

    The round-trip probe only runs when the database check passed - a probe
    against a broken database is noise, not signal.

    Examples:

        # Check hook dependencies
        memory-mcp-cli hook-check

        # JSON output for scripting
        memory-mcp-cli --json hook-check

        # Skip the round-trip probe
        memory-mcp-cli hook-check --no-probe
    """
    import shutil

    use_json = ctx.obj["json"]
    checks: list[tuple[str, bool, str]] = []

    # Check uv
    uv_path = shutil.which("uv")
    if uv_path:
        checks.append(("uv", True, uv_path))
    else:
        checks.append(("uv", False, "Not found - install from https://astral.sh/uv"))

    # Check jq
    jq_path = shutil.which("jq")
    if jq_path:
        checks.append(("jq", True, jq_path))
    else:
        checks.append(("jq", False, "Not found - install with: brew install jq"))

    # Check database
    settings = get_settings()
    database_ok = False
    storage = None
    try:
        storage = Storage(settings)
        stats = storage.get_stats()
        database_ok = True
        checks.append(("database", True, f"{stats['total_memories']} memories"))
    except Exception as e:
        checks.append(("database", False, str(e)))

    # Round-trip probe - only meaningful once the database is known-good.
    if database_ok and not no_probe:
        result = run_probe(storage)
        if result.ok:
            checks.append(("loop_probe", True, "round trip ok"))
        else:
            checks.append(("loop_probe", False, f"stage={result.stage}: {result.error}"))

    if storage is not None:
        storage.close()

    # Check hook script
    hook_script = Path(__file__).parent.parent.parent / "hooks" / "memory-log-response.sh"
    if hook_script.exists():
        checks.append(("hook_script", True, str(hook_script)))
    else:
        checks.append(("hook_script", False, f"Not found at {hook_script}"))

    # Check log directory
    log_dir = Path.home() / ".memory-mcp"
    if log_dir.exists():
        log_file = log_dir / "hook.log"
        if log_file.exists():
            checks.append(("log_file", True, str(log_file)))
        else:
            checks.append(("log_file", True, f"{log_dir} (no logs yet)"))
    else:
        checks.append(("log_file", True, "Will be created on first run"))

    all_ok = all(c[1] for c in checks)

    if use_json:
        click.echo(
            json.dumps(
                {
                    "success": all_ok,
                    "checks": [
                        {"name": name, "ok": ok, "message": msg} for name, ok, msg in checks
                    ],
                }
            )
        )
    else:
        console.print("[bold]Hook Dependency Check[/bold]")
        for name, ok, msg in checks:
            status = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {status} {name}: {msg}")

        if all_ok:
            console.print("\n[green]All checks passed![/green]")
        else:
            console.print("\n[red]Some checks failed. Fix the issues above.[/red]")

    if not all_ok:
        raise SystemExit(1)


@cli.command("import-beads")
@click.option(
    "-f",
    "--file",
    "filepath",
    type=click.Path(exists=True),
    help="Read from JSONL file (default: pipe from bd export)",
)
@click.option(
    "--include-closed",
    is_flag=True,
    help="Include closed issues (default: open only)",
)
@click.option(
    "--promote",
    is_flag=True,
    help="Promote imported memories to hot cache",
)
@click.option(
    "-p",
    "--project-id",
    help="Project ID override (default: derived from cwd)",
)
@click.pass_context
def import_beads(
    ctx: click.Context,
    filepath: str | None,
    include_closed: bool,
    promote: bool,
    project_id: str | None,
) -> None:
    """Import beads issues as memories.

    Reads JSONL from stdin (pipe from bd export) or a file.

    Examples:

        # Import open issues from current project
        bd export --status open | memory-mcp-cli import-beads

        # Import all issues including closed
        bd export | memory-mcp-cli import-beads --include-closed

        # Import from file and promote to hot cache
        memory-mcp-cli import-beads -f issues.jsonl --promote

        # JSON output for scripting
        bd export | memory-mcp-cli --json import-beads
    """
    import subprocess

    use_json = ctx.obj["json"]
    settings = get_settings()

    # Read content from file or stdin
    if filepath:
        content = Path(filepath).read_text(encoding="utf-8")
    else:
        # Try to run bd export if stdin is empty/tty
        if sys.stdin.isatty():
            # No piped input, run bd export directly
            status_filter = [] if include_closed else ["--status", "open"]
            try:
                proc = subprocess.run(
                    ["bd", "export", *status_filter],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                content = proc.stdout
            except FileNotFoundError:
                click.echo(
                    "Error: bd command not found. Install beads or pipe JSONL to stdin.", err=True
                )
                raise SystemExit(1)
            except subprocess.TimeoutExpired:
                click.echo("Error: bd export timed out", err=True)
                raise SystemExit(1)
        else:
            content = sys.stdin.read()

    if not content.strip():
        click.echo("No issues to import", err=True)
        raise SystemExit(1)

    # Parse JSONL
    issues = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            issue = json.loads(line)
            # Filter closed if not included
            if not include_closed and issue.get("status") == "closed":
                continue
            issues.append(issue)
        except json.JSONDecodeError:
            continue

    if not issues:
        if use_json:
            click.echo(json.dumps({"success": True, "imported": 0, "skipped": 0}))
        else:
            click.echo("No issues to import")
        return

    # Get project_id if not provided
    if project_id is None and settings.project_awareness_enabled:
        project_id = get_current_project_id()

    storage = Storage(settings)
    # Create session for this import
    session_id = str(uuid.uuid4())
    storage.create_or_get_session(session_id, topic="beads import")

    created, skipped, errors = 0, 0, []

    try:
        for issue in issues:
            # Build memory content from issue
            issue_id = issue.get("id", "")
            title = issue.get("title", "")
            description = issue.get("description", "")
            status = issue.get("status", "open")
            priority = issue.get("priority", 2)
            issue_type = issue.get("issue_type", "task")
            notes = issue.get("notes", "")
            design = issue.get("design", "")

            # Format as structured content
            parts = [f"# {title}", f"**Issue**: {issue_id} ({issue_type}, P{priority}, {status})"]
            if description:
                parts.append(description)
            if notes:
                parts.append(f"**Notes**: {notes}")
            if design:
                parts.append(f"**Design**: {design}")

            content = "\n\n".join(parts)

            if len(content) > settings.max_content_length:
                content = content[: settings.max_content_length]
                errors.append(f"Truncated {issue_id}")

            # Store as project memory with beads tag
            memory_id, is_new = storage.store_memory(
                content=content,
                memory_type=MemoryType.PROJECT,
                source=MemorySource.MANUAL,
                project_id=project_id,
                session_id=session_id,
                tags=["beads", issue_type, f"p{priority}"],
            )

            if is_new:
                created += 1
                if promote:
                    storage.promote_to_hot(memory_id)
            else:
                skipped += 1

    finally:
        storage.close()

    result = {
        "imported": created,
        "skipped": skipped,
        "errors": errors,
        "promoted": created if promote else 0,
    }

    if use_json:
        click.echo(json.dumps({"success": True, **result}))
    else:
        console.print("[bold]Beads Import Results[/bold]")
        console.print(f"  Imported: [green]{created}[/green] memories")
        console.print(f"  Skipped: [yellow]{skipped}[/yellow] duplicates")
        if promote:
            console.print(f"  Promoted: [magenta]{created}[/magenta] to hot cache")
        if errors:
            console.print(f"  [dim]Warnings: {len(errors)}[/dim]")


@cli.command("recategorize")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without updating database",
)
@click.option(
    "--uncategorized-only",
    is_flag=True,
    default=True,
    help="Only recategorize memories without a category (default: True)",
)
@click.option(
    "--all",
    "recategorize_all",
    is_flag=True,
    help="Recategorize all memories, including those with existing categories",
)
@click.pass_context
def recategorize(
    ctx: click.Context,
    dry_run: bool,
    uncategorized_only: bool,
    recategorize_all: bool,
) -> None:
    """Re-run category inference on existing memories.

    Useful after adding new category patterns to update old memories.

    Examples:

        # Preview what would change (dry-run)
        memory-mcp-cli recategorize --dry-run

        # Recategorize all uncategorized memories
        memory-mcp-cli recategorize

        # Recategorize ALL memories (overwrite existing categories)
        memory-mcp-cli recategorize --all

        # JSON output for scripting
        memory-mcp-cli --json recategorize
    """
    use_json = ctx.obj["json"]
    settings = get_settings()

    # Import classification function
    from memory_mcp.helpers import infer_category

    def classify_category(content: str) -> str | None:
        """Classify content using ML if enabled, else regex."""
        if settings.ml_classification_enabled:
            from memory_mcp.ml_classification import hybrid_classify_category

            return hybrid_classify_category(content)
        return infer_category(content)

    storage = Storage(settings)
    try:
        # Determine which memories to process
        if recategorize_all:
            where_clause = "1=1"
            filter_desc = "all"
        else:
            where_clause = "category IS NULL"
            filter_desc = "uncategorized"

        with storage._connection() as conn:
            rows = conn.execute(
                f"SELECT id, content, category FROM memories WHERE {where_clause}"
            ).fetchall()

        updates = []
        unchanged = 0
        for row in rows:
            memory_id = row["id"]
            content = row["content"]
            old_category = row["category"]
            new_category = classify_category(content)

            if new_category != old_category:
                updates.append(
                    {
                        "id": memory_id,
                        "old": old_category,
                        "new": new_category,
                        "preview": content[:60].replace("\n", " "),
                    }
                )
            else:
                unchanged += 1

        if not dry_run and updates:
            with storage.transaction() as conn:
                for update in updates:
                    conn.execute(
                        "UPDATE memories SET category = ? WHERE id = ?",
                        (update["new"], update["id"]),
                    )

        result = {
            "filter": filter_desc,
            "total_checked": len(rows),
            "updated": len(updates) if not dry_run else 0,
            "would_update": len(updates) if dry_run else 0,
            "unchanged": unchanged,
            "dry_run": dry_run,
            "changes": updates[:20],  # Limit to first 20 for display
        }

        if use_json:
            click.echo(json.dumps({"success": True, **result}))
        else:
            action = "Would update" if dry_run else "Updated"
            count = result["would_update"] if dry_run else result["updated"]

            console.print(f"[bold]Recategorize Results ({filter_desc} memories)[/bold]")
            console.print(f"  Total checked: [cyan]{result['total_checked']}[/cyan]")
            console.print(f"  {action}: [green]{count}[/green]")
            console.print(f"  Unchanged: [dim]{result['unchanged']}[/dim]")

            if updates:
                console.print("\n[bold]Changes:[/bold]")
                table = Table(show_header=True)
                table.add_column("ID", style="cyan", width=6)
                table.add_column("Old", style="dim", width=12)
                table.add_column("New", style="green", width=12)
                table.add_column("Preview", width=50)
                for u in updates[:20]:
                    table.add_row(
                        str(u["id"]),
                        u["old"] or "(none)",
                        u["new"] or "(none)",
                        u["preview"],
                    )
                console.print(table)
                if len(updates) > 20:
                    console.print(f"  [dim]...and {len(updates) - 20} more[/dim]")

            if dry_run:
                console.print(
                    "\n[yellow]Dry run - no changes made. Run without --dry-run to apply.[/yellow]"
                )

    except Exception as e:
        if use_json:
            click.echo(json.dumps({"success": False, "error": str(e)}))
        else:
            console.print(f"[red]Error: {e}[/red]")
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
