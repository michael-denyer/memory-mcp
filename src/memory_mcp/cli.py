"""CLI commands for memory-mcp.

These commands can be called from shell scripts and Claude Code hooks.
"""

import argparse
import json
import sys
from pathlib import Path

from memory_mcp.config import get_settings
from memory_mcp.storage import MemorySource, MemoryType, Storage
from memory_mcp.text_parsing import parse_content_into_chunks


def log_output_cmd(args: argparse.Namespace) -> int:
    """Log output content for pattern mining."""
    settings = get_settings()

    if not settings.mining_enabled:
        print("Mining is disabled", file=sys.stderr)
        return 1

    # Read content from file or stdin
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
    elif args.content:
        content = args.content
    else:
        content = sys.stdin.read()

    if not content.strip():
        print("No content to log", file=sys.stderr)
        return 1

    if len(content) > settings.max_content_length:
        print(
            f"Content too long ({len(content)} chars). Max: {settings.max_content_length}",
            file=sys.stderr,
        )
        return 1

    storage = Storage(settings)
    try:
        log_id = storage.log_output(content)
        if args.json:
            print(json.dumps({"success": True, "log_id": log_id}))
        else:
            print(f"Logged output (id={log_id})")
        return 0
    finally:
        storage.close()


def run_mining_cmd(args: argparse.Namespace) -> int:
    """Run pattern mining on logged outputs."""
    settings = get_settings()

    if not settings.mining_enabled:
        print("Mining is disabled", file=sys.stderr)
        return 1

    from memory_mcp.mining import run_mining

    storage = Storage(settings)
    try:
        result = run_mining(storage, hours=args.hours)
        if args.json:
            print(json.dumps(result))
        else:
            print(f"Processed {result['outputs_processed']} outputs")
            print(f"Found {result['patterns_found']} patterns")
        return 0
    finally:
        storage.close()


def seed_cmd(args: argparse.Namespace) -> int:
    """Seed memories from a file."""
    path = Path(args.file).expanduser()
    if not path.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"Read error: {e}", file=sys.stderr)
        return 1

    try:
        mem_type = MemoryType(args.type)
    except ValueError:
        print(f"Invalid type: {args.type}", file=sys.stderr)
        return 1

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
                if args.promote:
                    storage.promote_to_hot(memory_id)
            else:
                skipped += 1
    finally:
        storage.close()

    if args.json:
        print(
            json.dumps(
                {
                    "memories_created": created,
                    "memories_skipped": skipped,
                    "errors": errors,
                }
            )
        )
    else:
        print(f"Created {created} memories, skipped {skipped} duplicates")
        if errors:
            print(f"Errors: {len(errors)}")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="memory-mcp-cli",
        description="CLI commands for memory-mcp",
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # log-output command
    log_parser = subparsers.add_parser(
        "log-output",
        help="Log output content for pattern mining",
    )
    log_parser.add_argument(
        "-c",
        "--content",
        help="Content to log (or use stdin)",
    )
    log_parser.add_argument(
        "-f",
        "--file",
        help="Read content from file",
    )
    log_parser.set_defaults(func=log_output_cmd)

    # run-mining command
    mining_parser = subparsers.add_parser(
        "run-mining",
        help="Run pattern mining on logged outputs",
    )
    mining_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours of logs to process (default: 24)",
    )
    mining_parser.set_defaults(func=run_mining_cmd)

    # seed command
    seed_parser = subparsers.add_parser(
        "seed",
        help="Seed memories from a file (e.g., CLAUDE.md)",
    )
    seed_parser.add_argument(
        "file",
        help="File to import memories from",
    )
    seed_parser.add_argument(
        "-t",
        "--type",
        default="project",
        help="Memory type (project, pattern, reference, conversation)",
    )
    seed_parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote all seeded memories to hot cache",
    )
    seed_parser.set_defaults(func=seed_cmd)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
