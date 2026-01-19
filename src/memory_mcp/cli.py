"""CLI commands for memory-mcp.

These commands can be called from shell scripts and Claude Code hooks.
"""

import argparse
import json
import sys

from memory_mcp.config import get_settings
from memory_mcp.storage import Storage


def log_output_cmd(args: argparse.Namespace) -> int:
    """Log output content for pattern mining."""
    settings = get_settings()

    if not settings.mining_enabled:
        print("Mining is disabled", file=sys.stderr)
        return 1

    # Read content from file or stdin
    if args.file:
        with open(args.file) as f:
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
