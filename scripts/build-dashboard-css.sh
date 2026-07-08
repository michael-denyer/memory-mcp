#!/usr/bin/env bash
#
# Compile the dashboard's Tailwind CSS into a vendored, offline-capable file.
# Run this after adding new Tailwind classes to any dashboard template so the
# generated static/tailwind.css keeps those classes.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INPUT="$(mktemp -t tailwind-input-XXXXXX).css"
trap 'rm -f "$INPUT"' EXIT
printf '@tailwind base;\n@tailwind components;\n@tailwind utilities;\n' >"$INPUT"

npx --yes tailwindcss@3.4.17 \
  --config scripts/tailwind.dashboard.config.js \
  --input "$INPUT" \
  --output src/memory_mcp/dashboard/static/tailwind.css \
  --minify

# Tailwind's minified output has no trailing newline; add one so the file
# satisfies the end-of-file-fixer pre-commit hook and stays byte-stable.
printf '\n' >>src/memory_mcp/dashboard/static/tailwind.css

echo "Wrote src/memory_mcp/dashboard/static/tailwind.css"
