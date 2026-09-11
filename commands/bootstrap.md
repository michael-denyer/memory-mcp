---
description: Bootstrap memories from project documentation
argument-hint: [path]
---

Seed the memory system from project documentation files.

If a path is provided as $1, use it as root. Otherwise, use current directory.

Use `mcp__memory__bootstrap_project` tool with:
- `promote_to_hot: true` to make memories instantly available
- Auto-detects: CLAUDE.md, README.md, CONTRIBUTING.md, ARCHITECTURE.md

After bootstrapping, show:
- Number of files processed
- Number of memories created
- Any errors encountered

If hot cache was empty, this is especially important for getting started quickly.
