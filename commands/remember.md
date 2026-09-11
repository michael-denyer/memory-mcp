---
description: Store a new memory
argument-hint: [content] [type] [tags...]
---

Store a new memory in the Memory MCP system.

If arguments are provided:
- $1: Memory content (required)
- $2: Memory type (project, pattern, reference, conversation, episodic)
- $3+: Tags (optional, space-separated)

If no arguments, ask the user for:
1. What to remember (required)
2. Memory type (default: project)
3. Tags for categorization (optional)

Use the `mcp__memory__remember` tool to store the memory.

After storing, show the memory ID and offer to:
- Add related memories with `mcp__memory__link_memories`
- Promote to hot cache with `mcp__memory__promote` if important
