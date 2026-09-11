---
description: List stored memories
argument-hint: [type] [limit]
---

List memories with optional filtering.

Arguments:
- $1: Memory type filter (project, pattern, reference, conversation, episodic)
- $2: Limit (default: 20)

Use `mcp__memory__list_memories` tool.

Present results in a clear table showing:
- Memory ID
- Type
- Content preview (truncated)
- Access count
- Created date

Offer follow-up actions:
- `/memory-mcp:show <id>` to see full details
- `/memory-mcp:recall <query>` for semantic search
