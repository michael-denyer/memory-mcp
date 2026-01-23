---
description: Delete a memory permanently
argument-hint: <memory-id>
---

Permanently delete a memory from storage.

Requires a memory ID as $1.

Use `mcp__memory__forget` tool.

**Warning**: This is permanent and cannot be undone.

Before deleting, consider:
- Is the memory just outdated? Use `/memory-mcp:trust invalidate` instead
- Is it superseded? Use `/memory-mcp:link` with `supersedes` relation
- Should it just leave hot cache? Use `/memory-mcp:hot-cache demote`

After deletion, the memory ID is logged in the audit history.
