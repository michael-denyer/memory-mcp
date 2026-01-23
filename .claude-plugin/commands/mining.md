---
description: Pattern mining from usage
argument-hint: [status|run|review|approve|reject] [pattern-id]
---

Extract and manage patterns from Claude's outputs.

**Subcommands:**
- `status`: Show mining statistics (`mcp__memory__mining_status`)
- `run`: Run pattern extraction on recent logs (`mcp__memory__run_mining`)
- `review`: Show candidates ready for approval (`mcp__memory__review_candidates`)
- `approve <id>`: Approve a pattern as memory (`mcp__memory__approve_candidate`)
- `reject <id>`: Reject a pattern (`mcp__memory__reject_candidate`)

Mining extracts:
- Import statements
- CLI commands
- Project facts
- Code patterns

Approved patterns become memories and can be promoted to hot cache for instant recall.
