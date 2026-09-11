---
description: Show memory statistics
---

Display comprehensive statistics about the memory system.

Use `mcp__memory__memory_stats` to retrieve and present:
- Total memories by type (project, pattern, reference, conversation, episodic)
- Hot cache size and utilization
- Storage size on disk

Optionally also show:
- `mcp__memory__hot_cache_status` for hot cache details
- `mcp__memory__metrics_status` for operational metrics
- `mcp__memory__mining_status` for pattern mining stats

Suggest actions based on stats:
- Many memories but empty hot cache? Run `/memory-mcp:bootstrap`
- High recall miss rate? Consider promoting frequent memories
- Many pending mining candidates? Run `/memory-mcp:review-mining`
