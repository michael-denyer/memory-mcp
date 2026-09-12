---
description: Run database maintenance
---

Run database maintenance with `mcp__memory__db_maintenance`.

It performs:
- VACUUM to reclaim unused space
- ANALYZE to update query planner stats
- Auto-demote stale hot memories

Report bytes reclaimed, total memory count, and how many memories were auto-demoted.
