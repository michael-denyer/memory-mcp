---
description: Consolidate duplicate memories
argument-hint: [preview|run] [type]
---

Merge semantically similar memories to reduce redundancy.

**Subcommands:**
- `preview [type]`: Show clusters without making changes (`mcp__memory__preview_consolidation`)
- `run [type]`: Actually consolidate memories (`mcp__memory__run_consolidation` with `dry_run: false`)

Optional type filter: project, pattern, reference, conversation

Consolidation:
1. Finds clusters of similar memories
2. Keeps the most accessed/valuable one as representative
3. Merges information from others
4. Deletes redundant entries

Always preview first before running actual consolidation.
