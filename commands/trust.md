---
description: Manage memory trust scores
argument-hint: [validate|invalidate] <memory-id> [reason]
---

Adjust trust scores for memories based on accuracy.

**Subcommands:**
- `validate <id> [reason]`: Mark memory as confirmed useful
- `invalidate <id> [reason]`: Mark memory as incorrect/outdated

**Validation reasons** (with default trust boosts):
- `used_correctly`: +0.05 (memory was applied successfully)
- `explicitly_confirmed`: +0.15 (user verified accuracy)
- `cross_validated`: +0.20 (corroborated by multiple sources)

**Invalidation reasons** (with default penalties):
- `outdated`: -0.10 (information is stale)
- `partially_incorrect`: -0.15 (some details wrong)
- `factually_wrong`: -0.30 (fundamentally incorrect)
- `superseded`: -0.05 (replaced by newer info)
- `low_utility`: -0.05 (not useful in practice)

Use `mcp__memory__validate_memory` or `mcp__memory__invalidate_memory`.

View trust history with `mcp__memory__get_trust_history`.
