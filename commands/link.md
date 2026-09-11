---
description: Link related memories
argument-hint: <from-id> <to-id> [relation-type]
---

Create a relationship between two memories in the knowledge graph.

Arguments:
- $1: Source memory ID (required)
- $2: Target memory ID (required)
- $3: Relation type (optional, default: relates_to)

**Relation types:**
- `relates_to`: General relationship
- `depends_on`: Prerequisite knowledge
- `supersedes`: Replaces older information
- `refines`: More specific version
- `contradicts`: Conflicting information
- `elaborates`: More detailed explanation
- `mentions`: References an entity

Use `mcp__memory__link_memories` tool.

After linking, offer to:
- View the knowledge graph with `mcp__memory__get_related_memories`
- Find contradictions with `mcp__memory__find_contradictions`
