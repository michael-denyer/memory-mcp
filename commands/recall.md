---
description: Search memories semantically
argument-hint: [query]
---

Search for memories using semantic similarity.

If a query is provided as $1, use it. Otherwise, ask the user what they want to recall.

Use the `mcp__memory__recall` tool with:
- `mode: "balanced"` for general searches
- `mode: "precision"` when user wants exact matches
- `mode: "exploratory"` when user wants broad results
- `expand_relations: true` to include related memories via knowledge graph

Present results clearly showing:
- Memory content
- Confidence level (high/medium/low)
- Memory type and tags
- Related memories if expanded

If no results found, suggest:
- Trying a different query
- Using `/memory-mcp:list` to browse all memories
- Checking if the topic was ever stored
