# Project Onboarding Workflow

Store key project facts in memory so Claude can reference them across sessions.

## Scenario

You're starting work on a new project and want Claude to remember the important details without repeating them each session.

## Quick Start: Auto-Bootstrap

If your project has README.md or CLAUDE.md, the hot cache auto-populates on first access:

```
# Via CLI
uv run memory-mcp-cli bootstrap

# Or via MCP tool
Claude: [calls bootstrap_project()]
→ Bootstrapped 15 memories from 2 file(s) (15 promoted to hot cache)
```

For manual control, follow the steps below.

## Step 1: Store Project Facts

Tell Claude about your project:

```
You: "Remember that this project uses FastAPI with PostgreSQL and pgvector for
     vector similarity search. We use pytest for testing and ruff for linting."

Claude: [calls remember(content="This project uses FastAPI with PostgreSQL and
         pgvector for vector similarity search. Uses pytest for testing and
         ruff for linting.", memory_type="project", tags=["tech-stack"])]

→ Stored as memory #1
```

Store architecture decisions:

```
You: "Remember that we use a hexagonal architecture with ports and adapters.
     Domain logic is in src/domain/, adapters in src/adapters/."

Claude: [calls remember(content="Uses hexagonal architecture with ports and
         adapters. Domain logic in src/domain/, adapters in src/adapters/.",
         memory_type="project", tags=["architecture"])]

→ Stored as memory #2
```

## Step 2: Promote to Hot Cache

For facts you'll need every session, promote them:

```
You: "Promote the tech stack memory to hot cache"

Claude: [calls promote(1)]

→ Memory #1 now in hot cache - instant recall
```

## Step 3: Use in Future Sessions

In a new session, Claude automatically has hot cache context. For other facts:

```
You: "What testing framework does this project use?"

Claude: [calls recall("testing framework")]

→ {confidence: "high", memories: [{content: "...pytest for testing..."}]}

Claude: "This project uses pytest for testing."
```

## Step 4: Check What's Stored

```
You: "What do you remember about this project?"

Claude: [calls hot_cache_status()]
Claude: [calls recall("project architecture conventions")]

→ Shows hot cache items + semantically similar memories
```

## Tips

- **Be specific**: "Uses PostgreSQL 15 with pgvector 0.5" is better than "uses a database"
- **Tag consistently**: Use tags like `tech-stack`, `architecture`, `conventions`
- **Promote strategically**: Only promote facts you need every session
- **Review periodically**: Run `memory_stats()` to see what's stored
