# Pattern Mining Workflow

Automatically extract and promote frequently-used patterns from your sessions.

## Scenario

You repeatedly use certain imports, commands, or code patterns. The mining pipeline detects these and suggests promoting them to hot cache.

## How Mining Works

```
1. Claude's responses are logged (via Stop hook)
2. run_mining() extracts patterns:
   - Import statements
   - Project facts ("This project uses X")
   - Code patterns (function signatures)
   - Shell commands
3. Patterns seen 3+ times become promotion candidates
4. You approve candidates to promote to hot cache
```

## Step 1: Enable Auto-Logging

Install the hook (see README):

```json
// ~/.claude/settings.json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "/path/to/memory-mcp/hooks/memory-log-response.sh"
      }]
    }]
  }
}
```

## Step 2: Work Normally

Over several sessions, Claude generates responses containing patterns:

```
Claude: "Here's how to run the tests:
```bash
uv run pytest tests/ -v
```"

Claude: "Let me add the import:
```python
from fastapi import FastAPI, HTTPException, Depends
```"

Claude: "This project uses SQLAlchemy 2.0 with async support..."
```

These get logged automatically.

## Step 3: Run Mining

Periodically run mining to extract patterns:

```
You: "Run pattern mining"

Claude: [calls run_mining(hours=24)]

→ Processed 47 outputs, found 12 patterns

You: "What patterns are ready for promotion?"

Claude: [calls review_candidates()]

→ [
    {id: 1, pattern: "from fastapi import FastAPI, HTTPException, Depends",
     type: "import", occurrences: 5},
    {id: 2, pattern: "uv run pytest tests/ -v", type: "command", occurrences: 4},
    {id: 3, pattern: "This project uses SQLAlchemy 2.0", type: "fact", occurrences: 3}
  ]
```

## Step 4: Approve or Reject

```
You: "Approve the FastAPI import pattern"

Claude: [calls approve_candidate(1)]

→ Pattern promoted to hot cache as memory #15

You: "Reject the pytest command - I use different flags now"

Claude: [calls reject_candidate(2)]

→ Pattern rejected and removed from candidates
```

## Step 5: Check Mining Status

```
You: "Show mining status"

Claude: [calls mining_status()]

→ {
    outputs_logged: 234,
    patterns_extracted: 45,
    candidates_pending: 8,
    promoted_this_week: 3,
    log_retention_days: 7
  }
```

## Pattern Types

| Type | What's Extracted | Example |
|------|------------------|---------|
| `import` | Python imports | `from pydantic import BaseModel` |
| `command` | Shell commands | `uv run pytest -v` |
| `fact` | Project facts | "This project uses PostgreSQL" |
| `code` | Function signatures | `def process_data(df: pd.DataFrame)` |

## CLI Mining

Run mining from command line:

```bash
# Process last 24 hours
uv run memory-mcp-cli run-mining --hours 24

# Process last week
uv run memory-mcp-cli run-mining --hours 168

# Get JSON output
uv run memory-mcp-cli --json run-mining
```

## Tips

- **Run mining daily**: `run_mining(hours=24)` catches new patterns
- **Review candidates weekly**: Don't let the queue grow too large
- **Reject outdated patterns**: If your workflow changed, reject old patterns
- **Check occurrences**: Higher count = more likely to be useful
