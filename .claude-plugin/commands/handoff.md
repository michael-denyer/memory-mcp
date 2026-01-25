# Session Handoff

Store structured memories. One call per fact. No prose.

## Memory Types

### DECISION
```
mcp__memory__remember(
  content="DECISION: [what] | REASON: [why] | ALTERNATIVE: [what was rejected]",
  memory_type="project", category="decision", tags=["handoff", "decision"]
)
```

### GOTCHA
```
mcp__memory__remember(
  content="GOTCHA: [problem] | FIX: [solution] | FILE: [path]",
  memory_type="project", category="gotcha", tags=["handoff", "gotcha"]
)
```

### BUG
```
mcp__memory__remember(
  content="BUG: [symptom] | ROOT_CAUSE: [why] | FIX: [how] | FILE: [path] | SEVERITY: [critical|high|medium|low]",
  memory_type="project", category="bug", tags=["handoff", "bug"]
)
```

### TASK
```
mcp__memory__remember(
  content="TASK: [name] | STATUS: [in_progress|blocked] | NEXT: [action] | BLOCKER: [if blocked]",
  memory_type="conversation", category="task", tags=["handoff", "task"]
)
```

### CONSTRAINT
```
mcp__memory__remember(
  content="CONSTRAINT: [limitation] | SCOPE: [where it applies] | WORKAROUND: [if any]",
  memory_type="project", category="constraint", tags=["handoff", "constraint"]
)
```

### PATTERN
```
mcp__memory__remember(
  content="PATTERN: [name] | USE: [when to use] | EXAMPLE: [code or reference]",
  memory_type="pattern", category="pattern", tags=["handoff", "pattern"]
)
```

## Rules

- One fact per memory
- Pipe-delimited fields only
- No sentences, no explanations
- Skip empty fields
- Use exact file paths and names

## Example

```python
mcp__memory__remember(
  content="DECISION: SQLite over Postgres | REASON: No server needed | ALTERNATIVE: Postgres",
  memory_type="project", category="decision", tags=["handoff", "decision"]
)

mcp__memory__remember(
  content="GOTCHA: vec0 no INSERT OR REPLACE | FIX: DELETE then INSERT | FILE: storage/vectors.py",
  memory_type="project", category="gotcha", tags=["handoff", "gotcha"]
)

mcp__memory__remember(
  content="BUG: Test fails with mining disabled | ROOT_CAUSE: Default changed to disabled | FIX: Accept both error messages | FILE: tests/test_server.py | SEVERITY: low",
  memory_type="project", category="bug", tags=["handoff", "bug"]
)

mcp__memory__remember(
  content="TASK: Session handoffs | STATUS: in_progress | NEXT: recall-handoffs CLI",
  memory_type="conversation", category="task", tags=["handoff", "task"]
)
```
