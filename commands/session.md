---
description: End a conversation session and consolidate its memories
argument-hint: <session-id>
---

End a conversation session and consolidate its episodic memories.

Use `mcp__memory__end_session` with the session ID as $1.

Options:
- `promote_top`: Promote top episodic memories to long-term storage (default: true)
- `promote_type`: Memory type for promoted memories, `project` or `pattern` (default: project)

Top episodic memories are selected by salience score, which combines importance, trust,
access count, and recency. Only memories above the threshold are promoted.
