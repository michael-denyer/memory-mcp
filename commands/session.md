---
description: Manage conversation sessions
argument-hint: [list|show|summarize|end] [session-id]
---

Manage conversation sessions and episodic memory consolidation.

**Subcommands:**
- `list`: Show recent sessions (`mcp__memory__get_sessions`)
- `show <id>`: Get session details (`mcp__memory__get_session`)
- `summarize <id>`: Get structured summary (`mcp__memory__summarize_session`)
- `end <id>`: End session and consolidate memories (`mcp__memory__end_session`)

Session summarization groups memories into:
- **Decisions**: Choices made and rationale
- **Insights**: Lessons learned, antipatterns, constraints
- **Action Items**: TODOs and tasks to complete
- **Context**: Background info, conventions, preferences

Ending a session promotes top episodic memories to long-term storage based on salience score.
