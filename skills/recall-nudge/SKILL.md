---
name: recall-nudge
description: This skill should be used when the user asks a retrospective question about past work or prior decisions, such as "didn't we already try this", "last time we did X", "how did we solve this before", or "have we seen this before". Nudges a call to the memory recall tool before answering from scratch, so the answer reflects what actually happened instead of a guess.
---

# Recall Nudge

When the user asks about past work or prior decisions, call the memory `recall` tool first. Answer grounded in what it returns.

If `recall` returns nothing, say so, then answer normally.
