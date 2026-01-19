# Cross-Session Continuity Workflow

Preserve context and decisions across Claude Code sessions.

## Scenario

You made important decisions in yesterday's session. Today you need to continue the work without re-explaining everything.

## Session 1: Making Decisions

```
You: "We decided to use JWT with RS256 for auth because we need key rotation.
     Remember this decision."

Claude: [calls remember(content="Auth decision: Using JWT with RS256 algorithm
         for authentication to enable key rotation without invalidating tokens.",
         memory_type="project", tags=["auth", "decision"])]

→ Stored as memory #5

You: "We also decided the token expiry should be 15 minutes with 7-day refresh
     tokens. Remember this too."

Claude: [calls remember(content="Token configuration: 15-minute access tokens,
         7-day refresh tokens. Short access tokens for security, longer refresh
         for UX.", memory_type="project", tags=["auth", "decision"])]

→ Stored as memory #6
```

## Session 2: Continuing Work

Next day, in a new session:

```
You: "I'm implementing the auth system. What did we decide about tokens?"

Claude: [calls recall("authentication token configuration decision")]

→ {confidence: "high", memories: [
     {content: "Auth decision: Using JWT with RS256...", similarity: 0.89},
     {content: "Token configuration: 15-minute access tokens...", similarity: 0.87}
   ]}

Claude: "Based on our previous decisions:
- Using JWT with RS256 for key rotation capability
- Access tokens expire in 15 minutes
- Refresh tokens last 7 days

Shall I implement the token generation code?"
```

## Storing Implementation Notes

As you work, store what you learn:

```
You: "Remember that we're using python-jose for JWT handling, installed with
     uv add python-jose[cryptography]"

Claude: [calls remember(content="JWT implementation: Using python-jose library
         with cryptography backend. Install: uv add python-jose[cryptography]",
         memory_type="pattern", tags=["auth", "dependency"])]

→ Stored as memory #7
```

## Recall Modes for Different Needs

```
# High-confidence answers only (for implementation)
Claude: [calls recall("JWT implementation", mode="precision")]

# Broader exploration (for research)
Claude: [calls recall("authentication approaches", mode="exploratory")]

# Search across types with fallback
Claude: [calls recall_with_fallback("auth token")]
```

## Tips

- **Store decisions with reasoning**: Include the "why" not just the "what"
- **Tag related decisions**: Use consistent tags like `auth`, `decision`, `api`
- **Store gotchas**: "This doesn't work because..." saves future debugging
- **Use precision mode**: When implementing, use high-confidence results only
