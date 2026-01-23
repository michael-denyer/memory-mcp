"""Secret redaction utilities.

This module is kept separate from mining.py to avoid circular imports,
since output_logging.py needs redaction but mining.py imports from storage.
"""

from __future__ import annotations

import re

# Patterns that indicate potentially sensitive content - never auto-approve
SENSITIVE_PATTERNS = (
    # Credential keywords
    r"password\s*[:=]",
    r"passwd\s*[:=]",
    r"secret\s*[:=]",
    r"token\s*[:=]",
    r"api[_-]?key\s*[:=]",
    r"auth[_-]?token\s*[:=]",
    r"private[_-]?key\s*[:=]",
    r"encryption[_-]?key\s*[:=]",
    # Connection strings with credentials
    r"://\w+:\w+@",  # user:pass@host in URLs
    r"mongodb\+srv://.*:.*@",
    r"postgres://.*:.*@",
    r"mysql://.*:.*@",
    # AWS/cloud credentials
    r"AKIA[0-9A-Z]{16}",  # AWS access key
    r"aws_secret",
    r"gcp_key",
    r"azure_key",
    # Bearer tokens
    r"bearer\s+[a-zA-Z0-9\-_\.]+",
    # Base64-encoded secrets (long random strings)
    r"[a-zA-Z0-9+/]{40,}={0,2}",  # Likely base64 encoded secret
)

# Compile for efficiency
_SENSITIVE_REGEX = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


def may_contain_secrets(text: str) -> bool:
    """Check if text may contain sensitive information.

    Used to prevent auto-approval of patterns that might contain secrets.
    """
    return bool(_SENSITIVE_REGEX.search(text))


# Redaction patterns: (compiled_regex, replacement)
# These are more specific than detection patterns - they match actual secret values
_REDACTION_PATTERNS: list[tuple[re.Pattern, str]] = []


def _init_redaction_patterns() -> None:
    """Initialize compiled redaction patterns (lazy load)."""
    global _REDACTION_PATTERNS
    if _REDACTION_PATTERNS:
        return

    patterns = [
        # API keys with specific formats
        (r"sk-[A-Za-z0-9]{48,}", "[OPENAI_KEY_REDACTED]"),
        (r"ghp_[A-Za-z0-9]{36,}", "[GITHUB_PAT_REDACTED]"),
        (r"gho_[A-Za-z0-9]{36,}", "[GITHUB_OAUTH_REDACTED]"),
        (r"AKIA[0-9A-Z]{16}", "[AWS_KEY_REDACTED]"),
        # Key-value pairs with secrets (captures the key, redacts value)
        (
            r"((?:password|passwd|secret|token|api[_-]?key|auth[_-]?token|private[_-]?key)"
            r"\s*[:=]\s*)['\"]?[A-Za-z0-9_\-./+]{8,}['\"]?",
            r"\1[REDACTED]",
        ),
        # Connection strings with credentials
        (r"(://[^:]+:)[^@]+(@)", r"\1[REDACTED]\2"),
        # Bearer tokens
        (r"(bearer\s+)[A-Za-z0-9\-_.]{20,}", r"\1[REDACTED]"),
    ]

    _REDACTION_PATTERNS.extend((re.compile(p, re.IGNORECASE), r) for p, r in patterns)


def redact_secrets(text: str) -> str:
    """Redact detected secrets from text before storage.

    Replaces detected secrets with [REDACTED] or specific redaction markers.
    This should be called BEFORE storing content to prevent secret persistence.

    Args:
        text: Content that may contain secrets

    Returns:
        Text with secrets redacted
    """
    _init_redaction_patterns()

    result = text
    for pattern, replacement in _REDACTION_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
