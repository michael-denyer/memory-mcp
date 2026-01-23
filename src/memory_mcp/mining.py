"""Pattern mining from output logs."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from memory_mcp.embeddings import content_hash
from memory_mcp.storage import MemoryType, Storage

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


def _may_contain_secrets(text: str) -> bool:
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


# Global lazy-loaded NER pipeline
_ner_pipeline: Any = None


class PatternType(str, Enum):
    """Types of patterns that can be mined."""

    # Original types
    IMPORT = "import"  # Import statements
    FACT = "fact"  # "This project uses X" statements
    COMMAND = "command"  # Shell commands
    CODE = "code"  # Code snippets
    CODE_BLOCK = "code_block"  # Fenced code blocks from markdown

    # Enhanced regex types
    DECISION = "decision"  # Architecture/design decisions
    ARCHITECTURE = "architecture"  # System architecture descriptions
    TECH_STACK = "tech_stack"  # Technology mentions with context
    EXPLANATION = "explanation"  # Rationale and reasoning
    CONFIG = "config"  # Configuration facts

    # NER entity types (from DistilBERT-NER)
    ENTITY_PERSON = "entity_person"  # Person names
    ENTITY_ORG = "entity_org"  # Organization names
    ENTITY_LOCATION = "entity_location"  # Location names
    ENTITY_MISC = "entity_misc"  # Miscellaneous entities

    # Additional high-value pattern types
    DEPENDENCY = "dependency"  # Package dependencies with versions
    API_ENDPOINT = "api_endpoint"  # REST/HTTP endpoints

    # Technology entity type (for knowledge graph linking)
    ENTITY_TECHNOLOGY = "entity_technology"  # Technology/framework/tool mentions
    ENTITY_DECISION = "entity_decision"  # Architecture/design decisions with rationale

    # Long-form contextual content
    INSIGHT = "insight"  # Key insights, summaries, and contextual explanations


# Common CLI tool prefixes for command extraction
COMMAND_PREFIXES = (
    "npm",
    "yarn",
    "pnpm",
    "uv",
    "pip",
    "python",
    "node",
    "git",
    "docker",
    "make",
    "cargo",
    "go",
)

# Env var names containing these substrings are considered sensitive and never extracted
SENSITIVE_ENV_NAMES = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "key",
    "api_key",
    "apikey",
    "auth",
    "credential",
    "private",
    "encryption",
    "signing",
    "database_url",
    "db_url",
    "connection_string",
    "dsn",
    "uri",
)


@dataclass
class ExtractedPattern:
    """A pattern extracted from output."""

    pattern: str
    pattern_type: PatternType
    confidence: float = 0.5  # Extraction confidence (0-1)
    metadata: dict[str, Any] | None = None  # Optional metadata (e.g., entity_type)


# ========== NER Pipeline (Lazy-Loaded) ==========


def _get_ner_pipeline() -> Any:
    """Lazy-load NER pipeline. Auto-downloads model on first use (~250MB).

    Returns the pipeline if transformers is installed, None otherwise.
    """
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import pipeline

            # Suppress verbose transformers logging
            logging.getLogger("transformers").setLevel(logging.ERROR)

            # Model auto-downloads on first use
            _ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="average",  # Combine subword tokens for multiword entities
            )
        except ImportError:
            _ner_pipeline = False  # Mark as unavailable
    return _ner_pipeline if _ner_pipeline else None


def _split_into_chunks(text: str, max_length: int = 512) -> list[str]:
    """Split text into chunks for NER processing.

    BERT models have a max token limit (~512). We split on sentence
    boundaries to avoid cutting entities.
    """
    # Simple sentence splitting (could be improved with nltk)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence[:max_length]  # Truncate long sentences

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:max_length]]


def extract_entities_ner(text: str, min_confidence: float = 0.7) -> list[ExtractedPattern]:
    """Extract named entities using DistilBERT-NER if available.

    Extracts Person, Organization, Location, and Miscellaneous entities
    with confidence scores from the NER model. Includes surrounding context
    to make entities more useful for recall.

    Falls back to empty list if transformers is not installed.
    """
    ner = _get_ner_pipeline()
    if ner is None:
        return []  # NER not available, will fall back to regex

    patterns = []
    seen_entities: set[str] = set()

    # Map NER labels to our pattern types and human-readable descriptions
    label_map = {
        "PER": (PatternType.ENTITY_PERSON, "person"),
        "ORG": (PatternType.ENTITY_ORG, "organization"),
        "LOC": (PatternType.ENTITY_LOCATION, "location"),
        "MISC": (PatternType.ENTITY_MISC, "entity"),
    }

    # Process in chunks to handle long text
    for chunk in _split_into_chunks(text, max_length=512):
        try:
            entities = ner(chunk)
        except Exception:
            continue  # Skip chunks that fail

        for entity in entities:
            word = entity.get("word", "").strip()
            score = entity.get("score", 0)
            label = entity.get("entity_group", "")
            start = entity.get("start", 0)
            end = entity.get("end", len(word))

            # Filter by confidence and minimum length
            if score < min_confidence or len(word) < 2:
                continue

            # Skip common false positives
            if word.lower() in {"the", "a", "an", "this", "that", "it", "i", "we", "you"}:
                continue

            # Deduplicate (case-insensitive) - keep first occurrence with best context
            word_lower = word.lower()
            if word_lower in seen_entities:
                continue
            seen_entities.add(word_lower)

            pattern_type, type_label = label_map.get(label, (PatternType.ENTITY_MISC, "entity"))

            # Extract surrounding context (sentence or ~40 chars each side)
            # This makes the entity more useful for recall
            context_start = max(0, start - 40)
            context_end = min(len(chunk), end + 40)
            context = chunk[context_start:context_end].strip()
            # Clean up - normalize whitespace and trim to sentence boundaries if possible
            context = " ".join(context.split())

            # If context is too short, just use entity with type annotation
            if len(context) < len(word) + 10:
                pattern_text = f"{word} ({type_label})"
            else:
                pattern_text = f"...{context}... [{word} is a {type_label}]"

            # Convert numpy float to Python float for SQLite compatibility
            patterns.append(ExtractedPattern(pattern_text, pattern_type, confidence=float(score)))

    return patterns


# ========== Pattern Extractors ==========


def extract_imports(text: str) -> list[ExtractedPattern]:
    """Extract Python import statements."""
    patterns = []

    # Python imports: import X, from X import Y
    import_re = re.compile(
        r"^(?:from\s+[\w.]+\s+import\s+[\w,\s*]+|import\s+[\w,\s.]+)$",
        re.MULTILINE,
    )

    for match in import_re.findall(text):
        # Normalize whitespace
        normalized = " ".join(match.split())
        if len(normalized) > 10:  # Skip trivial imports
            patterns.append(ExtractedPattern(normalized, PatternType.IMPORT))

    return patterns


def extract_facts(text: str) -> list[ExtractedPattern]:
    """Extract factual statements about the project."""
    patterns = []

    # Common fact patterns
    fact_patterns = [
        r"[Tt]his project uses\s+[\w\s,]+",
        r"[Ww]e use\s+[\w\s,]+(?:for|to)\s+[\w\s,]+",
        r"[Tt]he (?:API|database|server|client) (?:is|uses|runs)\s+[\w\s,]+",
        r"[Aa]uthentication (?:uses|is handled by)\s+[\w\s,]+",
        r"[Tt]ests (?:use|are run with)\s+[\w\s,]+",
    ]

    for pattern in fact_patterns:
        fact_re = re.compile(pattern)
        for match in fact_re.findall(text):
            normalized = match.strip()
            if 10 < len(normalized) < 200:  # Reasonable length
                patterns.append(ExtractedPattern(normalized, PatternType.FACT))

    return patterns


def extract_commands(text: str) -> list[ExtractedPattern]:
    """Extract shell commands."""
    patterns = []

    # Common command patterns
    command_re = re.compile(
        r"(?:^|\n)[$>]\s*(.+?)(?:\n|$)|"  # $ or > prompts
        r"`([^`]+)`|"  # Backtick commands
        r"(?:run|execute|use):\s*`([^`]+)`",  # "run: `command`"
        re.MULTILINE,
    )

    for match in command_re.findall(text):
        cmd = next((m for m in match if m), None)
        if not cmd:
            continue

        normalized = cmd.strip()
        is_known_command = normalized.startswith(COMMAND_PREFIXES)
        has_valid_length = 5 < len(normalized) < 200

        if is_known_command and has_valid_length:
            patterns.append(ExtractedPattern(normalized, PatternType.COMMAND))

    return patterns


def extract_code_patterns(text: str) -> list[ExtractedPattern]:
    """Extract notable code patterns."""
    patterns = []

    # Function definitions
    func_re = re.compile(
        r"(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->[\w\[\],\s|]+)?:",
        re.MULTILINE,
    )

    for match in func_re.findall(text):
        # Get the full line for context
        full_pattern = f"def {match}(...)"
        if not match.startswith("_"):  # Skip private functions
            patterns.append(ExtractedPattern(full_pattern, PatternType.CODE))

    # Class definitions
    class_re = re.compile(r"class\s+(\w+)\s*(?:\([^)]*\))?:", re.MULTILINE)
    for match in class_re.findall(text):
        patterns.append(ExtractedPattern(f"class {match}", PatternType.CODE))

    return patterns


def extract_code_blocks(text: str) -> list[ExtractedPattern]:
    """Extract fenced code blocks from markdown.

    Extracts code blocks like:
    ```python
    def example():
        pass
    ```
    """
    patterns = []

    # Match fenced code blocks with optional language identifier
    code_block_re = re.compile(
        r"```(\w*)\n(.*?)```",
        re.DOTALL,
    )

    for match in code_block_re.finditer(text):
        language = match.group(1).lower() or None
        code = match.group(2).strip()

        # Skip very short or very long blocks
        if len(code) < 20 or len(code) > 2000:
            continue

        # Skip blocks that are just error messages or output
        if code.startswith("Error:") or code.startswith("Traceback"):
            continue

        # Higher confidence for blocks with language identifier
        confidence = 0.7 if language else 0.5

        # Include language in pattern for context
        pattern_text = f"[{language}]\n{code}" if language else code
        patterns.append(
            ExtractedPattern(pattern_text, PatternType.CODE_BLOCK, confidence=confidence)
        )

    return patterns


# ========== Enhanced Regex Extractors ==========

# Known technologies for tech stack extraction (case-insensitive)
KNOWN_TECH = {
    "frameworks": [
        "fastapi",
        "django",
        "flask",
        "react",
        "vue",
        "angular",
        "express",
        "nextjs",
        "nuxt",
        "svelte",
        "rails",
        "spring",
        "laravel",
        "gin",
        "echo",
        "fiber",
        "actix",
        "axum",
        "rocket",
    ],
    "databases": [
        "postgresql",
        "postgres",
        "mysql",
        "mongodb",
        "redis",
        "sqlite",
        "dynamodb",
        "cassandra",
        "elasticsearch",
        "neo4j",
        "supabase",
        "firestore",
        "cockroachdb",
        "mariadb",
    ],
    "tools": [
        "docker",
        "kubernetes",
        "k8s",
        "terraform",
        "ansible",
        "jenkins",
        "github actions",
        "gitlab ci",
        "circleci",
        "aws",
        "gcp",
        "azure",
        "vercel",
        "netlify",
        "heroku",
        "cloudflare",
    ],
    "languages": [
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "java",
        "kotlin",
        "swift",
        "ruby",
        "php",
        "c#",
        "c++",
        "scala",
        "elixir",
    ],
}

# Flatten for easy lookup
ALL_TECH: set[str] = set()
for category in KNOWN_TECH.values():
    ALL_TECH.update(t.lower() for t in category)


def extract_decisions(text: str) -> list[ExtractedPattern]:
    """Extract architecture and design decision statements."""
    patterns = []

    decision_patterns = [
        # Active decisions
        r"(?:decided|chose|went with|settled on|opted for)\s+(.{10,150})",
        # Comparisons
        r"(?:instead of|rather than)\s+(\w+).{0,30}?(?:use|chose|using)\s+(.{5,100})",
        # Trade-offs
        r"(?:trade-?off|compromise)[:.]?\s*(.{10,200})",
        # Deliberate choices
        r"(?:we|I)\s+(?:will|should|need to)\s+(?:use|implement|build)\s+(.{10,100})",
    ]

    for pattern in decision_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Combine all groups (handles multi-group patterns like "instead of X, use Y")
                groups = [g for g in match.groups() if g]
                if groups:
                    content = " → ".join(groups).strip()
                    if 10 < len(content) < 200:
                        patterns.append(
                            ExtractedPattern(content, PatternType.DECISION, confidence=0.8)
                        )
        except re.error:
            continue

    return patterns


def extract_architecture(text: str) -> list[ExtractedPattern]:
    """Extract system architecture descriptions."""
    patterns = []

    arch_patterns = [
        # Component responsibilities
        r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:uses|runs on|is built with|is powered by)\s+(.{5,100})",
        r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:handles|manages|is responsible for)\s+(.{5,100})",
        # Communication patterns
        r"(\w+)\s+(?:communicates|connects|talks)\s+(?:with|to|via)\s+(\w+(?:\s+\w+)?)",
        # Architecture style mentions
        r"(?:uses?|following|implementing)\s+(?:a\s+)?(\w+)\s+(?:architecture|pattern|approach)",
        # Data flow
        r"data\s+(?:flows?|is sent|goes)\s+(?:from|to|through)\s+(.{5,100})",
    ]

    for pattern in arch_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Combine all groups into a meaningful statement
                groups = [g for g in match.groups() if g]
                if groups:
                    content = " ".join(groups).strip()
                    if 10 < len(content) < 200:
                        patterns.append(
                            ExtractedPattern(content, PatternType.ARCHITECTURE, confidence=0.75)
                        )
        except re.error:
            continue

    return patterns


def extract_tech_stack(text: str) -> list[ExtractedPattern]:
    """Extract technology mentions with context.

    Only extracts tech when there's surrounding context to avoid
    over-matching casual mentions.
    """
    patterns = []
    seen: set[str] = set()

    # Build regex pattern from known tech
    tech_pattern = "|".join(re.escape(t) for t in sorted(ALL_TECH, key=len, reverse=True))

    # Context patterns that indicate meaningful tech usage
    context_patterns = [
        rf"(?:uses?|using|built with|powered by|runs? on|based on|written in)\s+({tech_pattern})",
        rf"(?:chose|selected|picked|went with|decided on)\s+({tech_pattern})",  # Decision verbs
        rf"({tech_pattern})\s+(?:handles?|manages?|provides?|supports?|server|client|app)",
        rf"({tech_pattern})\s*(?:v?\d+\.[\d.]+|\d+)",  # With version (optional space)
        rf"(?:the|our|this)\s+({tech_pattern})\s+(?:app|api|server|service|project)",
    ]

    for pattern in context_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                tech = match.group(1).lower()
                if tech not in seen:
                    seen.add(tech)
                    # Get surrounding context (up to 50 chars each side)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    # Clean up context
                    context = " ".join(context.split())
                    if len(context) > 10:
                        patterns.append(
                            ExtractedPattern(context, PatternType.TECH_STACK, confidence=0.85)
                        )
        except re.error:
            continue

    return patterns


def extract_explanations(text: str) -> list[ExtractedPattern]:
    """Extract rationale and reasoning statements."""
    patterns = []

    explanation_patterns = [
        # Because clauses
        r"(.{10,100}?)\s+because\s+(.{10,150})",
        # Purpose clauses
        r"(.{10,100})\s+(?:in order to|so that|which allows|which enables)\s+(.{10,150})",
        # Explicit reasons
        r"(?:the reason (?:is|was)|this is why|that's why)[:\s]+(.{10,200})",
        # Necessity
        r"(?:this|it|we)\s+(?:need|require|must)\s+(.{10,150})\s+(?:to|for|because)",
    ]

    for pattern in explanation_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = [g for g in match.groups() if g]
                if groups:
                    content = " ... ".join(groups).strip()
                    if 20 < len(content) < 300:
                        patterns.append(
                            ExtractedPattern(content, PatternType.EXPLANATION, confidence=0.6)
                        )
        except re.error:
            continue

    return patterns


def extract_insights(text: str) -> list[ExtractedPattern]:
    """Extract key insights and longer contextual explanations.

    Captures substantive paragraphs that contain explanatory language,
    summaries, or important context that would be valuable to remember.

    Targets content like:
    - "The key insight is..." / "The main takeaway..."
    - "This means that..." / "In other words..."
    - "The problem was..." / "The solution is..."
    - Paragraphs with causal language (because, therefore, thus)
    - Summary statements with bullets or numbered lists

    Min length: 100 chars (filters trivial content)
    Max length: 800 chars (prevents noise from huge blocks)
    """
    patterns = []

    # Split into paragraphs (double newline or markdown section breaks)
    paragraphs = re.split(r"\n\s*\n|\n(?=#{1,3}\s)", text)

    # Indicators of valuable contextual content
    insight_indicators = [
        # Direct insight markers
        r"(?:the )?(?:key|main|important|critical) (?:insight|takeaway|point|thing)",
        r"(?:in )?summary",
        r"(?:this|the) means",
        r"in other words",
        r"the (?:problem|issue|challenge) (?:is|was)",
        r"the (?:solution|fix|answer) (?:is|was)",
        r"(?:what|here's what) (?:this|we|you) (?:need|should)",
        # Causal/explanatory language
        r"because of this",
        r"as a result",
        r"therefore",
        r"consequently",
        r"this is why",
        r"the reason (?:is|was|being)",
        # Summary/conclusion markers
        r"to (?:summarize|recap|sum up)",
        r"in conclusion",
        r"the (?:bottom line|upshot)",
        r"(?:essentially|fundamentally|basically),",
        # Lists with context (numbered or bulleted)
        r"(?:here are|the following|these are) (?:the )?(?:\d+|several|some|a few)",
    ]

    # Compile pattern for efficiency
    indicator_pattern = re.compile("|".join(insight_indicators), re.IGNORECASE)

    for para in paragraphs:
        para = para.strip()

        # Skip if too short or too long
        if len(para) < 100 or len(para) > 800:
            continue

        # Skip code blocks (already handled by extract_code_blocks)
        if para.startswith("```") or para.startswith("    "):
            continue

        # Skip if mostly non-text (tables, URLs, etc.)
        alpha_ratio = sum(c.isalpha() for c in para) / max(len(para), 1)
        if alpha_ratio < 0.5:
            continue

        # Check for insight indicators
        if indicator_pattern.search(para):
            # Clean up whitespace
            cleaned = " ".join(para.split())
            # Higher confidence for explicit markers, lower for causal language
            has_explicit_marker = any(
                re.search(p, para, re.IGNORECASE)
                for p in insight_indicators[:8]  # First 8 are explicit
            )
            confidence = 0.75 if has_explicit_marker else 0.6

            patterns.append(ExtractedPattern(cleaned, PatternType.INSIGHT, confidence=confidence))

    return patterns


def extract_config(text: str) -> list[ExtractedPattern]:
    """Extract configuration facts and settings.

    SECURITY: Env var values are NOT extracted to avoid storing secrets.
    Only env var names and non-sensitive config facts are captured.
    """
    patterns = []

    config_patterns = [
        # Defaults statements - only capture numeric/safe values
        # Match "The timeout defaults to 30 seconds"
        r"(?:the\s+)?(\w+)\s+defaults?\s+to\s+(\d+\s*\w*)",
        # Dependencies and requirements (safe - describes relationships not values)
        r"(?:requires|depends on|needs)\s+([A-Za-z][\w\s]{4,50})",
        # Specific settings (safe numeric values only)
        r"(?:port|timeout|limit|max|min|size|threshold)\s+(?:is|=|:)\s*(\d+\w*)",
        # File paths (no env var values)
        r"(?:stored|saved|located|found)\s+(?:in|at)\s+([/~][\w./\-]+)",
    ]

    for pattern in config_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = [g for g in match.groups() if g]
                if groups:
                    content = " ".join(groups).strip()
                    # SECURITY: Skip if content might contain secrets
                    if _may_contain_secrets(content):
                        continue
                    if 3 < len(content) < 150:
                        patterns.append(
                            ExtractedPattern(content, PatternType.CONFIG, confidence=0.65)
                        )
        except re.error:
            continue

    # Extract env var NAMES only (never values) for documentation purposes
    # Pattern: set/export VAR_NAME=... -> only extract "VAR_NAME is configured"
    env_var_pattern = r"(?:set|export)\s+([A-Z_][A-Z0-9_]+)\s*="
    for match in re.finditer(env_var_pattern, text):
        var_name = match.group(1)
        # Skip if it looks sensitive
        if any(sensitive in var_name.lower() for sensitive in SENSITIVE_ENV_NAMES):
            continue
        # Store only the fact that this env var exists, not its value
        patterns.append(
            ExtractedPattern(
                f"{var_name} environment variable is configured",
                PatternType.CONFIG,
                confidence=0.5,  # Lower confidence since we don't have the value
            )
        )

    return patterns


def extract_dependencies(text: str) -> list[ExtractedPattern]:
    """Extract package dependencies with version constraints.

    Captures patterns like:
    - requires python>=3.10
    - uses sqlalchemy==2.0.0
    - dependency: fastapi~=0.100
    """
    patterns = []
    seen: set[str] = set()

    # Package with optional extras and version constraint
    pkg_pattern = r"[\w\-]+(?:\[[\w,]+\])?\s*[~=<>!]+\s*[\d.]+"
    dependency_patterns = [
        # Python-style: requires package>=version
        rf"(?:requires?|needs?|depends on|dependency:?)\s*({pkg_pattern})",
        # requirements.txt style: package==version
        rf"^({pkg_pattern})",
        # pip install style
        rf"pip install[^\n]*?({pkg_pattern})",
        # pyproject.toml style: "package>=version"
        rf'"({pkg_pattern})"',
    ]

    for pattern in dependency_patterns:
        try:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                groups = [g for g in match.groups() if g]
                if groups:
                    content = "".join(groups).strip()
                    # Normalize spacing
                    content = re.sub(r"\s+", "", content)
                    if content not in seen and len(content) > 5:
                        seen.add(content)
                        patterns.append(
                            ExtractedPattern(content, PatternType.DEPENDENCY, confidence=0.85)
                        )
        except re.error:
            continue

    return patterns


def extract_api_endpoints(text: str) -> list[ExtractedPattern]:
    """Extract REST/HTTP API endpoints.

    Captures patterns like:
    - GET /users/{id}
    - @router.post("/data")
    - app.get("/api/v1/items")
    """
    patterns = []
    seen: set[str] = set()

    endpoint_patterns = [
        # HTTP method + path: GET /users, POST /api/data
        # Groups: 1=method, 2=path (not wrapping both in another group)
        r"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+['\"]?(/[\w\-{}/:.?&=]+)['\"]?",
        # FastAPI/Flask decorators: @app.get("/path"), @router.post("/path")
        r"@[\w.]+\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
        # Express.js style: app.get('/path', ...)
        r"app\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
        # OpenAPI/Swagger paths
        r"['\"](/(?:api/)?v\d+/[\w\-{}/]+)['\"]:\s*\{",
    ]

    for pattern in endpoint_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = [g for g in match.groups() if g]
                if not groups:
                    continue

                # Combine method + path or just path
                if len(groups) >= 2:
                    content = f"{groups[0].upper()} {groups[1]}"
                else:
                    content = groups[0]

                if content not in seen and len(content) > 3:
                    seen.add(content)
                    patterns.append(
                        ExtractedPattern(content, PatternType.API_ENDPOINT, confidence=0.9)
                    )
        except re.error:
            continue

    return patterns


# ========== Entity Extractors (for Knowledge Graph Linking) ==========


def extract_tech_entities(text: str) -> list[ExtractedPattern]:
    """Extract technology entities for knowledge graph linking.

    Unlike extract_tech_stack() which captures contextual usage statements,
    this extractor outputs normalized entity patterns designed for linking
    via the MENTIONS relation type.

    Output patterns include:
    - Normalized entity name
    - entity_type='technology' metadata
    - Subcategory (framework, database, tool, language)
    """
    patterns = []
    seen: set[str] = set()

    # Build regex pattern from known tech (longest first to avoid partial matches)
    tech_pattern = "|".join(re.escape(t) for t in sorted(ALL_TECH, key=len, reverse=True))

    # Context patterns that indicate meaningful tech usage (not just casual mention)
    context_patterns = [
        rf"(?:uses?|using|built with|powered by|runs? on|based on|written in)\s+({tech_pattern})",
        rf"(?:chose|selected|picked|went with|decided on|migrated to)\s+({tech_pattern})",
        rf"({tech_pattern})\s+(?:server|client|app|api|service|database|db|cache|components?)",
        rf"(?:the|our|this)\s+({tech_pattern})\s+(?:app|api|server|service|project|backend|frontend)",
        rf"({tech_pattern})\s*(?:v?\d+\.[\d.]+)",  # With version number
        rf"(?:stored|persisted|saved)\s+(?:in|with|using)\s+({tech_pattern})",  # Data storage
        # Role assignment pattern
        rf"({tech_pattern})\s+(?:for the|as the)\s+(?:backend|frontend|database|cache|api)",
    ]

    for pattern in context_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                tech = match.group(1).lower()
                if tech in seen:
                    continue
                seen.add(tech)

                # Determine subcategory
                subcategory = "tool"  # Default
                for cat, items in KNOWN_TECH.items():
                    if tech in [t.lower() for t in items]:
                        subcategory = cat.rstrip("s")  # "frameworks" -> "framework"
                        break

                # Create entity pattern with normalized name
                # Format: "Technology: {name}" for easy recall and linking
                pattern_text = f"Technology: {tech.title()}"

                patterns.append(
                    ExtractedPattern(
                        pattern_text,
                        PatternType.ENTITY_TECHNOLOGY,
                        confidence=0.85,
                        metadata={
                            "entity_type": "technology",
                            "entity_name": tech,
                            "subcategory": subcategory,
                        },
                    )
                )
        except re.error:
            continue

    return patterns


def _build_decision_metadata(
    groups: list[str],
    has_rationale: bool,
    has_alternative: bool,
    is_inverted_pattern: bool,
) -> tuple[str, float, dict[str, Any]]:
    """Build metadata for a decision entity pattern.

    Returns:
        Tuple of (pattern_text, confidence, metadata).
    """
    if has_alternative and len(groups) >= 2:
        # Handle "instead of Y, use X" vs "chose X instead of Y"
        if is_inverted_pattern:
            alternative, decision = groups[0], groups[1]
        else:
            decision, alternative = groups[0], groups[1]
        return (
            f"Decision: {decision} (over {alternative})",
            0.8,
            {
                "entity_type": "decision",
                "decision": decision,
                "alternative": alternative,
                "has_rationale": False,
            },
        )

    if has_rationale and len(groups) >= 2:
        decision, rationale = groups[0], groups[1]
        return (
            f"Decision: {decision} (reason: {rationale})",
            0.9,
            {
                "entity_type": "decision",
                "decision": decision,
                "rationale": rationale,
                "has_rationale": True,
            },
        )

    # Simple decision
    decision = groups[0]
    return (
        f"Decision: {decision}",
        0.7,
        {
            "entity_type": "decision",
            "decision": decision,
            "has_rationale": False,
        },
    )


def extract_decision_entities(text: str) -> list[ExtractedPattern]:
    """Extract decision entities for knowledge graph linking.

    Unlike extract_decisions() which captures contextual decision statements,
    this extractor outputs normalized entity patterns designed for linking
    via the MENTIONS relation type.

    Output patterns include:
    - Normalized decision summary
    - entity_type='decision' metadata
    - has_rationale flag for higher confidence
    - alternatives if "instead of"/"rather than" present
    """
    patterns = []
    seen_decisions: set[str] = set()

    # Pattern groups: (pattern, has_rationale, has_alternative, is_inverted)
    # Order matters - more specific patterns first
    decision_patterns = [
        # Decision with alternative considered (check these first)
        (
            r"(?:decided on|chose|went with)\s+(.{5,60}?)\s+(?:instead of|rather than|over)"
            r"\s+([^.]{5,60}?)(?:\s+for\s+|$|\.)",
            False,
            True,
            False,
        ),
        (
            r"(?:instead of|rather than)\s+(.{5,60}?)[,.]?\s*(?:we |I )?"
            r"(?:use|chose|using|went with)\s+(.{5,60})",
            False,
            True,
            True,
        ),
        # Decision with rationale (higher confidence)
        (
            r"(?:we |I )?(?:decided|chose|went with|opted for)\s+(.{5,80}?)"
            r"\s+because(?:\s+of)?\s+(.{10,100})",
            True,
            False,
            False,
        ),
        (
            r"(?:we |I )?(?:decided|chose|went with|opted for)\s+(.{5,80}?)"
            r"\s+since\s+(.{10,100})",
            True,
            False,
            False,
        ),
        (
            r"(?:we |I )?(?:decided|chose|went with|opted for)\s+(.{5,80}?)" r"\s+for\s+(.{10,60})",
            True,
            False,
            False,
        ),
        # Simple decisions (lower confidence)
        (
            r"(?:we |I )?(?:decided to|chose to|going with|opted to)" r"\s+(.{10,80})",
            False,
            False,
            False,
        ),
        (r"(?:the )?decision (?:was |is )?to\s+(.{10,80})", False, False, False),
    ]

    for pattern, has_rationale, has_alternative, is_inverted in decision_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = [g.strip() for g in match.groups() if g]
                if not groups:
                    continue

                pattern_text, confidence, metadata = _build_decision_metadata(
                    groups, has_rationale, has_alternative, is_inverted
                )

                # Deduplicate by decision key
                decision_key = metadata["decision"].lower()[:50]
                if decision_key in seen_decisions:
                    continue
                seen_decisions.add(decision_key)

                patterns.append(
                    ExtractedPattern(
                        pattern_text,
                        PatternType.ENTITY_DECISION,
                        confidence=confidence,
                        metadata=metadata,
                    )
                )
        except re.error:
            continue

    return patterns


# ========== Main Mining Function ==========


PATTERN_EXTRACTORS = [
    # Original extractors
    extract_imports,
    extract_facts,
    extract_commands,
    extract_code_patterns,
    extract_code_blocks,
    # Enhanced regex extractors
    extract_decisions,
    extract_architecture,
    extract_tech_stack,
    extract_explanations,
    extract_insights,  # Long-form contextual content
    extract_config,
    # High-value extractors
    extract_dependencies,
    extract_api_endpoints,
    # Entity extractors (for knowledge graph linking)
    extract_tech_entities,
    extract_decision_entities,
]


def extract_patterns(
    text: str, ner_enabled: bool = True, ner_confidence: float = 0.7
) -> list[ExtractedPattern]:
    """Extract all patterns from text, deduplicated by pattern content.

    Args:
        text: Text to extract patterns from.
        ner_enabled: Whether to run NER entity extraction.
        ner_confidence: Minimum confidence for NER entity extraction.
    """
    # Run all regex extractors
    all_patterns = [pattern for extractor in PATTERN_EXTRACTORS for pattern in extractor(text)]

    # Run NER extractor if enabled
    if ner_enabled:
        all_patterns.extend(extract_entities_ner(text, min_confidence=ner_confidence))

    # Deduplicate while preserving order (first occurrence wins)
    seen: dict[str, ExtractedPattern] = {}
    for p in all_patterns:
        if p.pattern not in seen:
            seen[p.pattern] = p

    return list(seen.values())


def _group_entities_by_source(
    entity_memories: list[tuple[int, str, int | None]],
) -> dict[int, list[tuple[int, str]]]:
    """Group entity memories by their source_log_id."""
    entities_by_source: dict[int, list[tuple[int, str]]] = {}
    for memory_id, pattern_type, source_log_id in entity_memories:
        if source_log_id:
            entities_by_source.setdefault(source_log_id, []).append((memory_id, pattern_type))
    return entities_by_source


def _create_entity_links(
    storage: Storage,
    entity_memories: list[tuple[int, str, int | None]],
) -> int:
    """Create knowledge graph links for extracted entities.

    For each entity memory:
    1. Find other memories from the same source_log_id
    2. Create source_memory -[MENTIONS]-> entity_memory links
    3. Link related entities (e.g., decision -[DEPENDS_ON]-> technology if decision involves tech)

    Args:
        storage: Storage instance.
        entity_memories: List of (memory_id, pattern_type, source_log_id) tuples.

    Returns:
        Number of links created.
    """
    from memory_mcp.storage import RelationType

    def try_link(from_id: int, to_id: int, rel_type: RelationType) -> int:
        """Attempt to create a link, returning 1 if successful, 0 otherwise."""
        return 1 if storage.link_memories(from_id, to_id, rel_type) else 0

    links_created = 0
    entities_by_source = _group_entities_by_source(entity_memories)

    for source_log_id, entities in entities_by_source.items():
        source_memories = storage.get_memories_by_source_log(source_log_id)
        entity_ids = {mem_id for mem_id, _ in entities}
        non_entity_memories = [m for m in source_memories if m.id not in entity_ids]

        # Create MENTIONS links from non-entity memories to entities
        for entity_id, _ in entities:
            for source_mem in non_entity_memories:
                links_created += try_link(source_mem.id, entity_id, RelationType.MENTIONS)

        # Link decisions to technologies (decision -[DEPENDS_ON]-> technology)
        tech_ids = [m_id for m_id, pt in entities if pt == "entity_technology"]
        decision_ids = [m_id for m_id, pt in entities if pt == "entity_decision"]

        for decision_id in decision_ids:
            for tech_id in tech_ids:
                links_created += try_link(decision_id, tech_id, RelationType.DEPENDS_ON)

    return links_created


def _get_memory_type_for_pattern(pattern_type: str) -> MemoryType:
    """Map pattern type to memory type."""
    if pattern_type == "fact":
        return MemoryType.PROJECT
    if pattern_type == "command":
        return MemoryType.REFERENCE
    return MemoryType.PATTERN


def run_mining(storage: Storage, hours: int = 24, project_id: str | None = None) -> dict:
    """Run pattern mining on recent outputs.

    Args:
        storage: Storage instance.
        hours: How many hours of logs to process.
        project_id: If provided, only mine logs from this project.
                    This prevents cross-project pattern leakage.

    Returns statistics about patterns found and stored.

    Patterns are stored as memories immediately when they meet the minimum
    confidence threshold. Hot cache promotion happens separately when patterns
    reach the occurrence threshold.

    Project Attribution:
        Each mined memory inherits the project_id from its source log, not the
        current session. This prevents cross-project pollution when mining logs
        from multiple projects.
    """
    from memory_mcp.storage import MemorySource

    outputs = storage.get_recent_outputs(hours=hours, project_id=project_id)
    settings = storage.settings

    total_patterns = 0
    new_memories = 0
    updated_patterns = 0
    promoted_to_hot = 0

    # Track entity memories for cross-linking
    entity_memories: list[
        tuple[int, str, int | None]
    ] = []  # (memory_id, pattern_type, source_log_id)

    for log_id, content, _, log_project_id in outputs:
        patterns = extract_patterns(
            content,
            ner_enabled=settings.ner_enabled,
            ner_confidence=settings.ner_confidence_threshold,
        )
        total_patterns += len(patterns)

        for pattern in patterns:
            hash_val = content_hash(pattern.pattern)
            is_existing = storage.mined_pattern_exists(hash_val)

            # SECURITY: Skip patterns that might contain sensitive data
            if _may_contain_secrets(pattern.pattern):
                continue

            # Skip short fragments (too short to be useful knowledge)
            if len(pattern.pattern) < settings.mining_min_pattern_length:
                continue

            if is_existing:
                # Update occurrence count in mined_patterns table
                updated_patterns += 1
                storage.upsert_mined_pattern(
                    pattern.pattern,
                    pattern.pattern_type.value,
                    source_log_id=log_id,
                    confidence=pattern.confidence,
                )
            else:
                # New pattern - store as memory immediately if confidence is sufficient
                # Skip low-value categories (command, snippet) - they go to mined_patterns only
                created_memory_id = None
                skip_memory_storage = pattern.pattern_type.value in ("command", "snippet")

                if (
                    not skip_memory_storage
                    and pattern.confidence >= settings.mining_auto_approve_confidence
                ):
                    mem_type = _get_memory_type_for_pattern(pattern.pattern_type.value)

                    # Use project_id from source log, not current session
                    # This prevents cross-project pollution
                    memory_id, is_new = storage.store_memory(
                        content=pattern.pattern,
                        memory_type=mem_type,
                        source=MemorySource.MINED,
                        tags=["mined"],
                        project_id=log_project_id,
                        source_log_id=log_id,
                    )

                    if is_new:
                        new_memories += 1
                        created_memory_id = memory_id

                    # Track entity patterns for knowledge graph linking
                    # Note: Track even when is_new=False (merged with existing) - the memory
                    # exists and should be linked to other memories from the same output log
                    if pattern.pattern_type.value in (
                        "entity_technology",
                        "entity_decision",
                    ):
                        entity_memories.append((memory_id, pattern.pattern_type.value, log_id))

                # Track in mined_patterns for occurrence counting
                pattern_id = storage.upsert_mined_pattern(
                    pattern.pattern,
                    pattern.pattern_type.value,
                    source_log_id=log_id,
                    confidence=pattern.confidence,
                )

                # Link pattern to its memory for exact-match promotion
                if created_memory_id is not None:
                    storage.link_pattern_to_memory(pattern_id, created_memory_id)

    # Promote high-occurrence patterns to hot cache
    if settings.mining_auto_approve_enabled:
        from memory_mcp.storage import PatternStatus

        candidates = storage.get_promotion_candidates(threshold=1, status=PatternStatus.PENDING)
        for candidate in candidates:
            if candidate.occurrence_count < settings.mining_auto_approve_occurrences:
                continue

            # Prefer exact match via linked memory_id, fallback to semantic search
            # (semantic search needed for patterns created before v17 migration)
            memory_id_to_promote = candidate.memory_id
            if memory_id_to_promote is None:
                memories = storage.recall(candidate.pattern, limit=1, threshold=0.95).memories
                memory_id_to_promote = memories[0].id if memories else None

            if memory_id_to_promote is not None and storage.promote_to_hot(memory_id_to_promote):
                promoted_to_hot += 1

    # Create knowledge graph links for entities
    entity_links_created = _create_entity_links(storage, entity_memories)

    return {
        "outputs_processed": len(outputs),
        "patterns_found": total_patterns,
        "new_memories": new_memories,
        "updated_patterns": updated_patterns,
        "promoted_to_hot": promoted_to_hot,
        "entity_links_created": entity_links_created,
    }
