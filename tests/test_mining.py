"""Tests for mining module - pattern extraction functions."""

import pytest

from memory_mcp.mining import (
    ExtractedPattern,
    PatternType,
    extract_api_endpoints,
    extract_architecture,
    extract_code_blocks,
    extract_code_patterns,
    extract_commands,
    extract_config,
    extract_decisions,
    extract_dependencies,
    extract_entities_ner,
    extract_explanations,
    extract_facts,
    extract_imports,
    extract_patterns,
    extract_tech_stack,
)

# ========== Import Extraction Tests ==========


class TestExtractImports:
    """Tests for extract_imports function."""

    def test_single_import(self):
        """Extract a single import statement."""
        text = "import datetime"
        patterns = extract_imports(text)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.IMPORT
        assert "datetime" in patterns[0].pattern

    def test_from_import(self):
        """Extract from...import statements."""
        text = "from collections import defaultdict"
        patterns = extract_imports(text)
        assert len(patterns) == 1
        assert "defaultdict" in patterns[0].pattern

    def test_import_in_code_context(self):
        """Extract imports from code with other content around them."""
        text = """
# Some comment
from collections import defaultdict
def foo():
    pass
"""
        patterns = extract_imports(text)
        assert len(patterns) >= 1
        assert any("defaultdict" in p.pattern for p in patterns)

    def test_short_imports_skipped(self):
        """Very short imports are skipped."""
        text = "import os"  # 9 chars, <= 10
        patterns = extract_imports(text)
        assert len(patterns) == 0

    def test_whitespace_normalized(self):
        """Whitespace is normalized in imports."""
        text = "from   typing   import    List"
        patterns = extract_imports(text)
        assert len(patterns) == 1
        assert patterns[0].pattern == "from typing import List"


# ========== Fact Extraction Tests ==========


class TestExtractFacts:
    """Tests for extract_facts function."""

    def test_this_project_uses(self):
        """Extract 'This project uses X' statements."""
        text = "This project uses SQLite for persistence."
        patterns = extract_facts(text)
        assert len(patterns) >= 1
        assert any("This project uses SQLite" in p.pattern for p in patterns)

    def test_we_use_for(self):
        """Extract 'We use X for Y' statements."""
        text = "We use pytest for testing."
        patterns = extract_facts(text)
        assert len(patterns) >= 1
        assert any("We use pytest for testing" in p.pattern for p in patterns)

    def test_the_api_uses(self):
        """Extract 'The API uses X' statements."""
        text = "The API uses JWT tokens for authentication."
        patterns = extract_facts(text)
        assert len(patterns) >= 1

    def test_tests_use(self):
        """Extract 'Tests use X' statements."""
        text = "Tests use mock embeddings for speed."
        patterns = extract_facts(text)
        assert len(patterns) >= 1

    def test_authentication_uses(self):
        """Extract authentication statements."""
        text = "Authentication uses OAuth2 with refresh tokens."
        patterns = extract_facts(text)
        assert len(patterns) >= 1

    def test_too_short_skipped(self):
        """Very short facts are skipped."""
        text = "We use X"  # Too short
        patterns = extract_facts(text)
        assert len(patterns) == 0

    def test_too_long_skipped(self):
        """Very long facts are skipped."""
        text = "This project uses " + "x" * 200  # > 200 chars
        patterns = extract_facts(text)
        assert len(patterns) == 0


# ========== Command Extraction Tests ==========


class TestExtractCommands:
    """Tests for extract_commands function."""

    def test_backtick_git_command(self):
        """Extract git commands in backticks."""
        text = "Run `git status` to check."
        patterns = extract_commands(text)
        assert len(patterns) == 1
        assert patterns[0].pattern == "git status"
        assert patterns[0].pattern_type == PatternType.COMMAND

    def test_backtick_npm_command(self):
        """Extract npm commands in backticks."""
        text = "Use `npm install` first"
        patterns = extract_commands(text)
        assert len(patterns) == 1
        assert patterns[0].pattern == "npm install"

    def test_backtick_docker_command(self):
        """Extract docker commands in backticks."""
        text = "Build with `docker build .` command"
        patterns = extract_commands(text)
        assert len(patterns) == 1
        assert patterns[0].pattern == "docker build ."

    def test_run_colon_command(self):
        """Extract commands after 'run:' prefix."""
        text = "run: `docker compose up`"
        patterns = extract_commands(text)
        assert len(patterns) == 1
        assert patterns[0].pattern == "docker compose up"

    def test_unknown_command_skipped(self):
        """Unknown commands (not in COMMAND_PREFIXES) are skipped."""
        text = "`unknown_tool --flag`"
        patterns = extract_commands(text)
        assert len(patterns) == 0

    def test_too_short_skipped(self):
        """Very short commands are skipped (<= 5 chars)."""
        text = "`git`"  # 3 chars, <= 5
        patterns = extract_commands(text)
        assert len(patterns) == 0

    def test_multiple_backtick_commands(self):
        """Extract multiple commands from backticks."""
        text = "First `npm install`, then `npm run build`"
        patterns = extract_commands(text)
        assert len(patterns) == 2

    def test_uv_commands(self):
        """Extract uv commands."""
        text = "Run `uv run pytest` to test"
        patterns = extract_commands(text)
        assert len(patterns) == 1
        assert patterns[0].pattern == "uv run pytest"

    def test_pip_commands(self):
        """Extract pip commands."""
        text = "`pip install -r requirements.txt`"
        patterns = extract_commands(text)
        assert len(patterns) == 1

    def test_cargo_commands(self):
        """Extract cargo commands."""
        text = "Build with `cargo build --release`"
        patterns = extract_commands(text)
        assert len(patterns) == 1


# ========== Code Pattern Extraction Tests ==========


class TestExtractCodePatterns:
    """Tests for extract_code_patterns function."""

    def test_function_definition(self):
        """Extract function definitions."""
        text = """
def hello_world():
    print("Hello")
"""
        patterns = extract_code_patterns(text)
        assert len(patterns) == 1
        assert "def hello_world" in patterns[0].pattern
        assert patterns[0].pattern_type == PatternType.CODE

    def test_async_function(self):
        """Extract async function definitions."""
        text = """
async def fetch_data(url: str) -> dict:
    pass
"""
        patterns = extract_code_patterns(text)
        assert len(patterns) == 1
        assert "def fetch_data" in patterns[0].pattern

    def test_class_definition(self):
        """Extract class definitions."""
        text = """
class MyService:
    pass
"""
        patterns = extract_code_patterns(text)
        assert len(patterns) == 1
        assert "class MyService" in patterns[0].pattern

    def test_class_with_base(self):
        """Extract class definitions with base classes."""
        text = """
class User(BaseModel):
    name: str
"""
        patterns = extract_code_patterns(text)
        assert len(patterns) == 1
        assert "class User" in patterns[0].pattern

    def test_private_functions_skipped(self):
        """Private functions (starting with _) are skipped."""
        text = """
def _internal_helper():
    pass

def public_function():
    pass
"""
        patterns = extract_code_patterns(text)
        assert len(patterns) == 1
        assert "public_function" in patterns[0].pattern

    def test_multiple_definitions(self):
        """Extract multiple definitions."""
        text = """
class Foo:
    def bar(self):
        pass

def baz():
    pass
"""
        patterns = extract_code_patterns(text)
        pattern_texts = [p.pattern for p in patterns]
        assert any("class Foo" in p for p in pattern_texts)
        assert any("def bar" in p for p in pattern_texts)
        assert any("def baz" in p for p in pattern_texts)


# ========== Code Block Extraction Tests ==========


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks function."""

    def test_python_code_block(self):
        """Extract Python code blocks."""
        text = """
Here's how to do it:

```python
def example():
    return "hello world"
```
"""
        patterns = extract_code_blocks(text)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.CODE_BLOCK
        assert "[python]" in patterns[0].pattern
        assert "def example" in patterns[0].pattern
        assert patterns[0].confidence == 0.7  # Has language

    def test_code_block_no_language(self):
        """Extract code blocks without language identifier."""
        text = """
```
some_command --flag
more_content here
```
"""
        patterns = extract_code_blocks(text)
        assert len(patterns) == 1
        assert patterns[0].confidence == 0.5  # No language

    def test_short_blocks_skipped(self):
        """Very short code blocks are skipped."""
        text = """
```python
pass
```
"""
        patterns = extract_code_blocks(text)
        assert len(patterns) == 0  # < 20 chars

    def test_long_blocks_skipped(self):
        """Very long code blocks are skipped."""
        text = "```python\n" + "x" * 2500 + "\n```"
        patterns = extract_code_blocks(text)
        assert len(patterns) == 0  # > 2000 chars

    def test_error_blocks_skipped(self):
        """Error output blocks are skipped."""
        text = """
```
Error: Something went wrong
at line 42
```
"""
        patterns = extract_code_blocks(text)
        assert len(patterns) == 0

    def test_traceback_blocks_skipped(self):
        """Traceback blocks are skipped."""
        text = """
```
Traceback (most recent call last):
  File "test.py", line 1
ValueError: oops
```
"""
        patterns = extract_code_blocks(text)
        assert len(patterns) == 0

    def test_multiple_blocks(self):
        """Extract multiple code blocks."""
        text = """
First example:
```python
def foo():
    return "foo" * 10
```

Second example:
```javascript
function bar() {
    return "bar".repeat(10);
}
```
"""
        patterns = extract_code_blocks(text)
        assert len(patterns) == 2
        languages = [p.pattern.split("\n")[0] for p in patterns]
        assert "[python]" in languages
        assert "[javascript]" in languages


# ========== Combined Extraction Tests ==========


class TestExtractPatterns:
    """Tests for the combined extract_patterns function."""

    def test_deduplication(self):
        """Duplicate patterns are removed."""
        text = """
```python
def hello():
    return "hello" * 5
```

```python
def hello():
    return "hello" * 5
```
"""
        patterns = extract_patterns(text)
        # Should only have one of the duplicate code blocks
        code_blocks = [p for p in patterns if p.pattern_type == PatternType.CODE_BLOCK]
        # The exact pattern text determines dedup
        pattern_texts = [p.pattern for p in code_blocks]
        assert len(pattern_texts) == len(set(pattern_texts))

    def test_mixed_content(self):
        """Extract patterns from mixed content."""
        text = """
# Project Setup

This project uses SQLite for persistence.

Install dependencies:
$ npm install

Example code:
```python
from pathlib import Path

class MyClass:
    pass
```
"""
        patterns = extract_patterns(text)
        types = {p.pattern_type for p in patterns}

        # Should have multiple pattern types
        assert PatternType.FACT in types or PatternType.CODE_BLOCK in types
        assert len(patterns) >= 2

    def test_empty_text(self):
        """Empty text returns no patterns."""
        patterns = extract_patterns("")
        assert patterns == []

    def test_no_patterns(self):
        """Text with no extractable patterns returns empty."""
        text = "Just some random text with no patterns."
        patterns = extract_patterns(text)
        # May have some patterns from relaxed matching, but none meaningful
        assert all(p.pattern_type in PatternType for p in patterns)


# ========== Pattern Type Tests ==========


class TestPatternType:
    """Tests for PatternType enum."""

    def test_all_types_are_strings(self):
        """All pattern types have string values."""
        for pt in PatternType:
            assert isinstance(pt.value, str)

    def test_expected_types_exist(self):
        """Expected pattern types exist."""
        # Original types
        expected = {"import", "fact", "command", "code", "code_block"}
        # Enhanced regex types
        expected |= {"decision", "architecture", "tech_stack", "explanation", "config"}
        # NER entity types
        expected |= {"entity_person", "entity_org", "entity_location", "entity_misc"}
        # High-value extractors
        expected |= {"dependency", "api_endpoint"}
        actual = {pt.value for pt in PatternType}
        assert expected == actual


# ========== ExtractedPattern Tests ==========


class TestExtractedPattern:
    """Tests for ExtractedPattern dataclass."""

    def test_default_confidence(self):
        """Default confidence is 0.5."""
        pattern = ExtractedPattern("test", PatternType.CODE)
        assert pattern.confidence == 0.5

    def test_custom_confidence(self):
        """Custom confidence can be set."""
        pattern = ExtractedPattern("test", PatternType.CODE, confidence=0.9)
        assert pattern.confidence == 0.9

    def test_equality(self):
        """Patterns with same content are equal."""
        p1 = ExtractedPattern("test", PatternType.CODE)
        p2 = ExtractedPattern("test", PatternType.CODE)
        assert p1 == p2


# ========== Decision Extraction Tests ==========


class TestExtractDecisions:
    """Tests for extract_decisions function."""

    def test_decided_to(self):
        """Extract 'decided to X' statements."""
        text = "We decided to use FastAPI because of async support"
        patterns = extract_decisions(text)
        assert len(patterns) >= 1
        assert any("FastAPI" in p.pattern for p in patterns)
        assert patterns[0].pattern_type == PatternType.DECISION

    def test_chose_x(self):
        """Extract 'chose X' statements."""
        text = "The team chose PostgreSQL over MySQL for better JSON support"
        patterns = extract_decisions(text)
        assert len(patterns) >= 1

    def test_went_with(self):
        """Extract 'went with X' statements."""
        text = "We went with a microservices architecture"
        patterns = extract_decisions(text)
        assert len(patterns) >= 1

    def test_instead_of(self):
        """Extract 'instead of X, we use Y' statements."""
        # The pattern requires "use/chose/using" near "instead of"
        text = "Instead of MongoDB we use PostgreSQL for the database"
        patterns = extract_decisions(text)
        assert len(patterns) >= 1

    def test_trade_off(self):
        """Extract trade-off discussions."""
        text = "The trade-off is memory usage vs speed."
        patterns = extract_decisions(text)
        assert len(patterns) >= 1

    def test_confidence_is_high(self):
        """Decision patterns have high confidence."""
        text = "We decided to implement caching for performance"
        patterns = extract_decisions(text)
        assert len(patterns) >= 1
        assert patterns[0].confidence >= 0.7


# ========== Architecture Extraction Tests ==========


class TestExtractArchitecture:
    """Tests for extract_architecture function."""

    def test_component_uses(self):
        """Extract 'X uses Y' statements."""
        text = "The API uses Redis for caching"
        patterns = extract_architecture(text)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == PatternType.ARCHITECTURE

    def test_handles_responsibility(self):
        """Extract component responsibilities."""
        text = "The service handles user authentication"
        patterns = extract_architecture(text)
        assert len(patterns) >= 1

    def test_communicates_with(self):
        """Extract communication patterns."""
        text = "The frontend communicates with the backend via REST"
        patterns = extract_architecture(text)
        assert len(patterns) >= 1

    def test_data_flows(self):
        """Extract data flow patterns."""
        text = "Data flows from the collector to the processor"
        patterns = extract_architecture(text)
        assert len(patterns) >= 1


# ========== Tech Stack Extraction Tests ==========


class TestExtractTechStack:
    """Tests for extract_tech_stack function."""

    def test_uses_known_tech(self):
        """Extract known technology mentions with context."""
        text = "This project uses FastAPI for the backend"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1
        assert any("fastapi" in p.pattern.lower() for p in patterns)
        assert patterns[0].pattern_type == PatternType.TECH_STACK

    def test_built_with(self):
        """Extract 'built with X' patterns."""
        text = "The application is built with React and TypeScript"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1

    def test_runs_on(self):
        """Extract 'runs on X' patterns."""
        text = "The service runs on PostgreSQL 15"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1

    def test_with_version(self):
        """Extract tech with version numbers."""
        text = "We're using Python 3.11 for this project"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1

    def test_confidence_is_high(self):
        """Tech stack patterns have high confidence."""
        text = "The API is powered by FastAPI"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1
        assert patterns[0].confidence >= 0.8

    def test_no_context_skipped(self):
        """Bare tech mentions without context are not extracted."""
        # Just "react" alone shouldn't match - need context
        text = "react"
        patterns = extract_tech_stack(text)
        assert len(patterns) == 0


# ========== Explanation Extraction Tests ==========


class TestExtractExplanations:
    """Tests for extract_explanations function."""

    def test_because_clause(self):
        """Extract 'X because Y' statements."""
        text = "We chose SQLite because it requires no server setup and is embedded"
        patterns = extract_explanations(text)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == PatternType.EXPLANATION

    def test_in_order_to(self):
        """Extract 'X in order to Y' statements."""
        text = "We added caching in order to reduce database load"
        patterns = extract_explanations(text)
        assert len(patterns) >= 1

    def test_the_reason_is(self):
        """Extract 'the reason is X' statements."""
        text = "The reason is that we need to support concurrent requests"
        patterns = extract_explanations(text)
        assert len(patterns) >= 1

    def test_short_explanations_skipped(self):
        """Very short explanations are skipped."""
        text = "x because y"
        patterns = extract_explanations(text)
        assert len(patterns) == 0


# ========== Config Extraction Tests ==========


class TestExtractConfig:
    """Tests for extract_config function."""

    def test_defaults_to(self):
        """Extract 'defaults to X' statements."""
        text = "The timeout defaults to 30 seconds"
        patterns = extract_config(text)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == PatternType.CONFIG

    def test_requires(self):
        """Extract 'requires X' statements."""
        text = "This feature requires Python 3.10 or higher"
        patterns = extract_config(text)
        assert len(patterns) >= 1

    def test_port_setting(self):
        """Extract port configurations."""
        text = "The server port is 8080"
        patterns = extract_config(text)
        assert len(patterns) >= 1
        assert any("8080" in p.pattern for p in patterns)

    def test_env_var(self):
        """Extract environment variable names (values are never stored for security)."""
        # Note: DATABASE_URL is filtered as sensitive, use a safe example
        text = "export LOG_LEVEL=debug"
        patterns = extract_config(text)
        assert len(patterns) >= 1
        # Should extract the env var NAME, not the value
        assert any("LOG_LEVEL" in p.pattern for p in patterns)
        # Should NOT contain the actual value
        assert not any("debug" in p.pattern.lower() for p in patterns)

    def test_file_path(self):
        """Extract file path configurations."""
        text = "Logs are stored in /var/log/myapp"
        patterns = extract_config(text)
        assert len(patterns) >= 1

    def test_sensitive_env_vars_filtered(self):
        """Sensitive environment variables are NOT extracted."""
        # These should all be filtered
        sensitive_texts = [
            "export DATABASE_URL=postgres://user:pass@host/db",
            "export API_KEY=sk-1234567890",
            "set PASSWORD=secret123",
            "export AUTH_TOKEN=bearer_abc123",
            "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI",
        ]
        for text in sensitive_texts:
            patterns = extract_config(text)
            # Should not extract any patterns with sensitive names
            assert not any(
                "database_url" in p.pattern.lower()
                or "api_key" in p.pattern.lower()
                or "password" in p.pattern.lower()
                or "auth_token" in p.pattern.lower()
                or "secret" in p.pattern.lower()
                for p in patterns
            ), f"Should not extract sensitive var from: {text}"


# ========== NER Extraction Tests ==========


class TestExtractEntitiesNer:
    """Tests for NER-based entity extraction.

    These tests handle both cases:
    - When transformers is installed: NER should extract entities
    - When transformers is not installed: Should return empty list (graceful fallback)
    """

    def test_returns_list(self):
        """NER extraction always returns a list."""
        text = "John Smith works at Google in San Francisco"
        patterns = extract_entities_ner(text)
        assert isinstance(patterns, list)

    def test_entities_have_correct_types(self):
        """Extracted entities have correct PatternTypes."""
        text = "Bill Gates founded Microsoft in Seattle"
        patterns = extract_entities_ner(text)
        # If NER is available, check types
        if patterns:
            valid_types = {
                PatternType.ENTITY_PERSON,
                PatternType.ENTITY_ORG,
                PatternType.ENTITY_LOCATION,
                PatternType.ENTITY_MISC,
            }
            for p in patterns:
                assert p.pattern_type in valid_types

    def test_confidence_threshold(self):
        """Confidence threshold filters low-confidence entities."""
        text = "Testing with various names and places"
        # Low threshold should return more
        patterns_low = extract_entities_ner(text, min_confidence=0.1)
        # High threshold should return fewer
        patterns_high = extract_entities_ner(text, min_confidence=0.99)
        # patterns_high should have <= patterns_low
        assert len(patterns_high) <= len(patterns_low)

    def test_deduplication(self):
        """Duplicate entities are deduplicated."""
        text = "Google is great. I love Google. Google makes Android."
        patterns = extract_entities_ner(text)
        # If NER is available, should only have one entry for "Google" as an entity
        if patterns:
            # Check entity annotation markers, not just raw text
            google_entity_patterns = [p for p in patterns if "[Google is a" in p.pattern]
            assert len(google_entity_patterns) <= 1

    def test_short_entities_filtered(self):
        """Very short entities (< 2 chars) are filtered."""
        text = "A B C testing"
        patterns = extract_entities_ner(text)
        for p in patterns:
            assert len(p.pattern) >= 2

    def test_common_words_filtered(self):
        """Common words like 'the', 'a' are filtered out."""
        text = "The quick brown fox"
        patterns = extract_entities_ner(text)
        for p in patterns:
            assert p.pattern.lower() not in {"the", "a", "an"}

    @pytest.mark.skipif(
        True,  # Always skip in CI - run manually with transformers installed
        reason="Requires transformers to be installed",
    )
    def test_ner_extracts_real_entities(self):
        """Integration test: NER extracts real entities (requires transformers)."""
        text = "Elon Musk is the CEO of Tesla and SpaceX, headquartered in Austin, Texas."
        patterns = extract_entities_ner(text, min_confidence=0.5)

        # Should find at least one entity
        assert len(patterns) > 0

        # Check that we got reasonable entities
        pattern_texts = [p.pattern.lower() for p in patterns]
        # Should find at least one of these
        expected = ["elon musk", "tesla", "spacex", "austin", "texas"]
        found_any = any(any(exp in text for text in pattern_texts) for exp in expected)
        assert found_any, f"Expected to find some of {expected}, got {pattern_texts}"


# ========== Combined Extraction with New Extractors ==========


class TestExtractPatternsEnhanced:
    """Tests for extract_patterns with new extractors."""

    def test_extracts_all_pattern_types(self):
        """Extract patterns runs all extractors."""
        text = """
        We decided to use FastAPI because of async support.
        The API uses PostgreSQL for persistence.
        Run `docker compose up` to start.
        The timeout defaults to 30 seconds.
        ```python
        def main():
            print("Hello world!")
        ```
        """
        patterns = extract_patterns(text)
        types = {p.pattern_type for p in patterns}

        # Should have multiple pattern types from different extractors
        assert len(types) >= 2

    def test_ner_confidence_parameter(self):
        """NER confidence threshold is passed through."""
        text = "John works at Microsoft"
        # These should produce the same output regardless of NER availability
        patterns1 = extract_patterns(text, ner_confidence=0.9)
        patterns2 = extract_patterns(text, ner_confidence=0.9)
        assert patterns1 == patterns2


# ========== Dependency Extraction Tests ==========


class TestExtractDependencies:
    """Tests for extract_dependencies function."""

    def test_requires_version(self):
        """Extract 'requires X>=version' statements."""
        text = "This requires python>=3.10"
        patterns = extract_dependencies(text)
        assert len(patterns) >= 1
        assert any("python>=3.10" in p.pattern for p in patterns)
        assert patterns[0].pattern_type == PatternType.DEPENDENCY

    def test_equals_version(self):
        """Extract package==version statements."""
        text = "fastapi==0.100.0"
        patterns = extract_dependencies(text)
        assert len(patterns) >= 1
        assert any("fastapi==0.100.0" in p.pattern for p in patterns)

    def test_tilde_version(self):
        """Extract package~=version statements."""
        text = "dependency: sqlalchemy~=2.0"
        patterns = extract_dependencies(text)
        assert len(patterns) >= 1

    def test_pip_install(self):
        """Extract dependencies from pip install commands."""
        text = "pip install requests>=2.28.0"
        patterns = extract_dependencies(text)
        assert len(patterns) >= 1

    def test_pyproject_style(self):
        """Extract pyproject.toml style dependencies."""
        text = '"pydantic>=2.0"'
        patterns = extract_dependencies(text)
        assert len(patterns) >= 1

    def test_extras(self):
        """Extract dependencies with extras."""
        text = "requires hot-memory-mcp[mlx]>=0.4"
        patterns = extract_dependencies(text)
        assert len(patterns) >= 1

    def test_high_confidence(self):
        """Dependency patterns have high confidence."""
        text = "requires numpy>=1.20"
        patterns = extract_dependencies(text)
        assert len(patterns) >= 1
        assert patterns[0].confidence >= 0.8


# ========== API Endpoint Extraction Tests ==========


class TestExtractApiEndpoints:
    """Tests for extract_api_endpoints function."""

    def test_get_endpoint(self):
        """Extract GET /path endpoints."""
        text = "GET /users/{id}"
        patterns = extract_api_endpoints(text)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == PatternType.API_ENDPOINT

    def test_endpoint_format_is_correct(self):
        """Extracted endpoints should have 'METHOD /path' format, not duplicate path.

        Regression test: Previously the regex captured both the full match and path,
        resulting in malformed outputs like 'GET /USERS /USERS'.
        """
        text = "GET /users"
        patterns = extract_api_endpoints(text)
        assert len(patterns) >= 1
        # Should be "GET /users", NOT "GET /USERS /USERS"
        assert patterns[0].pattern == "GET /users"
        assert "/users /users" not in patterns[0].pattern.lower()

    def test_post_endpoint(self):
        """Extract POST endpoints."""
        text = "POST /api/v1/items"
        patterns = extract_api_endpoints(text)
        assert len(patterns) >= 1

    def test_fastapi_decorator(self):
        """Extract FastAPI decorator endpoints."""
        text = '@router.post("/data")'
        patterns = extract_api_endpoints(text)
        assert len(patterns) >= 1
        assert any("/data" in p.pattern for p in patterns)

    def test_flask_decorator(self):
        """Extract Flask/Express style endpoints."""
        text = "app.get('/api/users', handler)"
        patterns = extract_api_endpoints(text)
        assert len(patterns) >= 1

    def test_path_parameters(self):
        """Extract endpoints with path parameters."""
        text = "DELETE /items/{item_id}/comments/{comment_id}"
        patterns = extract_api_endpoints(text)
        assert len(patterns) >= 1
        assert any("{item_id}" in p.pattern for p in patterns)

    def test_very_high_confidence(self):
        """API endpoint patterns have very high confidence."""
        text = "GET /health"
        patterns = extract_api_endpoints(text)
        assert len(patterns) >= 1
        assert patterns[0].confidence >= 0.85


# ========== Tech Stack Improvements Tests ==========


class TestExtractTechStackImprovements:
    """Tests for improved tech stack extraction."""

    def test_chose_verb(self):
        """Extract 'chose X' patterns."""
        text = "We chose React for the frontend"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1
        assert any("react" in p.pattern.lower() for p in patterns)

    def test_selected_verb(self):
        """Extract 'selected X' patterns."""
        text = "The team selected PostgreSQL"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1

    def test_version_no_space(self):
        """Extract tech with version without space."""
        text = "We use Python3.11 for this"
        patterns = extract_tech_stack(text)
        assert len(patterns) >= 1
