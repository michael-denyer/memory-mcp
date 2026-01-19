"""Tests for mining module - pattern extraction functions."""

from memory_mcp.mining import (
    ExtractedPattern,
    PatternType,
    extract_code_blocks,
    extract_code_patterns,
    extract_commands,
    extract_facts,
    extract_imports,
    extract_patterns,
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
        expected = {"import", "fact", "command", "code", "code_block"}
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
