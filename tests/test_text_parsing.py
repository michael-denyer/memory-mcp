"""Tests for text parsing utilities."""

from memory_mcp.text_parsing import (
    parse_content_into_chunks,
    parse_markdown_with_context,
)


class TestParseContentIntoChunks:
    """Tests for the basic chunk parser."""

    def test_list_items(self):
        """Parse bullet list items."""
        content = """
- First item here
- Second item here
- Third item here
"""
        chunks = parse_content_into_chunks(content)
        assert len(chunks) == 3
        assert "First item here" in chunks[0]
        assert "Second item here" in chunks[1]
        assert "Third item here" in chunks[2]

    def test_numbered_list(self):
        """Parse numbered list items."""
        content = """
1. First numbered item
2. Second numbered item
3. Third numbered item
"""
        chunks = parse_content_into_chunks(content)
        assert len(chunks) == 3
        assert "First numbered item" in chunks[0]

    def test_paragraphs(self):
        """Parse paragraphs when no list items present."""
        content = """This is the first paragraph with enough text.

This is the second paragraph with enough text.

This is the third paragraph with enough text."""
        chunks = parse_content_into_chunks(content)
        assert len(chunks) == 3
        assert "first paragraph" in chunks[0]

    def test_min_length_filtering(self):
        """Short chunks are filtered by default."""
        content = """
- Short
- This one is long enough to include
- Tiny
"""
        chunks = parse_content_into_chunks(content)
        # Only the long one should be included
        assert len(chunks) == 1
        assert "long enough" in chunks[0]

    def test_fact_like_short_items_kept(self):
        """Short items that look like facts (key: value) are kept."""
        content = """
- Port: 8080
- timeout = 30
- This is a longer description that exceeds min length
"""
        chunks = parse_content_into_chunks(content)
        # All three should be kept - two are facts, one is long enough
        assert len(chunks) == 3
        assert any("Port: 8080" in c for c in chunks)
        assert any("timeout = 30" in c for c in chunks)

    def test_source_name_prefix(self):
        """Source name is prefixed to chunks when provided."""
        content = "- This is a test item with enough content"
        chunks = parse_content_into_chunks(content, source_name="README.md")
        assert len(chunks) == 1
        assert chunks[0].startswith("[README.md]")

    def test_no_source_name_no_prefix(self):
        """No prefix when source_name not provided."""
        content = "- This is a test item with enough content"
        chunks = parse_content_into_chunks(content)
        assert len(chunks) == 1
        assert not chunks[0].startswith("[")


class TestParseMarkdownWithContext:
    """Tests for markdown-aware parsing with section context."""

    def test_section_context_preserved(self):
        """Chunks include section heading context."""
        content = """# Main Title

Introduction text here is important.

## Getting Started

This is how to get started with the project.

## Configuration

Port: 8080
"""
        chunks = parse_markdown_with_context(content, source_name="README.md")

        # Should have chunks from each section
        assert len(chunks) >= 3

        # Check context is preserved
        intro_chunk = next((c for c in chunks if "Introduction" in c), None)
        assert intro_chunk is not None
        assert "[README.md > Main Title]" in intro_chunk

        started_chunk = next((c for c in chunks if "get started" in c), None)
        assert started_chunk is not None
        assert "[README.md > Getting Started]" in started_chunk

        config_chunk = next((c for c in chunks if "Port: 8080" in c), None)
        assert config_chunk is not None
        assert "[README.md > Configuration]" in config_chunk

    def test_nested_headings(self):
        """Nested headings update context correctly."""
        content = """# Project

## Features

### Feature A

This describes feature A in detail.

### Feature B

This describes feature B in detail.
"""
        chunks = parse_markdown_with_context(content, source_name="DOC.md")

        feature_a = next((c for c in chunks if "feature A" in c), None)
        assert feature_a is not None
        # Should have the most recent heading
        assert "[DOC.md > Feature A]" in feature_a

        feature_b = next((c for c in chunks if "feature B" in c), None)
        assert feature_b is not None
        assert "[DOC.md > Feature B]" in feature_b

    def test_list_items_with_context(self):
        """List items in sections get section context."""
        content = """## Testing

- Use pytest for all tests
- Run with `uv run pytest`
- Check coverage reports
"""
        chunks = parse_markdown_with_context(content, source_name="CLAUDE.md")

        # All list items should have Testing section context
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "[CLAUDE.md > Testing]" in chunk

    def test_no_source_name(self):
        """Works without source name, just section context."""
        content = """## Config

timeout = 30
"""
        chunks = parse_markdown_with_context(content)
        assert len(chunks) >= 1
        assert "[Config]" in chunks[0]
        assert "timeout = 30" in chunks[0]

    def test_short_facts_preserved(self):
        """Short fact-like items are kept in markdown parsing."""
        # With list markers, each becomes its own chunk
        content = """## Settings

- Port: 8080
- Max: 100
- Enabled: true
"""
        chunks = parse_markdown_with_context(content)
        # Each list item should be a separate chunk
        assert len(chunks) == 3
        assert any("Port: 8080" in c for c in chunks)
        assert any("Max: 100" in c for c in chunks)
        assert any("Enabled: true" in c for c in chunks)

    def test_consecutive_facts_grouped(self):
        """Consecutive lines without markers are grouped into one chunk."""
        content = """## Settings

Port: 8080
Max: 100
Enabled: true
"""
        chunks = parse_markdown_with_context(content)
        # Consecutive lines become one chunk
        assert len(chunks) == 1
        assert "Port: 8080" in chunks[0]
        assert "Max: 100" in chunks[0]
        assert "Enabled: true" in chunks[0]

    def test_empty_sections_skipped(self):
        """Empty sections don't produce chunks."""
        content = """## Empty Section

## Filled Section

This section has content that should be included.
"""
        chunks = parse_markdown_with_context(content)
        # Only the filled section should produce a chunk
        assert len(chunks) == 1
        assert "Filled Section" in chunks[0]


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_content(self):
        """Empty content returns empty list."""
        assert parse_content_into_chunks("") == []
        assert parse_markdown_with_context("") == []

    def test_whitespace_only(self):
        """Whitespace-only content returns empty list."""
        assert parse_content_into_chunks("   \n\n   ") == []
        assert parse_markdown_with_context("   \n\n   ") == []

    def test_heading_only(self):
        """Heading with no content returns empty list."""
        content = "# Just a heading"
        chunks = parse_markdown_with_context(content)
        assert chunks == []

    def test_mixed_list_styles(self):
        """Handles mixed list styles."""
        content = """
- Bullet item one
* Asterisk item two
1. Numbered item three
"""
        chunks = parse_content_into_chunks(content)
        assert len(chunks) == 3
