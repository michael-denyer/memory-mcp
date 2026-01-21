"""Text parsing utilities for extracting memories from content."""

import re

# Pattern for key-value facts (e.g., "Port: 8080", "timeout = 30s")
_FACT_PATTERN = re.compile(r"^[\w\s-]+[:=]\s*\S+", re.IGNORECASE)


def _is_fact_like(text: str) -> bool:
    """Check if text looks like a key-value fact that should be kept despite short length."""
    return bool(_FACT_PATTERN.match(text.strip()))


def _should_include_chunk(text: str, min_length: int) -> bool:
    """Check if chunk should be included based on length or fact-like structure.

    Chunks pass if they exceed min_length OR are short but look like key-value facts.
    """
    if len(text) > min_length:
        return True
    return len(text) >= 5 and _is_fact_like(text)


def parse_content_into_chunks(
    content: str,
    min_length: int = 10,
    source_name: str | None = None,
) -> list[str]:
    """Parse text content into individual chunks for memory storage.

    Splits content on:
    - Lines starting with '- ' or '* ' (list items)
    - Lines starting with numbers like '1. ' (numbered lists)
    - Double newlines (paragraphs) if no list items found

    Args:
        content: Text content to parse.
        min_length: Minimum chunk length to include (relaxed for fact-like items).
        source_name: Optional source file name to prefix chunks with context.

    Returns:
        List of cleaned text chunks, optionally prefixed with source context.
    """
    chunks = []
    list_pattern = re.compile(r"^[\s]*[-*][\s]+|^[\s]*\d+\.[\s]+", re.MULTILINE)

    def format_chunk(chunk: str) -> str:
        """Add source context prefix if source_name provided."""
        return f"[{source_name}] {chunk}" if source_name else chunk

    if list_pattern.search(content):
        # Has list items - split on them
        items = re.split(r"\n(?=[\s]*[-*][\s]+)|(?=[\s]*\d+\.[\s]+)", content)
        for item in items:
            clean = re.sub(r"^[\s]*[-*\d.]+[\s]+", "", item.strip())
            if clean and _should_include_chunk(clean, min_length):
                chunks.append(format_chunk(clean))
    else:
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)
        for p in paragraphs:
            clean = p.strip()
            if clean and _should_include_chunk(clean, min_length):
                chunks.append(format_chunk(clean))

    return chunks


def parse_markdown_with_context(
    content: str,
    source_name: str | None = None,
    min_length: int = 10,
) -> list[str]:
    """Parse markdown content preserving section context.

    Similar to parse_content_into_chunks but tracks markdown headings
    and includes them as context in chunks.

    Args:
        content: Markdown text content to parse.
        source_name: Optional source file name (e.g., "CLAUDE.md").
        min_length: Minimum chunk length (relaxed for fact-like items).

    Returns:
        List of chunks with section context, e.g.:
        "[CLAUDE.md > Testing] Use pytest for all tests"
    """
    chunks = []
    current_section: str | None = None
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    list_pattern = re.compile(r"^[\s]*[-*][\s]+|^[\s]*\d+\.[\s]+", re.MULTILINE)

    def build_context_prefix() -> str:
        """Build context prefix from source name and current section."""
        parts = []
        if source_name:
            parts.append(source_name)
        if current_section:
            parts.append(current_section)
        if parts:
            return f"[{' > '.join(parts)}] "
        return ""

    def add_chunk(text: str) -> None:
        """Add chunk with context prefix if it passes inclusion criteria."""
        clean = text.strip()
        if clean and _should_include_chunk(clean, min_length):
            chunks.append(f"{build_context_prefix()}{clean}")

    # Split content into lines for heading tracking
    lines = content.split("\n")
    current_block: list[str] = []

    for line in lines:
        # Check if this is a heading
        heading_match = heading_pattern.match(line)
        if heading_match:
            # Flush any accumulated block before heading change
            if current_block:
                block_text = "\n".join(current_block).strip()
                if block_text:
                    _process_block(block_text, list_pattern, add_chunk)
                current_block = []

            # Update current section (use the heading text)
            current_section = heading_match.group(2).strip()
            continue

        # Check for list item start (process previous block, start new)
        if list_pattern.match(line):
            # Flush previous non-list content
            if current_block and not list_pattern.match(current_block[0]):
                block_text = "\n".join(current_block).strip()
                if block_text:
                    _process_block(block_text, list_pattern, add_chunk)
                current_block = []

        # Check for paragraph break (double newline effect via empty line)
        if not line.strip() and current_block:
            block_text = "\n".join(current_block).strip()
            if block_text:
                _process_block(block_text, list_pattern, add_chunk)
            current_block = []
            continue

        # Accumulate non-empty lines
        if line.strip():
            current_block.append(line)

    # Don't forget the last block
    if current_block:
        block_text = "\n".join(current_block).strip()
        if block_text:
            _process_block(block_text, list_pattern, add_chunk)

    return chunks


def _process_block(block: str, list_pattern: re.Pattern, add_chunk_fn) -> None:
    """Process a text block, splitting on list items if present."""
    if list_pattern.search(block):
        # Split on list items
        items = re.split(r"\n(?=[\s]*[-*][\s]+)|(?=[\s]*\d+\.[\s]+)", block)
        for item in items:
            clean = re.sub(r"^[\s]*[-*\d.]+[\s]+", "", item.strip())
            if clean:
                add_chunk_fn(clean)
    else:
        # Single paragraph
        add_chunk_fn(block)
