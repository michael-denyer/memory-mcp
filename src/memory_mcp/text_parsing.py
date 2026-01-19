"""Text parsing utilities for extracting memories from content."""

import re


def parse_content_into_chunks(content: str, min_length: int = 10) -> list[str]:
    """Parse text content into individual chunks for memory storage.

    Splits content on:
    - Lines starting with '- ' or '* ' (list items)
    - Lines starting with numbers like '1. ' (numbered lists)
    - Double newlines (paragraphs) if no list items found

    Args:
        content: Text content to parse.
        min_length: Minimum chunk length to include.

    Returns:
        List of cleaned text chunks.
    """
    chunks = []
    list_pattern = re.compile(r"^[\s]*[-*][\s]+|^[\s]*\d+\.[\s]+", re.MULTILINE)

    if list_pattern.search(content):
        # Has list items - split on them
        items = re.split(r"\n(?=[\s]*[-*][\s]+)|(?=[\s]*\d+\.[\s]+)", content)
        for item in items:
            clean = re.sub(r"^[\s]*[-*\d.]+[\s]+", "", item.strip())
            if clean and len(clean) > min_length:
                chunks.append(clean)
    else:
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)
        for p in paragraphs:
            clean = p.strip()
            if clean and len(clean) > min_length:
                chunks.append(clean)

    return chunks
