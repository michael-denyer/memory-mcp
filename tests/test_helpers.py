"""Tests for pure helper functions used by the server and the CLI hooks."""

from datetime import datetime, timezone

from memory_mcp.helpers import format_hot_cache_for_injection
from memory_mcp.models import Memory, MemorySource, MemoryType


def _memory(memory_id: int, content: str, tags: list[str]) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        content_hash=f"hash-{memory_id}",
        memory_type=MemoryType.PROJECT,
        source=MemorySource.MANUAL,
        is_hot=True,
        is_pinned=False,
        promotion_source=None,
        tags=tags,
        access_count=0,
        last_accessed_at=None,
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
    )


class TestFormatHotCacheForInjection:
    """The hook's stdout is the injected context, so its exact shape matters."""

    def test_format_hot_cache_for_injection_shape(self):
        memories = [
            _memory(7, "The deploy password hint is zebra-42.", ["deploy", "secret", "ops", "x"]),
            _memory(9, "Run make lint before every push.", []),
        ]

        assert format_hot_cache_for_injection(memories, max_chars=200) == (
            "[MEMORY: Hot cache]\n"
            "- [id:7] The deploy password hint is zebra-42. [deploy, secret, ops]\n"
            "- [id:9] Run make lint before every push.\n"
            "Call mark_memory_used(id) when one of these was useful."
        )

        assert format_hot_cache_for_injection([], max_chars=200) == ""

    def test_truncates_long_content(self):
        memories = [_memory(1, "abcdefghij", [])]

        assert format_hot_cache_for_injection(memories, max_chars=4) == (
            "[MEMORY: Hot cache]\n"
            "- [id:1] abcd...\n"
            "Call mark_memory_used(id) when one of these was useful."
        )
