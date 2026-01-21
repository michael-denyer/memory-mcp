"""Bootstrap methods mixin for Storage class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from memory_mcp.logging import get_logger
from memory_mcp.models import MemorySource, MemoryType
from memory_mcp.storage.memory_crud import ValidationError

if TYPE_CHECKING:
    pass

log = get_logger("storage.bootstrap")


class BootstrapMixin:
    """Mixin providing bootstrap methods for Storage."""

    def bootstrap_from_files(
        self,
        file_paths: list[Path],
        memory_type: MemoryType = MemoryType.PROJECT,
        promote_to_hot: bool = True,
        tags: list[str] | None = None,
    ) -> dict:
        """Seed memories from multiple files with deduplication.

        Reads each file, parses into chunks, and stores as memories.
        Handles edge cases gracefully (empty files, permission errors, etc.).

        Args:
            file_paths: List of file paths to process.
            memory_type: Type to assign to all created memories.
            promote_to_hot: Whether to promote new memories to hot cache.
            tags: Optional tags to apply to all memories.

        Returns:
            Dict with:
                - success: Always True (errors are reported, not raised)
                - files_found: Number of files in input
                - files_processed: Files successfully read
                - memories_created: New memories stored
                - memories_skipped: Duplicates (already existed)
                - hot_cache_promoted: Memories added to hot cache
                - errors: List of error messages
                - message: Human-readable summary
        """
        from memory_mcp.text_parsing import parse_content_into_chunks, parse_markdown_with_context

        # Track results with explicit types to satisfy mypy
        errors: list[str] = []
        files_processed = 0
        memories_created = 0
        memories_skipped = 0
        hot_cache_promoted = 0

        tag_list = tags or []

        # Get project_id if project awareness is enabled
        project_id = None
        if self.settings.project_awareness_enabled:
            from memory_mcp.project import get_current_project_id

            project_id = get_current_project_id()

        files_found = len(file_paths)

        for path in file_paths:
            # Check file exists
            if not path.exists():
                errors.append(f"{path.name}: file not found")
                continue

            # Check it's a file
            if not path.is_file():
                errors.append(f"{path.name}: not a file")
                continue

            # Detect binary files (skip them)
            if self._is_binary_file(path):
                errors.append(f"{path.name}: binary file skipped")
                continue

            # Read file content
            try:
                content = path.read_text(encoding="utf-8")
            except PermissionError:
                errors.append(f"{path.name}: permission denied")
                continue
            except UnicodeDecodeError:
                errors.append(f"{path.name}: encoding error (not UTF-8)")
                continue
            except OSError as e:
                errors.append(f"{path.name}: read error ({e})")
                continue

            # Skip empty files
            if not content.strip():
                log.debug("Skipping empty file: {}", path.name)
                continue

            files_processed += 1

            # Parse into chunks (use markdown-aware parser for .md files)
            source_name = path.name
            if path.suffix.lower() in (".md", ".markdown"):
                chunks = parse_markdown_with_context(content, source_name=source_name)
            else:
                chunks = parse_content_into_chunks(content, source_name=source_name)

            for chunk in chunks:
                # Skip chunks that are too long
                if len(chunk) > self.settings.max_content_length:
                    errors.append(f"{path.name}: chunk too long ({len(chunk)} chars), skipped")
                    continue

                # Store the memory (catch validation errors to honor "errors reported not raised")
                try:
                    memory_id, is_new = self.store_memory(
                        content=chunk,
                        memory_type=memory_type,
                        source=MemorySource.MANUAL,
                        tags=tag_list,
                        project_id=project_id,
                    )
                except ValidationError as e:
                    errors.append(f"{path.name}: {e}")
                    continue

                if is_new:
                    memories_created += 1
                    if promote_to_hot:
                        if self.promote_to_hot(memory_id):
                            hot_cache_promoted += 1
                else:
                    memories_skipped += 1

        # Build summary message
        if files_found == 0:
            message = "No files provided. Pass file paths or use auto-detection."
        elif files_processed == 0 and files_found > 0:
            message = (
                f"No files could be processed from {files_found} provided. "
                "Check file permissions and paths."
            )
        elif memories_created == 0 and files_processed > 0:
            if memories_skipped > 0:
                message = (
                    f"All {memories_skipped} memories already exist from {files_processed} file(s)."
                )
            else:
                message = (
                    f"No memories extracted from {files_processed} file(s). "
                    "Files may be empty or contain only non-extractable content."
                )
        else:
            message = f"Bootstrapped {memories_created} memories from {files_processed} file(s)"
            if hot_cache_promoted > 0:
                message += f" ({hot_cache_promoted} promoted to hot cache)"

        log.info(
            "bootstrap_from_files: files={}/{} created={} skipped={} errors={}",
            files_processed,
            files_found,
            memories_created,
            memories_skipped,
            len(errors),
        )

        return {
            "success": True,
            "files_found": files_found,
            "files_processed": files_processed,
            "memories_created": memories_created,
            "memories_skipped": memories_skipped,
            "hot_cache_promoted": hot_cache_promoted,
            "errors": errors,
            "message": message,
        }

    def _is_binary_file(self, path: Path) -> bool:
        """Check if a file is binary by reading first bytes.

        Args:
            path: File path to check.

        Returns:
            True if file appears to be binary.
        """
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                # Check for null bytes (common in binary files)
                if b"\x00" in chunk:
                    return True
                # Check for very high ratio of non-text bytes
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
                non_text = sum(1 for byte in chunk if byte not in text_chars)
                return len(chunk) > 0 and non_text / len(chunk) > 0.30
        except OSError:
            return False  # Can't read, let the main read handle the error
