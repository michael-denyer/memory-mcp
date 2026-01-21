"""Project detection and management for Memory MCP.

This module handles automatic detection of the current project (git repo)
and provides utilities for project-aware memory operations.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from memory_mcp.logging import get_logger

log = get_logger("project")


@dataclass
class ProjectInfo:
    """Information about a detected project."""

    id: str  # Normalized identifier (e.g., "owner/repo" or path hash)
    name: str  # Human-readable name
    path: str  # Absolute path to project root
    remote_url: str | None = None  # Git remote URL if available


def _run_git_command(args: list[str], cwd: str | None = None) -> str | None:
    """Run a git command and return output, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _normalize_git_url(url: str) -> str | None:
    """Extract owner/repo from various git URL formats.

    Handles:
    - https://github.com/owner/repo.git
    - git@github.com:owner/repo.git
    - https://gitlab.com/owner/repo
    - ssh://git@bitbucket.org/owner/repo.git

    Returns:
        Normalized "host/owner/repo" string, or None if unparseable.
    """
    if not url:
        return None

    # Remove .git suffix
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    # SSH format: git@github.com:owner/repo
    ssh_match = re.match(r"git@([^:]+):(.+)", url)
    if ssh_match:
        host, path = ssh_match.groups()
        # Normalize common hosts
        host = _normalize_host(host)
        return f"{host}/{path}"

    # HTTPS format: https://github.com/owner/repo
    https_match = re.match(r"https?://([^/]+)/(.+)", url)
    if https_match:
        host, path = https_match.groups()
        host = _normalize_host(host)
        return f"{host}/{path}"

    # SSH with ssh:// prefix: ssh://git@github.com/owner/repo
    ssh_prefix_match = re.match(r"ssh://[^@]+@([^/]+)/(.+)", url)
    if ssh_prefix_match:
        host, path = ssh_prefix_match.groups()
        host = _normalize_host(host)
        return f"{host}/{path}"

    return None


def _normalize_host(host: str) -> str:
    """Normalize git host names for consistency."""
    host = host.lower()
    # Map common variations
    host_map = {
        "github.com": "github",
        "gitlab.com": "gitlab",
        "bitbucket.org": "bitbucket",
    }
    return host_map.get(host, host)


def get_git_root(path: str | None = None) -> str | None:
    """Get the root directory of the git repository.

    Args:
        path: Starting path (defaults to cwd)

    Returns:
        Absolute path to git root, or None if not in a git repo.
    """
    cwd = path or os.getcwd()
    result = _run_git_command(["rev-parse", "--show-toplevel"], cwd=cwd)
    return result


def get_git_remote_url(path: str | None = None) -> str | None:
    """Get the remote URL for the git repository.

    Args:
        path: Path within the git repo (defaults to cwd)

    Returns:
        Remote URL string, or None if not available.
    """
    cwd = path or os.getcwd()
    # Try origin first, then any remote
    result = _run_git_command(["remote", "get-url", "origin"], cwd=cwd)
    if result:
        return result

    # Fall back to first available remote
    remotes = _run_git_command(["remote"], cwd=cwd)
    if remotes:
        first_remote = remotes.split("\n")[0].strip()
        if first_remote:
            return _run_git_command(["remote", "get-url", first_remote], cwd=cwd)

    return None


@lru_cache(maxsize=16)
def detect_project(path: str | None = None) -> ProjectInfo | None:
    """Detect the current project from git repository.

    Args:
        path: Path to check (defaults to cwd)

    Returns:
        ProjectInfo if a git repo is detected, None otherwise.
    """
    cwd = path or os.getcwd()

    # Get git root
    git_root = get_git_root(cwd)
    if not git_root:
        log.debug("No git repository found at {}", cwd)
        return None

    # Get remote URL
    remote_url = get_git_remote_url(cwd)
    normalized = _normalize_git_url(remote_url) if remote_url else None

    # Determine project ID and name
    if normalized:
        # Use normalized remote URL as ID (e.g., "github/owner/repo")
        project_id = normalized
        # Extract repo name as human-readable name
        name = normalized.split("/")[-1]
    else:
        # Fall back to directory name for local-only repos
        project_id = f"local/{Path(git_root).name}"
        name = Path(git_root).name

    project = ProjectInfo(
        id=project_id,
        name=name,
        path=git_root,
        remote_url=remote_url,
    )

    log.debug("Detected project: {} at {}", project.id, project.path)
    return project


def clear_project_cache() -> None:
    """Clear the project detection cache.

    Useful when the working directory changes.
    """
    detect_project.cache_clear()


def get_current_project_id(path: str | None = None) -> str | None:
    """Get the current project ID, or None if not in a project.

    This is a convenience function for use in memory operations.
    """
    project = detect_project(path)
    return project.id if project else None
