"""Structural validation for the Claude Code plugin."""

import json
import re
from pathlib import Path

PLUGIN_ROOT = Path(__file__).parent.parent
MANIFEST_DIR = PLUGIN_ROOT / ".claude-plugin"


def test_plugin_json_is_valid():
    manifest = json.loads((MANIFEST_DIR / "plugin.json").read_text())
    assert manifest["name"]


def test_plugin_components_live_at_root():
    """Claude Code only loads components from the plugin root, never .claude-plugin/."""
    assert sorted(p.name for p in MANIFEST_DIR.iterdir()) == ["marketplace.json", "plugin.json"]
    assert (PLUGIN_ROOT / "commands" / "recall.md").exists()
    assert (PLUGIN_ROOT / "skills" / "recall-nudge" / "SKILL.md").exists()


def test_plugin_hooks_inject_hot_cache():
    """Hook stdout is the only path that puts memories in context with no tool call."""
    hooks = json.loads((MANIFEST_DIR / "plugin.json").read_text())["hooks"]

    def commands(event: str) -> list[str]:
        return [h["command"] for entry in hooks[event] for h in entry["hooks"]]

    assert commands("SessionStart") == ["memory-mcp-cli hot-cache --force"]
    assert commands("UserPromptSubmit") == ["memory-mcp-cli hot-cache"]
    assert commands("Stop") == ["memory-mcp-cli log-response"]
    assert commands("PreCompact") == ["memory-mcp-cli pre-compact"]


def test_recall_nudge_skill_structure():
    skill = PLUGIN_ROOT / "skills" / "recall-nudge" / "SKILL.md"
    assert skill.exists()
    text = skill.read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "missing YAML frontmatter"
    frontmatter = match.group(1)
    assert "name: recall-nudge" in frontmatter
    assert "description:" in frontmatter
    for phrase in ("didn't we", "last time", "how did we"):
        assert phrase in text.lower()


def test_command_relative_links_resolve():
    """The testing resources moved with the commands that link to them."""
    for command in (PLUGIN_ROOT / "commands").glob("*.md"):
        for target in re.findall(r"\]\((?!https?:)([^)#]+\.md)\)", command.read_text()):
            assert (command.parent / target).exists(), f"{command.name} -> {target}"
