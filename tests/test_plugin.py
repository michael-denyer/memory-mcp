"""Structural validation for the Claude Code plugin."""

import json
import re
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent.parent / ".claude-plugin"


def test_plugin_json_is_valid():
    manifest = json.loads((PLUGIN_DIR / "plugin.json").read_text())
    assert manifest["name"]


def test_recall_nudge_skill_structure():
    skill = PLUGIN_DIR / "skills" / "recall-nudge" / "SKILL.md"
    assert skill.exists()
    text = skill.read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "missing YAML frontmatter"
    frontmatter = match.group(1)
    assert "name: recall-nudge" in frontmatter
    assert "description:" in frontmatter
    for phrase in ("didn't we", "last time", "how did we"):
        assert phrase in text.lower()
