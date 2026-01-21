---
name: publish
description: Publish a new version to PyPI, MCP Registry, and Homebrew. Use after /release to publish.
disable-model-invocation: true
argument-hint: "[patch|minor|major]"
---

# Publish Skill

Publishes to PyPI, MCP Registry, and Homebrew. Run `/release` first to prepare the code.

## Usage

```
/publish [patch|minor|major]
```

Default is `patch` if not specified. The version type is passed as `$ARGUMENTS`.

## Steps

1. **Pre-flight checks**
   - Ensure working directory is clean (no uncommitted changes)
   - Ensure on main branch
   - Run tests to verify everything passes

2. **Version bump**
   - Read current version from `pyproject.toml`
   - Bump version according to semver ($ARGUMENTS or default to patch)
   - Update version in:
     - `pyproject.toml`
     - `server.json` (both root version and package version)
   - Update `uv.lock`

3. **Commit and tag**
   - Stage version files: `pyproject.toml`, `server.json`, `uv.lock`
   - Commit with message: `chore: release vX.Y.Z`
   - Create git tag: `vX.Y.Z`
   - Push commit and tag to origin

4. **Create GitHub release**
   - Use `gh release create vX.Y.Z`
   - Title: `vX.Y.Z`
   - Auto-generate release notes from commits since last tag

5. **Wait for PyPI publish**
   - Monitor GitHub Actions `publish.yml` workflow
   - Wait for completion (typically 30-60 seconds)
   - Verify package appears on PyPI

6. **Publish to MCP Registry**
   - Run `/tmp/mcp-registry/bin/mcp-publisher publish`
   - If mcp-publisher not found, build it first:
     ```bash
     cd /tmp && git clone --depth 1 https://github.com/modelcontextprotocol/registry.git mcp-registry
     cd mcp-registry && make publisher
     ```
   - If not logged in, run `mcp-publisher login github` first

7. **Update Homebrew tap**
   - Get new SHA256 from PyPI:
     ```bash
     curl -sL https://pypi.org/pypi/hot-memory-mcp/json | jq -r '.urls[] | select(.packagetype=="sdist") | .digests.sha256'
     ```
   - Clone tap repo:
     ```bash
     cd /tmp && git clone https://github.com/michael-denyer/homebrew-tap.git
     ```
   - Update `Formula/hot-memory-mcp.rb`:
     - Update `url` with new version
     - Update `sha256` with new hash
   - Commit and push:
     ```bash
     cd /tmp/homebrew-tap && git add -A && git commit -m "chore: bump hot-memory-mcp to vX.Y.Z" && git push
     ```

8. **Summary**
   - Report success with links to:
     - GitHub release
     - PyPI package: https://pypi.org/project/hot-memory-mcp/
     - MCP Registry: io.github.michael-denyer/hot-memory-mcp
     - Homebrew: `brew tap michael-denyer/tap && brew install hot-memory-mcp`
