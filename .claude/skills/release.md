# Release Skill

Automates version bump, PyPI publish, and MCP registry publish.

## Usage

```
/release [patch|minor|major]
```

Default is `patch` if not specified.

## Steps

1. **Pre-flight checks**
   - Ensure working directory is clean (no uncommitted changes)
   - Ensure on main branch
   - Run tests to verify everything passes

2. **Version bump**
   - Read current version from `pyproject.toml`
   - Bump version according to semver (patch/minor/major)
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

7. **Summary**
   - Report success with links to:
     - GitHub release
     - PyPI package
     - MCP Registry entry

## Example

```
User: /release patch
Assistant: Releasing v0.4.2...
- Tests passed
- Version bumped: 0.4.1 -> 0.4.2
- Committed and tagged
- GitHub release created
- PyPI publish complete
- MCP Registry updated

Links:
- GitHub: https://github.com/michael-denyer/memory-mcp/releases/tag/v0.4.2
- PyPI: https://pypi.org/project/hot-memory-mcp/0.4.2/
- Registry: io.github.michael-denyer/hot-memory-mcp v0.4.2
```
