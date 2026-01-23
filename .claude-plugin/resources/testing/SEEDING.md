Run Phase 8 of Memory MCP testing: Seeding & Bootstrap.

## Tests to Execute

**8.1 Seed from Text**:
```
mcp__memory__seed_from_text("- Item one\n- Item two\n- Item three", memory_type="project")
```
Verify 3 memories created.

**8.2 Seed from File**:
- `mcp__memory__seed_from_file(file_path, memory_type="reference")`
- Use an existing file like README.md

**8.3 Bootstrap Project**:
- `mcp__memory__bootstrap_project(root_path=".", promote_to_hot=false)`
- Verify it finds CLAUDE.md, README.md, etc.

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 8.1 Seed from Text | ⬜ | |
| 8.2 Seed from File | ⬜ | |
| 8.3 Bootstrap | ⬜ | |

Proceed to Phase 9-10: Predictive & Quality.
