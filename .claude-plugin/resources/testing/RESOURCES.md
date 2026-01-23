Run Phase 12 of Memory MCP testing: MCP Resources.

## Tests to Execute

**12.1 Hot Cache Resource**:
Read `memory://hot-cache` using `ReadMcpResourceTool`:
- Contains all promoted memories for instant recall
- Verify contents match `mcp__memory__hot_cache_status()` items

**12.2 Working Set Resource**:
Read `memory://working-set`:
- Session-aware active context (~10 items max)
- Combines: recently recalled, predicted next, top salience
- Verify smaller/more focused than hot-cache

**12.3 Project Context Resource**:
Read `memory://project-context`:
- Project-scoped memories for current working directory
- Should filter to current project only

**12.4 List All Resources**:
- `ListMcpResourcesTool(server="memory")` → see all available resources

## Tracking

Report results:
| Test | Status | Notes |
|------|--------|-------|
| 12.1 Hot Cache Resource | ⬜ | |
| 12.2 Working Set Resource | ⬜ | |
| 12.3 Project Context Resource | ⬜ | |
| 12.4 List Resources | ⬜ | |

Proceed to Phase 14: Edge Cases.
