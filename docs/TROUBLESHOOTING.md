# Troubleshooting

Common issues and solutions for Memory MCP.

## Server Won't Start

**Symptom**: Claude Code shows "memory" server as disconnected

1. **Check the command works directly**:
   ```bash
   memory-mcp
   ```

2. **Verify installation**:
   ```bash
   which memory-mcp  # Should return a path
   ```

3. **Check Python version**: Requires 3.10+
   ```bash
   python --version
   ```

## Dimension Mismatch Error

**Symptom**: `Vector dimension mismatch` error during recall

This happens when the embedding model changes. Rebuild vectors:

```bash
memory-mcp-cli db-rebuild-vectors
```

## Hot Cache Not Updating

**Symptom**: Promoted memories don't appear in hot cache

1. **Check hot cache status**:
   ```bash
   memory-mcp-cli status
   ```

2. **Verify memory exists**:
   ```
   [In Claude] list_memories(limit=10)
   ```

3. **Manually promote**:
   ```
   [In Claude] promote(memory_id)
   ```

## Pattern Mining Not Working

**Symptom**: `run_mining` finds no patterns

1. **Check mining is enabled**:
   ```bash
   echo $MEMORY_MCP_MINING_ENABLED  # Should not be "false"
   ```

2. **Verify logs exist**:
   ```bash
   memory-mcp-cli run-mining --hours 24
   ```

3. **Check hook is installed** (see [Reference - Automatic Output Logging](REFERENCE.md#automatic-output-logging))

## Hook Script Fails

**Symptom**: Hook runs but nothing is logged

1. **Check jq is installed**:
   ```bash
   which jq  # Should return a path
   ```

2. **Make script executable**:
   ```bash
   chmod +x hooks/memory-log-response.sh
   ```

3. **Test manually**:
   ```bash
   echo "test content" | memory-mcp-cli log-output
   ```

## Slow First Startup

**Symptom**: First run takes 30-60 seconds

This is expected - the embedding model (~90MB) downloads on first use. Subsequent starts take 2-5 seconds.

## Database Corruption

**Symptom**: SQLite errors or unexpected behavior

1. **Backup and recreate**:
   ```bash
   mv ~/.memory-mcp/memory.db ~/.memory-mcp/memory.db.bak
   # Server will create fresh database on next start
   ```

2. **Re-bootstrap from project docs**:
   ```bash
   memory-mcp-cli bootstrap
   ```

## Memory Analyst Shows Issues

**Symptom**: Health report shows "Needs Attention"

Run the recommended actions from the report. Common fixes:

- **Unresolved contradictions**: Use `resolve_contradiction()` to pick the correct memory
- **Consolidation opportunities**: Run `memory-mcp-cli consolidate`
- **Empty knowledge graph**: Use `link_memories()` to connect related concepts
- **Low hot cache utilization**: Run `memory-mcp-cli bootstrap` to seed from docs
