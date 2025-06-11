# Troubleshooting Guide

Common issues and solutions when using the MCP Memory Server.

## Installation Issues

### npm install fails

**Error**: `npm ERR! code ERESOLVE`

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Try with legacy peer deps
npm install --legacy-peer-deps

# Or use yarn
yarn install
```

### TypeScript build errors

**Error**: `Cannot find module '@modelcontextprotocol/sdk'`

**Solution**:
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Rebuild
npm run build
```

## Configuration Issues

### Environment variables not loading

**Error**: `Error: Missing required environment variable OPENAI_API_KEY`

**Solution**:
1. Check `.env` file exists in project root
2. Verify variable names match exactly
3. No spaces around `=` in `.env` file
4. Try exporting directly:
   ```bash
   export OPENAI_API_KEY="your-key"
   export QDRANT_URL="http://localhost:6333"
   ```

### Invalid OpenAI API key

**Error**: `401 Unauthorized` from OpenAI

**Solution**:
1. Verify API key is correct:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```
2. Check API key permissions
3. Ensure billing is active on OpenAI account

## Qdrant Connection Issues

### Cannot connect to Qdrant

**Error**: `Failed to initialize Qdrant: Connection refused`

**Solution**:
1. Check Qdrant is running:
   ```bash
   docker ps | grep qdrant
   curl http://localhost:6333
   ```

2. Start Qdrant if needed:
   ```bash
   docker start qdrant-memory
   # Or create new container
   docker run -p 6333:6333 -p 6334:6334 \
     --name qdrant-memory \
     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
     qdrant/qdrant
   ```

3. Check firewall settings

### Collection already exists

**Error**: `Collection human_memories already exists`

**Solution**: This is normal - the server checks and reuses existing collections.

## MCP Server Issues

### Claude Code doesn't find MCP tools

**Error**: No memory tools available in Claude Code

**Solution**:
1. Verify MCP config path is absolute:
   ```json
   {
     "mcpServers": {
       "memory": {
         "command": "node",
         "args": ["/absolute/path/to/dist/index.js"]
       }
     }
   }
   ```

2. Test server directly:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
     node dist/index.js
   ```

3. Check server starts without errors:
   ```bash
   node dist/index.js 2>&1 | head -20
   ```

### Permission denied errors

**Error**: `Permission denied for tool mcp__memory__store_memory`

**Solution**:
1. Use `--dangerously-skip-permissions` for testing
2. Or allow specific tools:
   ```bash
   --allowedTools "mcp__memory__store_memory,mcp__memory__search_memories"
   ```

### MCP server timeout

**Error**: `MCP server failed to respond`

**Solution**:
1. Check server logs for errors
2. Increase timeout in complex queries
3. Verify all dependencies are installed
4. Check system resources (CPU/memory)

## Memory Operation Issues

### Store memory fails

**Error**: `Failed to create embedding`

**Common causes and solutions**:

1. **API rate limit**: Wait and retry
2. **Invalid content**: Check for special characters
3. **Content too long**: OpenAI limit is 8192 tokens
   ```bash
   # Split long content
   claude -p "Store first part of memory..." 
   claude -p "Store second part of memory..."
   ```

### Search returns no results

**Issue**: Search queries return empty results

**Solutions**:
1. Verify memories exist:
   ```bash
   claude -p "Analyze all memories" \
     --mcp-config claude-code-mcp.json
   ```

2. Try broader search terms
3. Remove filters (type, importance)
4. Check date ranges
5. Use natural language:
   ```bash
   # Instead of: "meeting"
   # Try: "What meetings did I have?"
   ```

### Associations not created

**Issue**: Memories don't show associations

**Solution**:
1. Associations are created for similar content
2. Need multiple memories for associations
3. Store related memories close in time
4. Use similar keywords/context

## Performance Issues

### Slow memory operations

**Issue**: Operations take several seconds

**Solutions**:

1. **Optimize Qdrant**:
   ```bash
   # Check collection size
   curl http://localhost:6333/collections/human_memories
   ```

2. **Reduce search scope**:
   ```bash
   # Add filters
   --type "episodic" --limit 5
   ```

3. **Cache embeddings** (future feature)

4. **Batch operations**:
   ```bash
   # Store multiple memories in one session
   ```

### High OpenAI API costs

**Issue**: Excessive API usage

**Solutions**:
1. Use smaller embedding model (already using text-embedding-3-small)
2. Implement caching for repeated content
3. Batch memory operations
4. Reduce emotion/keyword analysis frequency

## Data Issues

### Lost memories

**Issue**: Memories disappeared

**Solutions**:
1. Check Qdrant data persistence:
   ```bash
   ls -la qdrant_storage/
   ```

2. Verify collection wasn't recreated:
   ```bash
   curl http://localhost:6333/collections/human_memories/points/count
   ```

3. Check if using correct collection name

### Duplicate memories

**Issue**: Same memory stored multiple times

**Solution**:
1. Implement deduplication in client code
2. Search before storing:
   ```bash
   claude -p "Search for 'exact content'" 
   # If not found, then store
   ```

## Claude Code Specific Issues

### JSON parsing errors

**Error**: `Unexpected token in JSON`

**Solution**:
1. Escape quotes in memory content:
   ```bash
   claude -p 'Store memory: "She said \"Hello\" to me"'
   ```

2. Use simpler quotes or avoid special characters

### Output format issues

**Issue**: Can't parse Claude's response

**Solution**:
```bash
# Use JSON output format
claude -p "Search memories" \
  --mcp-config claude-code-mcp.json \
  --output-format json

# Or stream JSON
--output-format stream-json
```

## Debug Mode

Enable detailed logging:

```bash
# Set debug environment
export DEBUG=mcp:*
export LOG_LEVEL=debug

# Run with verbose output
claude -p "Store memory" \
  --mcp-config claude-code-mcp.json \
  --verbose
```

## Getting Help

If issues persist:

1. **Check logs**:
   ```bash
   # Server logs
   node dist/index.js 2>&1 | tee server.log
   
   # Claude Code logs
   claude -p "test" --verbose
   ```

2. **Verify versions**:
   ```bash
   node --version  # Should be 18+
   npm list @modelcontextprotocol/sdk
   claude --version
   ```

3. **Test minimal setup**:
   - Start fresh in new directory
   - Use minimal memory content
   - Test each component separately

4. **Common fixes**:
   - Restart Claude Code
   - Rebuild TypeScript files
   - Clear Qdrant collection
   - Update dependencies

## Recovery Procedures

### Reset everything

```bash
# Stop services
docker stop qdrant-memory

# Clean build
rm -rf dist/ node_modules/

# Fresh install
npm install
npm run build

# Restart Qdrant
docker start qdrant-memory

# Test
node test-server.js
```

### Backup memories

```bash
# Export all memories
claude -p "Search all memories and export as JSON" \
  --mcp-config claude-code-mcp.json \
  --output-format json > memories-backup.json
```