# Development Setup Guide

## Quick Start

1. **Copy the development environment file**:
   ```bash
   cp .env.development .env.development.local
   ```

2. **Edit your local config** (add your OpenAI API key):
   ```bash
   # Edit .env.development.local
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   ```

3. **Start Qdrant locally** (if not running):
   ```bash
   # Using Docker
   docker run -p 6333:6333 -p 6334:6334 \
     --name qdrant-memory-dev \
     -v $(pwd)/qdrant_storage_dev:/qdrant/storage:z \
     qdrant/qdrant
   
   # Or using Docker Compose
   docker-compose up -d qdrant
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

## Environment Configuration

The `.env.development` file is already configured for local development:

```env
# Qdrant runs on localhost
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=human_memories_dev  # Separate from production

# Development-specific settings
NODE_ENV=development
DEBUG=true
LOG_LEVEL=debug
SIMILARITY_THRESHOLD=0.3
```

## Testing the Server

Once running, you can test the memory server:

```bash
# Test with Claude Code (if you have it)
claude -p "Store a memory: Testing development setup" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions

# Or test the MCP server directly
curl -X POST http://localhost:3000/health
```

## Customer Support Scoping Examples

Test the new hierarchical scoping features:

```bash
# Company level memory (no IDs)
claude -p "Store: Global escalation procedure for billing issues" \
  --mcp-config claude-code-mcp.json

# Customer level memory
claude -p "Store memory with customer context: {\"customer_id\": \"dev_123\", \"content\": \"Customer prefers email\"}" \
  --mcp-config claude-code-mcp.json

# Interaction level memory  
claude -p "Store memory with full context: {\"customer_id\": \"dev_123\", \"interaction_id\": \"call_456\", \"content\": \"Resolved login issue\"}" \
  --mcp-config claude-code-mcp.json
```

## Development Notes

- **Development collection**: Uses `human_memories_dev` to avoid conflicts with production
- **Debug logging**: Enabled in development mode
- **Auto-reload**: Uses `tsx` for TypeScript hot reloading
- **Separate storage**: Qdrant data stored in `qdrant_storage_dev/`

## Troubleshooting

1. **Qdrant connection issues**:
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   
   # Restart Qdrant if needed
   docker restart qdrant-memory-dev
   ```

2. **OpenAI API issues**:
   - Make sure your API key is valid and has credits
   - Check the key is correctly set in `.env.development.local`

3. **TypeScript compilation errors**:
   ```bash
   # Check for type errors
   npm run typecheck
   
   # Rebuild if needed
   npm run build
   ```

4. **MCP connection issues**:
   - Ensure you're using the correct MCP config file
   - Check that the server is running on the expected port (3000)

## Next Steps

After development setup is working:
1. Try storing different types of memories (episodic, semantic, procedural)
2. Test the customer support scoping with different customer_id values
3. Experiment with search queries and similarity thresholds
4. Check out the docs in `docs/` folder for advanced usage