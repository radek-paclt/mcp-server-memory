# Installation Guide

This guide will walk you through setting up the MCP Memory Server step by step.

## Prerequisites

- **Node.js** 18.0 or higher
- **npm** or **yarn**
- **Docker** (for local Qdrant) or access to a Qdrant Cloud instance
- **OpenAI API Key** for embeddings
- **Claude Code** CLI installed

## Step 1: Clone and Install

```bash
# Clone the repository (or create a new directory)
git clone <repository-url> cldmemory
cd cldmemory

# Install dependencies
npm install
```

## Step 2: Set Up Qdrant Database

You have two options for Qdrant:

### Option A: Local Qdrant with Docker

```bash
# Pull and run Qdrant
docker run -p 6333:6333 -p 6334:6334 \
  --name qdrant-memory \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# Verify it's running
curl http://localhost:6333/dashboard
```

### Option B: Qdrant Cloud

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster
3. Get your URL and API key
4. Use these in your `.env` configuration

## Step 3: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# For local Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=human_memories

# For Qdrant Cloud
# QDRANT_URL=https://your-cluster-url.qdrant.io:6333
# QDRANT_API_KEY=your-qdrant-api-key

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-openai-api-key
OPENAI_MODEL=text-embedding-3-small

# MCP Server Configuration
MCP_SERVER_NAME=memory-server
MCP_SERVER_PORT=3000
```

### Getting an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in or create an account
3. Navigate to API keys section
4. Create a new API key
5. Copy and paste into your `.env` file

## Step 4: Build the Project

```bash
# Build TypeScript files
npm run build

# Verify build succeeded
ls dist/
# Should see: index.js and other compiled files
```

## Step 5: Create MCP Configuration

Create `claude-code-mcp.json` in your project root:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["<absolute-path-to>/cldmemory/dist/index.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

Replace `<absolute-path-to>` with your actual path (e.g., `/home/username/projects`).

## Step 6: Test the Installation

```bash
# Test the server directly
node dist/index.js

# In another terminal, send a test request
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node dist/index.js
```

You should see a JSON response listing available tools.

## Step 7: Configure Claude Desktop (Optional)

If you also want to use this with Claude Desktop app:

1. Find your Claude config file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the memory server configuration to the file
3. Restart Claude Desktop

## Verification

Run this command to verify everything is working:

```bash
claude -p "List available memory tools" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions
```

You should see a list of 6 memory-related tools.

## Next Steps

- Read the [Configuration Guide](./configuration.md) for advanced settings
- Check out the [Usage Guide](./usage.md) to start using memories
- See [Examples](./examples.md) for real-world usage

## Troubleshooting Installation

If you encounter issues:

1. **Permission Denied**: Make sure all scripts are executable
   ```bash
   chmod +x start-mcp.sh test-memory.sh
   ```

2. **Module Not Found**: Rebuild the project
   ```bash
   npm run build
   ```

3. **API Key Issues**: Verify your OpenAI API key is valid
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

4. **Qdrant Connection**: Check Qdrant is running
   ```bash
   curl http://localhost:6333
   ```