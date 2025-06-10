<<<<<<< HEAD
# MCP Memory Server

A human-like memory system using Qdrant vector database and OpenAI embeddings, accessible through the Model Context Protocol (MCP).

## 🔄 Automatické aktualizace

### Rychlá aktualizace
```bash
./update.sh
# nebo
npm run update
```

### Automatické sledování aktualizací
```bash
./watch-updates.sh
# nebo
npm run watch-updates
```

Skript kontroluje nové verze každých 5 minut a upozorní vás.

## Features

- **Human-like Memory Types**:
  - Episodic (personal experiences)
  - Semantic (facts and knowledge)
  - Procedural (how to do things)
  - Emotional (emotional memories)
  - Sensory (sensory impressions)
  - Working (short-term memory)

- **Memory Characteristics**:
  - Importance scoring (0-1)
  - Emotional valence (-1 to 1)
  - Associations between memories
  - Context (location, people, mood, activity)
  - Decay factor and access tracking

## Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start Qdrant** (if using local):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
     --name qdrant-memory \
     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
     qdrant/qdrant
   ```

4. **Build the project**:
   ```bash
   npm run build
   ```

## Testing with Claude Code

Use the MCP configuration file with Claude Code CLI:

```bash
# Basic usage
claude -p "Store a memory about today's meeting" --mcp-config claude-code-mcp.json

# Skip permissions for automation
claude -p "Search my memories" --mcp-config claude-code-mcp.json --dangerously-skip-permissions

# List available tools
claude -p "List available memory tools" --mcp-config claude-code-mcp.json
```

Available MCP tools (prefixed with `mcp__memory__`):

- `store_memory` - Store a new memory
- `search_memories` - Search for memories using natural language
- `get_memory` - Retrieve a specific memory
- `update_memory` - Update an existing memory
- `delete_memory` - Delete a memory
- `analyze_memories` - Analyze memory patterns

## Example Usage

```
# Store a memory
store_memory({
  "content": "Had lunch with Sarah at the Italian restaurant",
  "type": "episodic",
  "context": {
    "location": "Downtown Italian restaurant",
    "people": ["Sarah"],
    "mood": "happy"
  },
  "importance": 0.8
})

# Search memories
search_memories({
  "query": "restaurant experiences",
  "limit": 5,
  "includeAssociations": true
})
```

## Development

- `npm run dev` - Run in development mode
- `npm run build` - Build TypeScript
- `npm run test` - Run tests
- `npm run lint` - Run linter
=======
# MCP Memory Server

A human-like memory system using Qdrant vector database and OpenAI embeddings, accessible through the Model Context Protocol (MCP).

## 🔄 Automatické aktualizace

### Rychlá aktualizace
```bash
./update.sh
# nebo
npm run update
```

### Automatické sledování aktualizací
```bash
./watch-updates.sh
# nebo
npm run watch-updates
```

Skript kontroluje nové verze každých 5 minut a upozorní vás.

## Features

- **Human-like Memory Types**:
  - Episodic (personal experiences)
  - Semantic (facts and knowledge)
  - Procedural (how to do things)
  - Emotional (emotional memories)
  - Sensory (sensory impressions)
  - Working (short-term memory)

- **Memory Characteristics**:
  - Importance scoring (0-1)
  - Emotional valence (-1 to 1)
  - Associations between memories
  - Context (location, people, mood, activity)
  - Decay factor and access tracking

## Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start Qdrant** (if using local):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
     --name qdrant-memory \
     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
     qdrant/qdrant
   ```

4. **Build the project**:
   ```bash
   npm run build
   ```

## Testing with Claude Code

Use the MCP configuration file with Claude Code CLI:

```bash
# Basic usage
claude -p "Store a memory about today's meeting" --mcp-config claude-code-mcp.json

# Skip permissions for automation
claude -p "Search my memories" --mcp-config claude-code-mcp.json --dangerously-skip-permissions

# List available tools
claude -p "List available memory tools" --mcp-config claude-code-mcp.json
```

Available MCP tools (prefixed with `mcp__memory__`):

- `store_memory` - Store a new memory
- `search_memories` - Search for memories using natural language
- `get_memory` - Retrieve a specific memory
- `update_memory` - Update an existing memory
- `delete_memory` - Delete a memory
- `analyze_memories` - Analyze memory patterns

## Example Usage

```
# Store a memory
store_memory({
  "content": "Had lunch with Sarah at the Italian restaurant",
  "type": "episodic",
  "context": {
    "location": "Downtown Italian restaurant",
    "people": ["Sarah"],
    "mood": "happy"
  },
  "importance": 0.8
})

# Search memories
search_memories({
  "query": "restaurant experiences",
  "limit": 5,
  "includeAssociations": true
})
```

## Development

- `npm run dev` - Run in development mode
- `npm run build` - Build TypeScript
- `npm run test` - Run tests
- `npm run lint` - Run linter
>>>>>>> origin/main
- `npm run typecheck` - Type check