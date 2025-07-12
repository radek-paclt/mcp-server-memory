# MCP Memory Server

A human-like memory system using Qdrant vector database and OpenAI embeddings, accessible through the Model Context Protocol (MCP).

## 🔄 Automatic Updates

### Quick Update
```bash
./update.sh
# or
npm run update
```

### Automatic Update Monitoring
```bash
./watch-updates.sh
# or
npm run watch-updates
```

The script checks for new versions every 5 minutes and notifies you.

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

- **🎯 Customer Support Scoping**:
  - **Company Level**: Global knowledge accessible to all agents
  - **Customer Level**: Customer-specific information across all interactions
  - **Interaction Level**: Conversation-specific notes and context

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

## Customer Support Scoping

The memory system supports **three-level hierarchical scoping** for customer support teams:

### **Company Level** (no customer_id, no interaction_id)
```bash
# Store global knowledge accessible to all agents
claude -p 'Store memory: "New refund policy: Premium customers get instant refunds, standard customers within 3 business days"' --mcp-config claude-code-mcp.json
```

### **Customer Level** (customer_id only)
```bash
# Store customer-specific information
claude -p 'Store memory: "Customer Sarah Johnson prefers phone support, has enterprise account, key contact for TechCorp implementation" with context: {"customer_id": "cust_12345", "tags": ["vip", "enterprise"]}' --mcp-config claude-code-mcp.json
```

### **Interaction Level** (both customer_id and interaction_id)
```bash
# Store conversation-specific notes
claude -p 'Store memory: "Customer reports login issues with 2FA, helped reset authenticator app, issue resolved" with context: {"customer_id": "cust_12345", "interaction_id": "call_789", "tags": ["support", "resolved"]}' --mcp-config claude-code-mcp.json
```

### **Smart Search with Scope Inheritance**

```bash
# Search inherits from all relevant scopes automatically
claude -p 'Search memories for "login issues" with customer_id: "cust_12345", interaction_id: "call_789"' --mcp-config claude-code-mcp.json

# Results include:
# 1. Current interaction memories (highest priority)
# 2. Customer-specific memories (medium priority)  
# 3. Company knowledge base (lowest priority, but still relevant)
```

### **Customer Support Use Cases**

```bash
# Agent preparation before call
claude -p 'Search customer history for customer_id: "cust_12345"' --mcp-config claude-code-mcp.json

# During live support - get full context
claude -p 'Search memories for "billing" with customer_id: "cust_12345", interaction_id: "call_current"' --mcp-config claude-code-mcp.json

# Store resolution for team learning
claude -p 'Store memory: "Billing API timeout fixed by increasing database connection pool from 10 to 50 connections"' --mcp-config claude-code-mcp.json
```

### **Benefits**

- **🔒 Automatic Isolation**: Customer data stays separate
- **📈 Contextual Relevance**: Right information at the right time
- **🧠 Team Learning**: Company knowledge grows with each resolution
- **⚡ Fast Context**: Instant access to customer history and current interaction
- **🔄 Inheritance**: Broader knowledge automatically included when relevant

### **Migration from Previous Version**

All existing memories automatically become "company level" - no breaking changes! Just start adding `customer_id` and `interaction_id` to new memories for scoping.

## Example Usage

```javascript
// Store a company-level memory (available to all agents)
store_memory({
  "content": "New API rate limits: 1000 requests per minute for free tier, 10000 for premium",
  "type": "semantic",
  "context": {
    "tags": ["api", "rate-limits", "policy"]
  },
  "importance": 0.9
})

// Store customer-specific memory
store_memory({
  "content": "Customer prefers email communication, works in EST timezone",
  "type": "semantic", 
  "context": {
    "customer_id": "cust_12345",
    "tags": ["preferences", "timezone"]
  },
  "importance": 0.7
})

// Store interaction-specific memory
store_memory({
  "content": "Resolved billing discrepancy of $45.32, issued refund",
  "type": "episodic",
  "context": {
    "customer_id": "cust_12345",
    "interaction_id": "call_789",
    "tags": ["billing", "refund", "resolved"]
  },
  "importance": 0.8
})

// Search with scoping
search_memories({
  "query": "billing issues",
  "customer_id": "cust_12345",
  "interaction_id": "call_789",
  "limit": 10
})
```

## Development

- `npm run dev` - Run in development mode
- `npm run build` - Build TypeScript
- `npm run test` - Run tests
- `npm run lint` - Run linter
- `npm run typecheck` - Type check