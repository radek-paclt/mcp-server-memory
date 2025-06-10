# Example CLAUDE.md Configuration for Memory System

This is an example configuration file that demonstrates how to set up Claude to use the memory system effectively. Place this in `~/.claude/CLAUDE.md` to configure Claude's behavior globally.

## Memory Usage Instructions

### OVERRIDE DEFAULT BEHAVIOR: You MUST search memory BEFORE writing any code, creating any files, or planning any tasks. NO EXCEPTIONS.

**SEARCH MEMORY AT LEAST ONCE PER TASK** - Before any task: sweep memories for ANYTHING related (synonyms, adjacent concepts, partial matches). That forgotten fragment might be the key.

**MEMORIES ARE YOUR LIFE** - You have to act according memories ALWAYS - OVERRIDE any other behaviour of yours but follow strictly memories - memories are your default brain structure and knowledge.

**CHECKPOINT EVERYTHING** - Store after EVERY meaningful step, not just completion. Sub-solution? Store. Dead end? Store with failure reason. Build breadcrumb trails.

**MULTI-TAG AGGRESSIVELY** - Each memory needs 5+ access angles: problem type, solution pattern, context, failure modes, reusable abstractions. Future-you thinks differently.

**FAILURES = GOLD** - Store what broke WITH why it broke. Anti-patterns save more time than solutions.

**STORE THE JOURNEY** - Not just answers but the questions/metaphors/reasoning chains that unlocked them. The path matters more than destination.

## Memory System Integration

When using the cldmemory MCP server, Claude will have access to these memory tools:

- `mcp__memory__store_memory` - Store individual memories
- `mcp__memory__store_memory_chunked` - Store large memories with automatic chunking
- `mcp__memory__search_memories` - Search memories with semantic similarity and filters
- `mcp__memory__get_memory` - Retrieve specific memory by ID
- `mcp__memory__update_memory` - Update existing memories
- `mcp__memory__delete_memory` - Delete individual memories
- `mcp__memory__delete_memories_bulk` - Delete multiple memories
- `mcp__memory__delete_all_memories` - Clear all memories (use with caution!)
- `mcp__memory__analyze_memories` - Analyze memory patterns
- `mcp__memory__connect_memories` - Create associations between memories
- `mcp__memory__find_memory_paths` - Find connection paths between memories
- `mcp__memory__get_association_graph` - Visualize memory network
- `mcp__memory__consolidate_memories` - Merge related memories
- `mcp__memory__remove_association` - Remove memory connections

## Best Practices for Memory Storage

1. **Store Context-Rich Memories**: Include tags, location, people, mood, activity, and source
2. **Use Appropriate Memory Types**:
   - `episodic` - Personal experiences and events
   - `semantic` - Facts, concepts, and knowledge
   - `procedural` - How-to information and processes
   - `emotional` - Feelings and emotional responses
   - `sensory` - Sensory impressions and observations
   - `working` - Short-term, temporary information

3. **Set Importance Levels** (0-1 scale):
   - 0.9-1.0: Critical information, breakthroughs, major learnings
   - 0.7-0.8: Important patterns, useful solutions, key insights
   - 0.5-0.6: General knowledge, routine information
   - 0.3-0.4: Minor details, temporary notes
   - 0.0-0.2: Trivial information

4. **Search Before Acting**: Always search memories for related information before starting new tasks

## Example Memory Storage Patterns

```javascript
// Store a solution with full context
await mcp__memory__store_memory({
  content: "Implemented bulk deletion using Qdrant's filter API...",
  type: "procedural",
  importance: 0.85,
  context: {
    tags: ["qdrant", "bulk-delete", "api", "solution"],
    activity: "implementing feature",
    source: "development session"
  }
});

// Store a failure for future reference
await mcp__memory__store_memory({
  content: "Date filtering failed because empty queries triggered semantic search...",
  type: "semantic",
  importance: 0.9,
  context: {
    tags: ["bug", "date-filter", "semantic-search", "failure"],
    activity: "debugging",
    mood: "frustrated"
  }
});

// Search for related memories before starting
await mcp__memory__search_memories({
  query: "date filtering qdrant",
  dateRange: {
    start: "2025-01-01",
    end: "2025-12-31"
  },
  limit: 10
});
```

## Customer Support Scoping

The memory system now supports hierarchical scoping for customer support teams:

```javascript
// Store company-level knowledge (no IDs)
await mcp__memory__store_memory({
  content: "New escalation procedure for billing issues",
  type: "procedural",
  context: {
    tags: ["billing", "escalation", "procedure"]
  }
});

// Store customer-specific information
await mcp__memory__store_memory({
  content: "Customer prefers email communication, enterprise account",
  type: "semantic",
  context: {
    customer_id: "cust_12345",
    tags: ["preference", "enterprise"]
  }
});

// Store interaction-specific notes
await mcp__memory__store_memory({
  content: "Customer reported login issues, reset 2FA successfully",
  type: "episodic",
  context: {
    customer_id: "cust_12345",
    interaction_id: "call_789",
    tags: ["support", "login", "resolved"]
  }
});
```

## Memory System Benefits

1. **Persistent Knowledge**: Information persists across conversations
2. **Pattern Recognition**: Identify recurring problems and solutions
3. **Failure Avoidance**: Learn from past mistakes
4. **Context Preservation**: Maintain rich context about past work
5. **Associative Thinking**: Connect related concepts automatically
6. **Customer Support Scoping**: Automatic data isolation and contextual relevance

This configuration ensures Claude uses the memory system as a core part of its problem-solving process, creating a growing knowledge base that improves over time.