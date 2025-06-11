# API Reference

Complete reference for all MCP tools provided by the Memory Server.

## Tool Names

All tools are prefixed with `mcp__memory__` when used in Claude Code.

## store_memory

Store a new memory with human-like characteristics.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The memory content |
| `type` | enum | Yes | Memory type: `episodic`, `semantic`, `procedural`, `emotional`, `sensory`, `working` |
| `context` | object | No | Additional context |
| `context.location` | string | No | Where this happened |
| `context.people` | string[] | No | People involved |
| `context.mood` | string | No | Emotional state |
| `context.activity` | string | No | What was happening |
| `context.tags` | string[] | No | Additional tags |
| `context.source` | string | No | Source of information |
| `importance` | number | No | Importance (0-1), defaults to auto-calculated |

### Returns

Memory object with:
- `id`: Unique identifier
- `timestamp`: Creation time
- `emotionalValence`: Calculated emotion (-1 to 1)
- `associations`: Related memory IDs
- All input parameters

### Example

```bash
claude -p 'Use store_memory to save: "Met with CEO about expansion plans" 
  with type "episodic", 
  context: {location: "Board room", people: ["CEO", "CFO"], mood: "excited"}, 
  importance: 0.9' \
  --mcp-config claude-code-mcp.json
```

## search_memories

Search for memories using natural language and filters.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `type` | enum | No | Filter by memory type |
| `minImportance` | number | No | Minimum importance (0-1) |
| `emotionalRange` | object | No | Emotional valence range |
| `emotionalRange.min` | number | No | Minimum emotion (-1 to 1) |
| `emotionalRange.max` | number | No | Maximum emotion (-1 to 1) |
| `limit` | number | No | Number of results (default: 10, max: 50) |
| `includeAssociations` | boolean | No | Include associated memories (default: false) |

### Returns

Array of memory objects matching the search criteria.

### Example

```bash
claude -p 'Use search_memories with query: "project meeting", 
  type: "episodic", 
  minImportance: 0.5, 
  limit: 5' \
  --mcp-config claude-code-mcp.json
```

## get_memory

Retrieve a specific memory by ID.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Memory ID |

### Returns

Memory object or null if not found.

### Example

```bash
claude -p 'Use get_memory to retrieve memory with id: "550e8400-e29b-41d4-a716-446655440000"' \
  --mcp-config claude-code-mcp.json
```

## update_memory

Update an existing memory.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Memory ID to update |
| `content` | string | No | New content |
| `importance` | number | No | New importance (0-1) |
| `context` | object | No | Updated context |
| `metadata` | object | No | Additional metadata |

### Returns

Updated memory object or null if not found.

### Example

```bash
claude -p 'Use update_memory for id: "550e8400-e29b-41d4-a716-446655440000" 
  with content: "Meeting rescheduled to next week", 
  importance: 0.8' \
  --mcp-config claude-code-mcp.json
```

## delete_memory

Delete a memory permanently.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Memory ID to delete |

### Returns

Success message or error.

### Example

```bash
claude -p 'Use delete_memory to remove memory with id: "550e8400-e29b-41d4-a716-446655440000"' \
  --mcp-config claude-code-mcp.json
```

## analyze_memories

Analyze memory patterns and connections.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timeRange` | object | No | Time range to analyze |
| `timeRange.start` | string | No | ISO date string for start |
| `timeRange.end` | string | No | ISO date string for end |
| `type` | enum | No | Filter by memory type |

### Returns

Analysis object containing:
- `totalMemories`: Total count
- `byType`: Count by memory type
- `averageImportance`: Mean importance score
- `averageEmotionalValence`: Mean emotional score
- `mostCommonTags`: Top 10 tags with counts

### Example

```bash
claude -p 'Use analyze_memories with timeRange: {
    start: "2024-01-01T00:00:00Z", 
    end: "2024-12-31T23:59:59Z"
  }' \
  --mcp-config claude-code-mcp.json
```

## Memory Object Structure

All memory objects follow this structure:

```typescript
{
  id: string;                    // Unique identifier
  content: string;               // Memory content
  type: MemoryType;             // Memory type
  timestamp: Date;              // Creation time
  importance: number;           // 0-1 scale
  emotionalValence: number;     // -1 to 1 scale
  associations: string[];       // Related memory IDs
  context: {
    location?: string;
    people?: string[];
    mood?: string;
    activity?: string;
    tags?: string[];
    source?: string;
  };
  metadata: Record<string, any>; // Additional data
  lastAccessed: Date;           // Last access time
  accessCount: number;          // Access counter
  decay: number;                // Memory decay factor
}
```

## Error Handling

All tools return errors in this format:

```json
{
  "error": "Error message",
  "details": "Additional error details"
}
```

Common errors:
- `Memory not found` - Invalid memory ID
- `Invalid memory type` - Unknown memory type
- `API key error` - OpenAI API issues
- `Database error` - Qdrant connection issues

## Rate Limits

- OpenAI API: Subject to your OpenAI plan limits
- Qdrant: No hard limits for local instance
- Recommended: Max 100 memories per minute

## Best Practices

1. **Batch Operations**: Group related memories together
2. **Use Associations**: Let the system create memory links
3. **Consistent Naming**: Use consistent names for people/places
4. **Regular Cleanup**: Delete old working memories
5. **Importance Scores**: Reserve 0.8+ for truly important memories