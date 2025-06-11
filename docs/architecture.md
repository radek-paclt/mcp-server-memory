# Architecture

This document describes the system architecture of the MCP Memory Server.

## System Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Code    │────▶│  MCP Server     │────▶│  Qdrant DB      │
│  CLI Client     │◀────│  (Node.js)      │◀────│  Vector Store   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  OpenAI API     │
                        │  Embeddings     │
                        └─────────────────┘
```

## Component Architecture

### 1. MCP Server Layer

The main server implementing the Model Context Protocol:

```typescript
// src/index.ts
class MemoryMCPServer {
  - server: Server          // MCP SDK server
  - memoryService: Service  // Core memory logic
  
  + setupHandlers()         // Register MCP handlers
  + start()                 // Initialize and start
}
```

**Responsibilities:**
- Handle MCP protocol communication
- Route requests to appropriate services
- Manage tool registration
- Format responses

### 2. Memory Service Layer

Core business logic for memory operations:

```typescript
// src/services/memory.ts
class MemoryService {
  - qdrant: QdrantService
  - openai: OpenAIService
  
  + createMemory()          // Store new memory
  + searchMemories()        // Vector search
  + updateMemory()          // Update existing
  + deleteMemory()          // Remove memory
  + consolidateMemories()   // Future: memory consolidation
}
```

**Responsibilities:**
- Memory lifecycle management
- Association creation
- Importance calculation
- Context enrichment

### 3. Integration Services

#### Qdrant Service

```typescript
// src/services/qdrant.ts
class QdrantService {
  - client: QdrantClient
  - collectionName: string
  
  + initialize()            // Setup collection
  + upsertMemory()         // Store with vector
  + searchSimilar()        // Vector similarity search
  + getMemory()            // Retrieve by ID
  + deleteMemory()         // Remove from collection
}
```

**Responsibilities:**
- Vector database operations
- Collection management
- Similarity search
- Metadata storage

#### OpenAI Service

```typescript
// src/services/openai.ts
class OpenAIService {
  - client: OpenAI
  
  + createEmbedding()      // Text to vector
  + createEmbeddings()     // Batch embeddings
  + extractKeywords()      // Keyword extraction
  + analyzeEmotion()       // Emotional analysis
}
```

**Responsibilities:**
- Text embeddings generation
- Natural language processing
- Emotion analysis
- Keyword extraction

## Data Flow

### 1. Storing a Memory

```
User Input → MCP Handler → Memory Service
                              ↓
                    Extract Keywords (OpenAI)
                              ↓
                    Analyze Emotion (OpenAI)
                              ↓
                    Create Embedding (OpenAI)
                              ↓
                    Store in Qdrant
                              ↓
                    Find Associations
                              ↓
                    Return Memory Object
```

### 2. Searching Memories

```
Search Query → MCP Handler → Memory Service
                                ↓
                      Create Query Embedding
                                ↓
                      Vector Similarity Search
                                ↓
                      Apply Filters
                                ↓
                      Update Access Patterns
                                ↓
                      Include Associations
                                ↓
                      Return Results
```

## Memory Model

### Core Memory Structure

```typescript
interface Memory {
  // Identity
  id: string
  content: string
  type: MemoryType
  
  // Temporal
  timestamp: Date
  lastAccessed: Date
  accessCount: number
  
  // Characteristics
  importance: number        // 0-1
  emotionalValence: number  // -1 to 1
  decay: number            // 0-1
  
  // Relationships
  associations: string[]
  
  // Context
  context: MemoryContext
  metadata: Record<string, any>
}
```

### Memory Types

Each type has different characteristics:

| Type | Use Case | Default Importance | Decay Rate |
|------|----------|-------------------|------------|
| Episodic | Personal experiences | 0.7 | Medium |
| Semantic | Facts and knowledge | 0.5 | Low |
| Procedural | How-to information | 0.6 | Very Low |
| Emotional | Feelings | 0.8 | High |
| Sensory | Sensations | 0.4 | Very High |
| Working | Short-term | 0.3 | Very High |

## Vector Space Design

### Embedding Strategy

1. **Content Preparation**: Combine memory content with context
2. **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
3. **Distance Metric**: Cosine similarity

### Search Algorithm

```
1. Convert query to embedding
2. Find k-nearest neighbors in vector space
3. Apply post-filters (type, importance, date)
4. Boost by recency and access patterns
5. Include associated memories if requested
```

## Scalability Considerations

### Current Design (v1.0)

- Single Qdrant instance
- Synchronous processing
- In-memory caching minimal
- Direct OpenAI API calls

### Future Enhancements

1. **Caching Layer**
   - Redis for embedding cache
   - Memory access patterns cache
   - Common query results cache

2. **Async Processing**
   - Queue for embedding generation
   - Batch memory operations
   - Background association updates

3. **Distributed Architecture**
   ```
   Load Balancer
        ↓
   MCP Servers (N instances)
        ↓
   Qdrant Cluster
   ```

4. **Memory Consolidation**
   - Periodic memory review
   - Merge similar memories
   - Update decay factors
   - Archive old memories

## Security Architecture

### API Key Management

- Environment variables for secrets
- No keys in configuration files
- Separate keys for each service

### Data Privacy

- Memories stored with user context
- No cross-user memory access
- Encrypted transport (HTTPS)

### Access Control

- MCP tool permissions
- Rate limiting planned
- Audit logging planned

## Performance Optimization

### Current Optimizations

1. **Vector Search**: HNSW index in Qdrant
2. **Batch Operations**: Multiple embeddings in one call
3. **Lazy Loading**: Associations loaded on demand

### Planned Optimizations

1. **Embedding Cache**: Reuse common embeddings
2. **Query Optimization**: Pre-filter before vector search
3. **Connection Pooling**: Reuse database connections
4. **Memory Pagination**: Large result set handling

## Monitoring and Observability

### Current State

- Basic error logging
- MCP protocol logging
- API call tracking

### Future State

- Prometheus metrics
- Distributed tracing
- Performance dashboards
- Memory usage analytics

## Extension Points

The architecture supports extensions:

1. **New Memory Types**: Add to enum and behaviors
2. **Additional NLP**: Plug in new analysis services
3. **Storage Backends**: Abstract storage interface
4. **Custom Tools**: Add new MCP tool handlers