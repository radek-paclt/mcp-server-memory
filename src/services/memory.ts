import { v4 as uuidv4 } from 'uuid';
import { Memory, MemoryType, MemorySearchParams, MemoryContext, CompactMemory, MemoryScope } from '../types/memory';
import { QdrantService } from './qdrant';
import { OpenAIService } from './openai';
import { ChunkingService, ChunkingOptions, TextChunk } from './chunking';
import {
  memoryOperationsCounter,
  memoryOperationDuration,
  memoryErrorsCounter,
} from '../metrics/custom-metrics';

export class MemoryService {
  private qdrant: QdrantService;
  private openai: OpenAIService;
  private chunking: ChunkingService;

  constructor() {
    this.qdrant = new QdrantService();
    this.openai = new OpenAIService();
    this.chunking = new ChunkingService();
  }

  async initialize(): Promise<void> {
    await this.qdrant.initialize();
  }

  async createMemoryWithChunking(
    content: string,
    type: MemoryType,
    context?: Partial<MemoryContext>,
    importance?: number,
    chunkingOptions?: ChunkingOptions,
    summary?: string
  ): Promise<Memory[]> {
    const startTime = Date.now();
    memoryOperationsCounter.add(1, { operation: 'create_chunked', type });
    
    try {
    // Use semantic chunking by default for long content
    const options: ChunkingOptions = chunkingOptions || {
      method: 'semantic',
      maxChunkSize: 1000, // tokens
      overlapSize: 100,   // tokens
      semanticThreshold: 0.75
    };
    
    // Check if content is long enough to warrant chunking
    const estimatedTokens = Math.ceil(content.length / 4);
    if (estimatedTokens <= (options.maxChunkSize || 1000)) {
      // Content is small enough, create single memory
      const memory = await this.createMemory(content, type, context, importance, summary);
      return [memory];
    }
    
    // Chunk the content
    const chunks = await this.chunking.chunkText(content, options);
    const memories: Memory[] = [];
    
    // Create a parent memory for the full content
    const parentSummary = summary || `Chunked memory containing ${chunks.length} parts about: ${content.substring(0, 100)}...`;
    const parentMemory = await this.createMemory(
      `[Chunked Memory - ${chunks.length} parts] ${content.substring(0, 200)}...`,
      type,
      { ...context, isParentChunk: true, totalChunks: chunks.length },
      importance || 0.8, // Parent chunks are generally important
      parentSummary
    );
    memories.push(parentMemory);
    
    // Create memories for each chunk
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkContext: Partial<MemoryContext> = {
        ...context,
        chunkIndex: i,
        chunkOf: parentMemory.id,
        totalChunks: chunks.length,
        semanticDensity: chunk.metadata?.semanticDensity
      };
      
      // Adjust importance based on semantic density
      const chunkImportance = importance || this.calculateImportance(chunk.content, type);
      const adjustedImportance = Math.min(1, chunkImportance * (chunk.metadata?.semanticDensity || 1));
      
      const chunkMemory = await this.createMemory(
        chunk.content,
        type,
        chunkContext,
        adjustedImportance
      );
      
      // Link chunk to parent
      await this.connectMemories(parentMemory.id, chunkMemory.id, true);
      
      memories.push(chunkMemory);
    }
    
    // Link adjacent chunks
    for (let i = 1; i < memories.length - 1; i++) {
      await this.connectMemories(memories[i].id, memories[i + 1].id, true);
    }
    
    return memories;
    } catch (error) {
      memoryErrorsCounter.add(1, { operation: 'create_chunked', error_type: error instanceof Error ? error.constructor.name : 'Unknown' });
      throw error;
    } finally {
      const duration = (Date.now() - startTime) / 1000;
      memoryOperationDuration.record(duration, { operation: 'create_chunked', type });
    }
  }

  async createMemory(
    content: string,
    type: MemoryType,
    context?: Partial<MemoryContext>,
    importance?: number,
    summary?: string
  ): Promise<Memory> {
    const startTime = Date.now();
    memoryOperationsCounter.add(1, { operation: 'create', type });
    
    try {
    // Validate content
    if (!content || content.trim().length === 0) {
      throw new Error('Memory content cannot be empty');
    }
    
    // Validate importance
    if (importance !== undefined && (importance < 0 || importance > 1)) {
      throw new Error('Importance must be between 0 and 1');
    }
    
    // Generate summary if not provided
    const memorySummary = summary || await this.generateSummary(content);
    
    // Determine scope based on context
    const scope = this.determineScope(context);
    
    const memory: Memory = {
      id: uuidv4(),
      summary: memorySummary,
      content,
      type,
      timestamp: new Date(),
      importance: importance ?? this.calculateImportance(content, type),
      emotionalValence: await this.openai.analyzeEmotion(content),
      associations: [],
      context: {
        scope,
        tags: await this.openai.extractKeywords(content),
        ...context,
      },
      metadata: {},
      lastAccessed: new Date(),
      accessCount: 0,
      decay: 0,
    };

    const embedding = await this.openai.createEmbedding(this.prepareMemoryForEmbedding(memory));
    await this.qdrant.upsertMemory(memory, embedding);

    // Find and link associations
    await this.updateAssociations(memory);

    return memory;
    } catch (error) {
      memoryErrorsCounter.add(1, { operation: 'create', error_type: error instanceof Error ? error.constructor.name : 'Unknown' });
      throw error;
    } finally {
      const duration = (Date.now() - startTime) / 1000;
      memoryOperationDuration.record(duration, { operation: 'create', type });
    }
  }

  async searchChunkedMemories(params: MemorySearchParams & { reconstructChunks?: boolean }): Promise<Memory[] | CompactMemory[]> {
    const memories = await this.searchMemories(params);
    
    if (!params.reconstructChunks || params.detailLevel === 'compact') {
      return memories;
    }
    
    // Find parent chunks and reconstruct full memories
    const reconstructed: Memory[] = [];
    const processedParents = new Set<string>();
    
    for (const memory of memories as Memory[]) {
      if (memory.context.isParentChunk && !processedParents.has(memory.id)) {
        // Get all chunks of this parent
        const chunks = await this.qdrant.searchByMetadata({
          chunkOf: memory.id
        });
        
        if (chunks.length > 0) {
          // Sort chunks by index
          chunks.sort((a, b) => (a.context.chunkIndex || 0) - (b.context.chunkIndex || 0));
          
          // Reconstruct full content
          const fullContent = chunks.map(c => c.content).join('\n\n');
          
          // Create reconstructed memory
          const reconstructedMemory: Memory = {
            ...memory,
            content: fullContent,
            metadata: {
              ...memory.metadata,
              reconstructed: true,
              chunkCount: chunks.length
            }
          };
          
          reconstructed.push(reconstructedMemory);
          processedParents.add(memory.id);
        } else {
          reconstructed.push(memory);
        }
      } else if (!memory.context.chunkOf) {
        // Regular non-chunked memory
        reconstructed.push(memory);
      }
      // Skip individual chunks if we're reconstructing
    }
    
    return reconstructed;
  }

  async searchMemories(params: MemorySearchParams): Promise<Memory[] | CompactMemory[]> {
    const startTime = Date.now();
    memoryOperationsCounter.add(1, { operation: 'search', type: params.type || 'all' });
    
    try {
    const filter: any = {
      must: [],
    };

    if (params.type) {
      filter.must.push({ key: 'type', match: { value: params.type } });
    }

    if (params.minImportance !== undefined) {
      filter.must.push({ key: 'importance', range: { gte: params.minImportance } });
    }

    if (params.emotionalRange) {
      filter.must.push({
        key: 'emotionalValence',
        range: { gte: params.emotionalRange.min, lte: params.emotionalRange.max },
      });
    }

    if (params.dateRange) {
      filter.must.push({
        key: 'timestamp',
        range: {
          gte: params.dateRange.start.toISOString(),
          lte: params.dateRange.end.toISOString(),
        },
      });
    }

    // Scope filtering logic
    const scopeFilter = this.buildScopeFilter(params.customer_id, params.interaction_id);
    if (scopeFilter) {
      if (scopeFilter.should) {
        // Multiple scope options - use should
        filter.should = scopeFilter.should;
      } else {
        // Single scope - use must
        filter.must.push(scopeFilter);
      }
    }

    let memories: Memory[];
    
    // If query is empty and we have filters, use filter-only search
    if (!params.query || params.query.trim() === '') {
      if (filter.must.length > 0) {
        memories = await this.qdrant.searchByFilters(filter, params.limit || 10);
      } else {
        // No query and no filters - return empty results
        memories = [];
      }
    } else {
      // Query provided - use similarity search
      const queryEmbedding = await this.openai.createEmbedding(params.query);
      memories = await this.qdrant.searchSimilar(
        queryEmbedding,
        params.limit || 10,
        filter.must.length > 0 ? filter : undefined,
        params.similarityThreshold
      );
    }

    // Update access patterns
    for (const memory of memories) {
      await this.qdrant.updateMemoryMetadata(memory.id, {
        lastAccessed: new Date(),
        accessCount: memory.accessCount + 1,
      });
    }

    // Include associations if requested
    if (params.includeAssociations) {
      const allMemories = [...memories];
      const processedIds = new Set(memories.map(m => m.id));

      for (const memory of memories) {
        for (const assocId of memory.associations) {
          if (!processedIds.has(assocId)) {
            const associated = await this.qdrant.getMemory(assocId);
            if (associated) {
              allMemories.push(associated);
              processedIds.add(assocId);
            }
          }
        }
      }

      return allMemories;
    }

    // Convert to compact format if requested
    if (params.detailLevel === 'compact') {
      return this.toCompactMemories(memories);
    }

    return memories;
    } catch (error) {
      memoryErrorsCounter.add(1, { operation: 'search', error_type: error instanceof Error ? error.constructor.name : 'Unknown' });
      throw error;
    } finally {
      const duration = (Date.now() - startTime) / 1000;
      memoryOperationDuration.record(duration, { operation: 'search' });
    }
  }

  private toCompactMemories(memories: Memory[]): CompactMemory[] {
    return memories.map(memory => ({
      id: memory.id,
      summary: memory.summary, // Use summary instead of truncated content
      type: memory.type,
      timestamp: memory.timestamp,
      importance: memory.importance,
      emotionalValence: memory.emotionalValence,
      tags: memory.context.tags
    }));
  }

  async getMemory(id: string): Promise<Memory | null> {
    const memory = await this.qdrant.getMemory(id);
    if (memory) {
      await this.qdrant.updateMemoryMetadata(id, {
        lastAccessed: new Date(),
        accessCount: memory.accessCount + 1,
      });
    }
    return memory;
  }

  async updateMemory(id: string, updates: Partial<Memory>): Promise<Memory | null> {
    const memory = await this.qdrant.getMemory(id);
    if (!memory) return null;

    // Validate updates
    if (updates.content !== undefined && updates.content.trim().length === 0) {
      throw new Error('Memory content cannot be empty');
    }
    
    if (updates.importance !== undefined && (updates.importance < 0 || updates.importance > 1)) {
      throw new Error('Importance must be between 0 and 1');
    }

    const updatedMemory = { ...memory, ...updates };
    
    if (updates.content) {
      const embedding = await this.openai.createEmbedding(
        this.prepareMemoryForEmbedding(updatedMemory)
      );
      await this.qdrant.upsertMemory(updatedMemory, embedding);
    } else {
      await this.qdrant.updateMemoryMetadata(id, updates);
    }

    return updatedMemory;
  }

  async deleteMemory(id: string): Promise<void> {
    await this.qdrant.deleteMemory(id);
  }

  async deleteMemories(ids: string[]): Promise<{ deleted: number; failed: string[] }> {
    const failed: string[] = [];
    let deleted = 0;

    // Remove associations to memories being deleted
    for (const id of ids) {
      const memory = await this.getMemory(id);
      if (memory) {
        // Remove this memory from all associations
        for (const assocId of memory.associations) {
          await this.removeAssociation(assocId, id, false);
        }
      }
    }

    // Delete the memories
    try {
      await this.qdrant.deleteMemories(ids);
      deleted = ids.length;
    } catch (error) {
      // If bulk delete fails, try individual deletes
      for (const id of ids) {
        try {
          await this.qdrant.deleteMemory(id);
          deleted++;
        } catch (err) {
          failed.push(id);
        }
      }
    }

    return { deleted, failed };
  }

  async deleteAllMemories(): Promise<{ deleted: boolean }> {
    try {
      await this.qdrant.deleteAllMemories();
      return { deleted: true };
    } catch (error) {
      throw new Error(`Failed to delete all memories: ${error}`);
    }
  }

  async consolidateMemories(
    memoryIds: string[],
    strategy: 'merge_content' | 'summarize' | 'keep_most_important' | 'create_composite',
    keepOriginals: boolean = false
  ): Promise<Memory> {
    const memories = await Promise.all(memoryIds.map(id => this.getMemory(id)));
    const validMemories = memories.filter((m): m is Memory => m !== null);
    
    if (validMemories.length < 2) {
      throw new Error('Need at least 2 valid memories to consolidate');
    }

    let consolidatedContent: string;
    let consolidatedImportance: number;
    let consolidatedContext: Partial<MemoryContext> = {};

    switch (strategy) {
      case 'merge_content':
        consolidatedContent = validMemories.map(m => m.content).join('\n\n---\n\n');
        consolidatedImportance = Math.max(...validMemories.map(m => m.importance));
        break;
      case 'summarize':
        consolidatedContent = await this.openai.summarizeTexts(validMemories.map(m => m.content));
        consolidatedImportance = validMemories.reduce((sum, m) => sum + m.importance, 0) / validMemories.length;
        break;
      case 'keep_most_important':
        const mostImportant = validMemories.reduce((max, m) => m.importance > max.importance ? m : max);
        consolidatedContent = mostImportant.content;
        consolidatedImportance = mostImportant.importance;
        consolidatedContext = mostImportant.context;
        break;
      case 'create_composite':
        consolidatedContent = `Composite memory from ${validMemories.length} sources:\n\n` +
          validMemories.map(m => `- ${m.content.substring(0, 100)}...`).join('\n');
        consolidatedImportance = Math.max(...validMemories.map(m => m.importance));
        break;
    }

    // Merge all tags and contexts
    const allTags = new Set<string>();
    validMemories.forEach(m => {
      m.context.tags?.forEach(tag => allTags.add(tag));
    });
    consolidatedContext.tags = Array.from(allTags);

    // Create the consolidated memory
    const consolidated = await this.createMemory(
      consolidatedContent,
      MemoryType.SEMANTIC,
      consolidatedContext,
      consolidatedImportance
    );

    // Link to all original memories
    for (const originalId of memoryIds) {
      await this.connectMemories(consolidated.id, originalId, false);
    }

    // Delete originals if requested
    if (!keepOriginals) {
      for (const id of memoryIds) {
        await this.deleteMemory(id);
      }
    }

    return consolidated;
  }

  async connectMemories(
    sourceId: string,
    targetId: string,
    bidirectional: boolean = true
  ): Promise<void> {
    const source = await this.getMemory(sourceId);
    const target = await this.getMemory(targetId);

    if (!source || !target) {
      throw new Error('One or both memories not found');
    }

    // Add association to source
    if (!source.associations.includes(targetId)) {
      await this.qdrant.updateMemoryMetadata(sourceId, {
        associations: [...source.associations, targetId].slice(0, 10), // Keep max 10 associations
      });
    }

    // Add reverse association if bidirectional
    if (bidirectional && !target.associations.includes(sourceId)) {
      await this.qdrant.updateMemoryMetadata(targetId, {
        associations: [...target.associations, sourceId].slice(0, 10),
      });
    }
  }

  async removeAssociation(
    sourceId: string,
    targetId: string,
    bidirectional: boolean = true
  ): Promise<void> {
    const source = await this.getMemory(sourceId);
    if (source && source.associations.includes(targetId)) {
      await this.qdrant.updateMemoryMetadata(sourceId, {
        associations: source.associations.filter(id => id !== targetId),
      });
    }

    if (bidirectional) {
      const target = await this.getMemory(targetId);
      if (target && target.associations.includes(sourceId)) {
        await this.qdrant.updateMemoryMetadata(targetId, {
          associations: target.associations.filter(id => id !== sourceId),
        });
      }
    }
  }

  async findMemoryPaths(
    startId: string,
    endId: string,
    maxDepth: number = 5,
    includeContent: boolean = false
  ): Promise<any> {
    const visited = new Set<string>();
    const paths: string[][] = [];

    const dfs = async (currentId: string, targetId: string, path: string[], depth: number) => {
      if (depth > maxDepth) return;
      if (currentId === targetId) {
        paths.push([...path, currentId]);
        return;
      }

      visited.add(currentId);
      const memory = await this.getMemory(currentId);
      
      if (memory) {
        for (const assocId of memory.associations) {
          if (!visited.has(assocId)) {
            await dfs(assocId, targetId, [...path, currentId], depth + 1);
          }
        }
      }
      visited.delete(currentId);
    };

    await dfs(startId, endId, [], 0);

    if (includeContent) {
      const pathsWithContent = await Promise.all(
        paths.map(async path => {
          const memories = await Promise.all(
            path.map(id => this.getMemory(id))
          );
          return memories.filter((m): m is Memory => m !== null);
        })
      );
      return pathsWithContent;
    }

    return paths;
  }

  async getAssociationGraph(
    centerMemoryId?: string,
    depth: number = 2,
    minImportance?: number,
    includeContent: boolean = false
  ): Promise<any> {
    const nodes = new Map<string, any>();
    const edges = new Set<string>();

    const explore = async (memoryId: string, currentDepth: number) => {
      if (currentDepth > depth) return;
      if (nodes.has(memoryId)) return;

      const memory = await this.getMemory(memoryId);
      if (!memory) return;
      if (minImportance && memory.importance < minImportance) return;

      nodes.set(memoryId, {
        id: memory.id,
        content: includeContent ? memory.content : memory.content.substring(0, 50) + '...',
        type: memory.type,
        importance: memory.importance,
        emotionalValence: memory.emotionalValence,
        timestamp: memory.timestamp,
      });

      for (const assocId of memory.associations) {
        const edgeKey = [memoryId, assocId].sort().join('-');
        edges.add(edgeKey);
        await explore(assocId, currentDepth + 1);
      }
    };

    if (centerMemoryId) {
      await explore(centerMemoryId, 0);
    } else {
      // If no center, get a sample of important memories
      const importantMemories = await this.searchMemories({
        query: '',
        minImportance: 0.7,
        limit: 10,
      });
      
      for (const memory of importantMemories) {
        await explore(memory.id, 0);
      }
    }

    return {
      nodes: Array.from(nodes.values()),
      edges: Array.from(edges).map(edge => {
        const [source, target] = edge.split('-');
        return { source, target };
      }),
      stats: {
        totalNodes: nodes.size,
        totalEdges: edges.size,
        averageConnections: edges.size / Math.max(nodes.size, 1),
      },
    };
  }

  private async generateSummary(content: string): Promise<string> {
    // Use OpenAI to generate proper summary
    return await this.openai.generateSummary(content);
  }

  private prepareMemoryForEmbedding(memory: Memory): string {
    const contextStr = Object.entries(memory.context)
      .filter(([_, v]) => v)
      .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(', ') : v}`)
      .join('; ');

    return `${memory.content}\n\nType: ${memory.type}\nContext: ${contextStr}`;
  }

  private calculateImportance(content: string, type: MemoryType): number {
    // Simple heuristic for importance
    let importance = 0.5;

    // Episodic memories are generally more important
    if (type === MemoryType.EPISODIC) importance += 0.2;
    
    // Emotional memories are important
    if (type === MemoryType.EMOTIONAL) importance += 0.15;
    
    // Longer content might be more detailed/important
    if (content.length > 200) importance += 0.1;
    
    // Cap at 1.0
    return Math.min(1.0, importance);
  }

  private async updateAssociations(memory: Memory): Promise<void> {
    // Find related memories
    const embedding = await this.openai.createEmbedding(
      this.prepareMemoryForEmbedding(memory)
    );
    
    const similar = await this.qdrant.searchSimilar(embedding, 5);
    
    // Filter out self and add strong associations
    const associations = similar
      .filter(m => m.id !== memory.id)
      .slice(0, 3)
      .map(m => m.id);

    if (associations.length > 0) {
      await this.qdrant.updateMemoryMetadata(memory.id, { associations });
      
      // Bidirectional associations
      for (const assocId of associations) {
        const assocMemory = await this.qdrant.getMemory(assocId);
        if (assocMemory && !assocMemory.associations.includes(memory.id)) {
          await this.qdrant.updateMemoryMetadata(assocId, {
            associations: [...assocMemory.associations, memory.id].slice(0, 5),
          });
        }
      }
    }
  }

  /**
   * Determine memory scope based on context
   */
  private determineScope(context?: Partial<MemoryContext>): MemoryScope {
    if (context?.interaction_id && context?.customer_id) {
      return MemoryScope.INTERACTION;
    }
    if (context?.customer_id) {
      return MemoryScope.CUSTOMER;
    }
    return MemoryScope.GENERAL;
  }

  /**
   * Build scope filter for Qdrant search
   */
  private buildScopeFilter(customer_id?: string, interaction_id?: string): any {
    if (customer_id && interaction_id) {
      // Interaction scope: see interaction + customer + general
      return {
        should: [
          { key: 'context.scope', match: { value: MemoryScope.GENERAL } },
          { 
            must: [
              { key: 'context.scope', match: { value: MemoryScope.CUSTOMER } },
              { key: 'context.customer_id', match: { value: customer_id } }
            ]
          },
          {
            must: [
              { key: 'context.scope', match: { value: MemoryScope.INTERACTION } },
              { key: 'context.customer_id', match: { value: customer_id } },
              { key: 'context.interaction_id', match: { value: interaction_id } }
            ]
          }
        ]
      };
    } else if (customer_id) {
      // Customer scope: see customer + general
      return {
        should: [
          { key: 'context.scope', match: { value: MemoryScope.GENERAL } },
          { 
            must: [
              { key: 'context.scope', match: { value: MemoryScope.CUSTOMER } },
              { key: 'context.customer_id', match: { value: customer_id } }
            ]
          }
        ]
      };
    } else {
      // General scope: see only general
      return { key: 'context.scope', match: { value: MemoryScope.GENERAL } };
    }
  }
}