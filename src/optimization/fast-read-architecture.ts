/**
 * Fast Read Architecture with Async Write Pipeline
 * Target: <50-100ms reads, background async writes
 */

import Redis from 'ioredis';
import { EventEmitter } from 'events';

// ============= FAST READ LAYER (Target: <100ms) =============

class FastReadService {
  private redis: Redis;
  private qdrantRead: QdrantReadReplica;
  private embeddingCache: Map<string, number[]> = new Map();
  
  constructor() {
    this.redis = new Redis({
      host: 'localhost',
      port: 6379,
      maxRetriesPerRequest: 1, // Fail fast for reads
      lazyConnect: true,
      keyPrefix: 'memory:',
    });
  }

  /**
   * ULTRA-FAST SEARCH: Target 30-80ms
   */
  async searchMemoriesInstant(params: SearchParams): Promise<Memory[]> {
    const startTime = Date.now();
    
    try {
      // Step 1: Check cache for exact query (5-10ms)
      const cacheKey = this.getCacheKey(params);
      const cachedResults = await this.redis.get(cacheKey);
      if (cachedResults) {
        console.log(`Cache hit: ${Date.now() - startTime}ms`);
        return JSON.parse(cachedResults);
      }

      // Step 2: Get or generate embedding (15-30ms total)
      let queryEmbedding: number[];
      const embeddingCacheKey = `emb:${params.query}`;
      
      // Try embedding cache first (1-2ms)
      if (this.embeddingCache.has(embeddingCacheKey)) {
        queryEmbedding = this.embeddingCache.get(embeddingCacheKey)!;
      } else {
        // Check Redis embedding cache (3-5ms)
        const cachedEmbedding = await this.redis.get(embeddingCacheKey);
        if (cachedEmbedding) {
          queryEmbedding = JSON.parse(cachedEmbedding);
          this.embeddingCache.set(embeddingCacheKey, queryEmbedding);
        } else {
          // Generate new embedding asynchronously and return fast approximation
          return this.getFastApproximateResults(params, startTime);
        }
      }

      // Step 3: Fast vector search from read replica (20-40ms)
      const memories = await this.qdrantRead.searchSimilarFast(
        queryEmbedding, 
        params.limit || 10,
        params.similarityThreshold || 0.3
      );

      // Step 4: Enrich with cached metadata (5-10ms)
      const enrichedMemories = await this.enrichMemoriesFromCache(memories);

      // Step 5: Cache results for future requests (async, non-blocking)
      this.cacheResults(cacheKey, enrichedMemories);

      // Step 6: Update access patterns asynchronously (non-blocking)
      this.updateAccessPatternsAsync(memories.map(m => m.id));

      const totalTime = Date.now() - startTime;
      console.log(`Fast search completed: ${totalTime}ms`);
      
      return enrichedMemories;

    } catch (error) {
      // Fallback to degraded service rather than error
      console.error('Fast search failed, falling back:', error);
      return this.getFallbackResults(params);
    }
  }

  /**
   * INSTANT MEMORY RETRIEVAL: Target 10-30ms
   */
  async getMemoryInstant(id: string): Promise<Memory | null> {
    const startTime = Date.now();

    // Step 1: Try Redis cache (3-8ms)
    const cached = await this.redis.get(`mem:${id}`);
    if (cached) {
      console.log(`Memory cache hit: ${Date.now() - startTime}ms`);
      
      // Update access async (non-blocking)
      this.updateAccessPatternsAsync([id]);
      
      return JSON.parse(cached);
    }

    // Step 2: Read replica lookup (10-20ms)
    const memory = await this.qdrantRead.getMemoryFast(id);
    if (memory) {
      // Cache for next time (async)
      this.redis.setex(`mem:${id}`, 3600, JSON.stringify(memory));
      
      // Update access async (non-blocking)
      this.updateAccessPatternsAsync([id]);
      
      console.log(`Memory DB hit: ${Date.now() - startTime}ms`);
      return memory;
    }

    return null;
  }

  private async getFastApproximateResults(params: SearchParams, startTime: number): Promise<Memory[]> {
    // When embedding is not cached, return fast approximate results
    // based on keyword matching and filters only
    
    const approximateResults = await this.qdrantRead.searchByKeywords(
      params.query!,
      params.limit || 10
    );

    // Queue embedding generation for future requests (async)
    BackgroundProcessor.instance.queueEmbeddingGeneration(params.query!);

    console.log(`Fast approximate search: ${Date.now() - startTime}ms`);
    return approximateResults;
  }

  private async enrichMemoriesFromCache(memories: Memory[]): Promise<Memory[]> {
    // Batch get associated metadata from cache
    const pipeline = this.redis.pipeline();
    
    memories.forEach(memory => {
      memory.associations.forEach(assocId => {
        pipeline.get(`mem:${assocId}`);
      });
    });

    const results = await pipeline.exec();
    
    // Enrich memories with cached association data
    return memories.map(memory => ({
      ...memory,
      associatedMemories: memory.associations
        .map(assocId => results?.find(r => r[0] === null)?.[1])
        .filter(Boolean)
        .map(data => JSON.parse(data as string))
    }));
  }

  private updateAccessPatternsAsync(memoryIds: string[]): void {
    // Fire and forget - don't wait for completion
    BackgroundProcessor.instance.queueAccessUpdates(memoryIds);
  }

  private cacheResults(key: string, results: Memory[]): void {
    // Async cache write - don't block the response
    this.redis.setex(key, 300, JSON.stringify(results)); // 5min TTL
  }

  private getCacheKey(params: SearchParams): string {
    return `search:${JSON.stringify(params)}`;
  }
}

// ============= BACKGROUND PROCESSING LAYER =============

class BackgroundProcessor extends EventEmitter {
  static instance = new BackgroundProcessor();
  
  private writeQueue: WriteOperation[] = [];
  private embeddingQueue: EmbeddingOperation[] = [];
  private accessUpdateQueue: AccessUpdate[] = [];
  private isProcessing = false;

  private constructor() {
    super();
    this.startProcessingLoop();
  }

  /**
   * ASYNC MEMORY CREATION: Non-blocking write
   */
  async createMemoryAsync(params: CreateMemoryParams): Promise<{ id: string; status: 'queued' }> {
    // Generate temporary ID immediately
    const temporaryId = `temp_${Date.now()}_${Math.random()}`;

    // Create minimal memory record immediately for fast reads
    const minimalMemory: Memory = {
      id: temporaryId,
      content: params.content,
      type: params.type,
      importance: params.importance || 0.5,
      summary: params.content.substring(0, 200) + '...', // Temporary summary
      timestamp: new Date(),
      lastAccessed: new Date(),
      accessCount: 0,
      tags: [], // Will be filled by background processing
      associations: [],
      emotionalValence: 0, // Will be analyzed in background
      context: params.context || {},
      embedding: [], // Will be generated in background
    };

    // Store minimal version in cache immediately
    await this.cacheMinimalMemory(minimalMemory);

    // Queue full processing
    this.queueMemoryProcessing({
      temporaryId,
      params,
      timestamp: Date.now(),
    });

    return { id: temporaryId, status: 'queued' };
  }

  private async cacheMinimalMemory(memory: Memory): Promise<void> {
    const redis = new Redis();
    await redis.setex(`mem:${memory.id}`, 3600, JSON.stringify(memory));
    await redis.disconnect();
  }

  queueMemoryProcessing(operation: MemoryProcessingOperation): void {
    this.writeQueue.push({
      type: 'create_memory',
      operation,
      priority: 1,
      timestamp: Date.now(),
    });
    
    this.emit('queue_updated');
  }

  queueEmbeddingGeneration(text: string): void {
    this.embeddingQueue.push({
      text,
      timestamp: Date.now(),
    });
    
    this.emit('queue_updated');
  }

  queueAccessUpdates(memoryIds: string[]): void {
    this.accessUpdateQueue.push({
      memoryIds,
      timestamp: Date.now(),
    });
    
    this.emit('queue_updated');
  }

  private startProcessingLoop(): void {
    setInterval(() => {
      if (!this.isProcessing) {
        this.processQueues();
      }
    }, 100); // Process every 100ms
  }

  private async processQueues(): Promise<void> {
    this.isProcessing = true;

    try {
      // Process in priority order
      await this.processAccessUpdates(); // Fastest (metadata updates)
      await this.processEmbeddings();    // Medium (API calls)
      await this.processMemoryCreation(); // Slowest (full pipeline)
      
    } catch (error) {
      console.error('Background processing error:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  private async processAccessUpdates(): Promise<void> {
    const batch = this.accessUpdateQueue.splice(0, 50); // Process 50 at a time
    
    if (batch.length === 0) return;

    // Group by memory ID to avoid duplicate updates
    const uniqueUpdates = new Map<string, number>();
    
    batch.forEach(update => {
      update.memoryIds.forEach(id => {
        uniqueUpdates.set(id, Date.now());
      });
    });

    // Batch update metadata
    const qdrant = new QdrantService();
    const promises = Array.from(uniqueUpdates.entries()).map(([id, timestamp]) =>
      qdrant.updateMemoryMetadata(id, {
        lastAccessed: new Date(timestamp),
        accessCount: 1, // Increment logic would be more complex in real implementation
      }).catch(error => console.error(`Failed to update access for ${id}:`, error))
    );

    await Promise.allSettled(promises);
    console.log(`Processed ${uniqueUpdates.size} access updates`);
  }

  private async processEmbeddings(): Promise<void> {
    const batch = this.embeddingQueue.splice(0, 10); // Process 10 at a time
    
    if (batch.length === 0) return;

    const openai = new OpenAIService();
    const redis = new Redis();

    // Parallel embedding generation
    const promises = batch.map(async (op) => {
      try {
        const embedding = await openai.createEmbedding(op.text);
        
        // Cache the embedding
        const cacheKey = `emb:${op.text}`;
        await redis.setex(cacheKey, 86400, JSON.stringify(embedding)); // 24h TTL
        
        console.log(`Generated embedding for: ${op.text.substring(0, 50)}...`);
      } catch (error) {
        console.error(`Failed to generate embedding for: ${op.text}`, error);
      }
    });

    await Promise.allSettled(promises);
    await redis.disconnect();
  }

  private async processMemoryCreation(): Promise<void> {
    const operation = this.writeQueue.shift(); // Process one at a time for complex operations
    
    if (!operation || operation.type !== 'create_memory') return;

    const { temporaryId, params } = operation.operation as MemoryProcessingOperation;

    try {
      // Full memory processing pipeline
      const openai = new OpenAIService();
      const qdrant = new QdrantService();

      // Parallel API calls for efficiency
      const [summary, emotionalValence, tags, embedding] = await Promise.all([
        params.summary || openai.generateSummary(params.content),
        openai.analyzeEmotion(params.content),
        openai.extractKeywords(params.content),
        openai.createEmbedding(params.content),
      ]);

      // Create full memory object
      const fullMemory: Memory = {
        id: temporaryId,
        content: params.content,
        type: params.type,
        importance: params.importance || 0.5,
        summary,
        emotionalValence,
        tags,
        embedding,
        timestamp: new Date(),
        lastAccessed: new Date(),
        accessCount: 0,
        associations: [],
        context: params.context || {},
      };

      // Store in Qdrant
      await qdrant.upsertMemory(fullMemory, embedding);

      // Update cache with full memory
      const redis = new Redis();
      await redis.setex(`mem:${temporaryId}`, 3600, JSON.stringify(fullMemory));
      await redis.disconnect();

      // Queue association discovery (async)
      this.queueAssociationDiscovery(fullMemory);

      console.log(`Completed full processing for memory: ${temporaryId}`);

    } catch (error) {
      console.error(`Failed to process memory ${temporaryId}:`, error);
      
      // Could implement retry logic or error handling here
    }
  }

  private queueAssociationDiscovery(memory: Memory): void {
    // Queue association discovery as separate low-priority task
    setImmediate(async () => {
      try {
        const memoryService = new MemoryService();
        await memoryService.updateAssociations(memory);
      } catch (error) {
        console.error(`Failed to update associations for ${memory.id}:`, error);
      }
    });
  }
}

// ============= READ REPLICA SERVICE =============

class QdrantReadReplica {
  private client: any; // Qdrant client configured for read-only operations
  
  constructor() {
    // Configure for fast reads - could be different Qdrant instance
    // optimized for read performance
  }

  async searchSimilarFast(
    embedding: number[], 
    limit: number, 
    threshold: number
  ): Promise<Memory[]> {
    const results = await this.client.search(this.collectionName, {
      vector: embedding,
      limit,
      score_threshold: threshold,
      with_payload: true,
      // Optimize for speed over accuracy
      ef: 32, // Lower ef for faster search
      params: {
        hnsw_ef: 32,
      }
    });

    return results.map((result: any) => this.convertToMemory(result));
  }

  async getMemoryFast(id: string): Promise<Memory | null> {
    const results = await this.client.retrieve(this.collectionName, {
      ids: [id],
      with_payload: true,
    });

    return results.length > 0 ? this.convertToMemory(results[0]) : null;
  }

  async searchByKeywords(query: string, limit: number): Promise<Memory[]> {
    // Fast keyword-based search when embedding is not available
    // This could use full-text search capabilities of Qdrant or
    // pre-computed keyword indexes
    
    const filter = {
      must: [
        {
          key: 'tags',
          match: {
            any: query.toLowerCase().split(' ')
          }
        }
      ]
    };

    const results = await this.client.scroll(this.collectionName, {
      filter,
      limit,
      with_payload: true,
    });

    return results.points?.map((point: any) => this.convertToMemory(point)) || [];
  }

  private convertToMemory(result: any): Memory {
    // Convert Qdrant result to Memory object
    return {
      id: result.id,
      ...result.payload,
      timestamp: new Date(result.payload.timestamp),
      lastAccessed: new Date(result.payload.lastAccessed),
    };
  }
}

// ============= TYPE DEFINITIONS =============

interface SearchParams {
  query?: string;
  type?: string;
  limit?: number;
  similarityThreshold?: number;
  importance?: { min?: number; max?: number };
  dateRange?: { start?: Date; end?: Date };
}

interface CreateMemoryParams {
  content: string;
  type: string;
  summary?: string;
  importance?: number;
  context?: any;
}

interface WriteOperation {
  type: 'create_memory' | 'update_memory' | 'delete_memory';
  operation: any;
  priority: number;
  timestamp: number;
}

interface MemoryProcessingOperation {
  temporaryId: string;
  params: CreateMemoryParams;
  timestamp: number;
}

interface EmbeddingOperation {
  text: string;
  timestamp: number;
}

interface AccessUpdate {
  memoryIds: string[];
  timestamp: number;
}

interface Memory {
  id: string;
  content: string;
  type: string;
  importance: number;
  summary: string;
  timestamp: Date;
  lastAccessed: Date;
  accessCount: number;
  tags: string[];
  associations: string[];
  emotionalValence: number;
  context: any;
  embedding: number[];
  associatedMemories?: Memory[];
}

export {
  FastReadService,
  BackgroundProcessor,
  QdrantReadReplica,
  SearchParams,
  CreateMemoryParams,
  Memory,
};