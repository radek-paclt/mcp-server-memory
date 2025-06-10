/**
 * Advanced Batch Processing and Caching Optimization
 * High-performance implementation for the MCP Memory Server
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { EventEmitter } from 'events';
import { LRUCache } from 'lru-cache';
import { Redis } from 'ioredis';
import { Memory, MemoryType } from '../types/memory';

// Request types for batch processing
export interface EmbeddingRequest {
  id: string;
  text: string;
  priority: number;
  timestamp: number;
  resolve: (result: EmbeddingResult) => void;
  reject: (error: Error) => void;
  metadata?: any;
}

export interface EmbeddingResult {
  id: string;
  embedding: Float32Array;
  model: string;
  tokens: number;
  latency: number;
}

export interface BatchProcessingConfig {
  maxBatchSize: number;
  maxWaitTime: number;
  maxTokensPerBatch: number;
  workerCount: number;
  priorityLevels: number;
  adaptiveBatching: boolean;
  enableCompression: boolean;
}

// Priority queue for batch processing
class PriorityQueue<T> {
  private items: Array<{ item: T; priority: number }> = [];
  
  enqueue(item: T, priority: number): void {
    this.items.push({ item, priority });
    this.items.sort((a, b) => b.priority - a.priority);
  }
  
  dequeue(): T | undefined {
    const result = this.items.shift();
    return result?.item;
  }
  
  peek(): T | undefined {
    return this.items[0]?.item;
  }
  
  size(): number {
    return this.items.length;
  }
  
  isEmpty(): boolean {
    return this.items.length === 0;
  }
  
  clear(): void {
    this.items = [];
  }
}

// Adaptive batch processor with intelligent batching
export class AdaptiveBatchProcessor extends EventEmitter {
  private queues: Map<number, PriorityQueue<EmbeddingRequest>> = new Map();
  private workers: Worker[] = [];
  private processing = false;
  private batchHistory: BatchMetrics[] = [];
  private performanceStats: PerformanceStats;
  private tokenEstimator: TokenEstimator;
  
  constructor(private config: BatchProcessingConfig) {
    super();
    
    // Initialize priority queues
    for (let i = 0; i < config.priorityLevels; i++) {
      this.queues.set(i, new PriorityQueue<EmbeddingRequest>());
    }
    
    this.performanceStats = new PerformanceStats();
    this.tokenEstimator = new TokenEstimator();
    this.initializeWorkers();
    this.startBatchProcessor();
  }
  
  private initializeWorkers(): void {
    for (let i = 0; i < this.config.workerCount; i++) {
      const worker = new Worker(__filename, {
        workerData: { workerId: i, config: this.config }
      });
      
      worker.on('message', (result) => {
        this.handleWorkerResult(result);
      });
      
      worker.on('error', (error) => {
        console.error(`Worker ${i} error:`, error);
        this.handleWorkerError(i, error);
      });
      
      this.workers.push(worker);
    }
  }
  
  // Add request to queue with priority calculation
  async addToQueue(request: Omit<EmbeddingRequest, 'resolve' | 'reject'>): Promise<EmbeddingResult> {
    return new Promise((resolve, reject) => {
      const priority = this.calculatePriority(request);
      const priorityLevel = Math.min(
        this.config.priorityLevels - 1,
        Math.floor(priority * this.config.priorityLevels)
      );
      
      const fullRequest: EmbeddingRequest = {
        ...request,
        resolve,
        reject,
        timestamp: Date.now()
      };
      
      const queue = this.queues.get(priorityLevel)!;
      queue.enqueue(fullRequest, priority);
      
      this.emit('requestQueued', {
        id: request.id,
        priority,
        priorityLevel,
        queueSize: queue.size()
      });
    });
  }
  
  private calculatePriority(request: Omit<EmbeddingRequest, 'resolve' | 'reject'>): number {
    let priority = request.priority || 0.5;
    
    // Boost priority for short texts (faster processing)
    const textLength = request.text.length;
    if (textLength < 100) priority += 0.1;
    else if (textLength > 10000) priority -= 0.1;
    
    // Age-based priority boost
    const age = Date.now() - request.timestamp;
    const ageBoost = Math.min(0.2, age / (5 * 60 * 1000)); // Max 0.2 boost after 5 minutes
    priority += ageBoost;
    
    // Metadata-based priority adjustments
    if (request.metadata?.urgent) priority += 0.3;
    if (request.metadata?.background) priority -= 0.2;
    
    return Math.max(0, Math.min(1, priority));
  }
  
  private async startBatchProcessor(): Promise<void> {
    while (true) {
      if (!this.processing) {
        this.processing = true;
        await this.processBatches();
        this.processing = false;
      }
      
      // Adaptive wait time based on queue pressure
      const totalQueueSize = Array.from(this.queues.values())
        .reduce((sum, queue) => sum + queue.size(), 0);
      
      const waitTime = totalQueueSize > 100 ? 10 : this.config.maxWaitTime;
      await this.sleep(waitTime);
    }
  }
  
  private async processBatches(): Promise<void> {
    // Process highest priority queues first
    for (let priorityLevel = this.config.priorityLevels - 1; priorityLevel >= 0; priorityLevel--) {
      const queue = this.queues.get(priorityLevel)!;
      
      if (queue.isEmpty()) continue;
      
      const batches = this.createOptimalBatches(queue);
      
      // Process batches in parallel across workers
      await Promise.all(
        batches.map((batch, index) => 
          this.processBatch(batch, index % this.workers.length)
        )
      );
    }
  }
  
  private createOptimalBatches(queue: PriorityQueue<EmbeddingRequest>): EmbeddingRequest[][] {
    const batches: EmbeddingRequest[][] = [];
    const items: EmbeddingRequest[] = [];
    
    // Extract all items from queue
    while (!queue.isEmpty()) {
      const item = queue.dequeue();
      if (item) items.push(item);
    }
    
    if (items.length === 0) return batches;
    
    // Group by similar characteristics for optimal batching
    const groups = this.groupByCharacteristics(items);
    
    for (const group of groups) {
      const groupBatches = this.createBatchesFromGroup(group);
      batches.push(...groupBatches);
    }
    
    return batches;
  }
  
  private groupByCharacteristics(items: EmbeddingRequest[]): EmbeddingRequest[][] {
    // Group by estimated token count for efficient API utilization
    const groups = new Map<string, EmbeddingRequest[]>();
    
    for (const item of items) {
      const tokens = this.tokenEstimator.estimate(item.text);
      const bucket = this.getTokenBucket(tokens);
      
      if (!groups.has(bucket)) {
        groups.set(bucket, []);
      }
      
      groups.get(bucket)!.push(item);
    }
    
    return Array.from(groups.values());
  }
  
  private getTokenBucket(tokens: number): string {
    if (tokens <= 50) return 'tiny';
    if (tokens <= 200) return 'small';
    if (tokens <= 1000) return 'medium';
    if (tokens <= 4000) return 'large';
    return 'xlarge';
  }
  
  private createBatchesFromGroup(group: EmbeddingRequest[]): EmbeddingRequest[][] {
    const batches: EmbeddingRequest[][] = [];
    let currentBatch: EmbeddingRequest[] = [];
    let currentTokens = 0;
    
    // Sort by priority within group
    group.sort((a, b) => b.priority - a.priority);
    
    for (const item of group) {
      const itemTokens = this.tokenEstimator.estimate(item.text);
      
      // Check if adding this item would exceed limits
      if (
        currentBatch.length >= this.config.maxBatchSize ||
        currentTokens + itemTokens > this.config.maxTokensPerBatch
      ) {
        if (currentBatch.length > 0) {
          batches.push(currentBatch);
        }
        currentBatch = [];
        currentTokens = 0;
      }
      
      currentBatch.push(item);
      currentTokens += itemTokens;
    }
    
    if (currentBatch.length > 0) {
      batches.push(currentBatch);
    }
    
    return batches;
  }
  
  private async processBatch(batch: EmbeddingRequest[], workerIndex: number): Promise<void> {
    const startTime = Date.now();
    const worker = this.workers[workerIndex];
    
    if (!worker) {
      throw new Error(`Worker ${workerIndex} not available`);
    }
    
    try {
      // Send batch to worker
      worker.postMessage({
        type: 'PROCESS_BATCH',
        batch: batch.map(item => ({
          id: item.id,
          text: item.text,
          metadata: item.metadata
        })),
        config: {
          model: this.selectOptimalModel(batch),
          retryAttempts: 3,
          enableCache: true
        }
      });
      
      // Track batch metrics
      const metrics: BatchMetrics = {
        batchSize: batch.length,
        totalTokens: batch.reduce((sum, item) => 
          sum + this.tokenEstimator.estimate(item.text), 0
        ),
        processingTime: 0,
        startTime,
        workerIndex,
        success: true
      };
      
      this.batchHistory.push(metrics);
      this.trimBatchHistory();
      
    } catch (error) {
      // Handle batch failure
      for (const item of batch) {
        item.reject(error instanceof Error ? error : new Error('Batch processing failed'));
      }
    }
  }
  
  private selectOptimalModel(batch: EmbeddingRequest[]): string {
    // Analyze batch characteristics to select optimal model
    const avgTextLength = batch.reduce((sum, item) => 
      sum + item.text.length, 0) / batch.length;
    
    const hasHighPriority = batch.some(item => item.priority > 0.8);
    
    // Model selection logic
    if (hasHighPriority && avgTextLength < 500) {
      return 'text-embedding-3-small'; // Fast for urgent, short texts
    } else if (avgTextLength > 2000) {
      return 'text-embedding-3-large'; // Better quality for long texts
    } else {
      return 'text-embedding-3-small'; // Default balanced option
    }
  }
  
  private handleWorkerResult(result: any): void {
    if (result.type === 'BATCH_COMPLETE') {
      const { batchResults, metrics } = result;
      
      // Update performance stats
      this.performanceStats.recordBatch(metrics);
      
      // Resolve individual requests
      for (const embedResult of batchResults) {
        this.emit('embeddingComplete', embedResult);
      }
    } else if (result.type === 'BATCH_ERROR') {
      console.error('Batch processing error:', result.error);
      this.emit('batchError', result);
    }
  }
  
  private handleWorkerError(workerIndex: number, error: Error): void {
    console.error(`Worker ${workerIndex} crashed:`, error);
    
    // Restart worker
    this.workers[workerIndex] = new Worker(__filename, {
      workerData: { workerId: workerIndex, config: this.config }
    });
  }
  
  private trimBatchHistory(): void {
    // Keep only recent batch history for adaptive optimization
    const maxHistory = 1000;
    if (this.batchHistory.length > maxHistory) {
      this.batchHistory = this.batchHistory.slice(-maxHistory);
    }
  }
  
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  // Performance monitoring methods
  getPerformanceStats(): any {
    return this.performanceStats.getStats();
  }
  
  getQueueStats(): any {
    const stats: any = {};
    
    for (const [level, queue] of this.queues) {
      stats[`priority_${level}`] = queue.size();
    }
    
    stats.total = Array.from(this.queues.values())
      .reduce((sum, queue) => sum + queue.size(), 0);
    
    return stats;
  }
}

// Multi-level caching system
export class MultiLevelCacheManager {
  private l1Cache: LRUCache<string, any>; // In-memory
  private l2Cache: Redis; // Redis
  private l3Cache: S3CacheAdapter; // S3 for large objects
  private compressionEnabled: boolean;
  private stats: CacheStats;
  
  constructor(config: CacheConfig) {
    // L1: Fast in-memory cache
    this.l1Cache = new LRUCache({
      max: config.l1MaxItems || 1000,
      ttl: config.l1TTL || 60 * 1000, // 1 minute
      updateAgeOnGet: true,
      sizeCalculation: this.calculateSize.bind(this)
    });
    
    // L2: Redis distributed cache
    this.l2Cache = new Redis({
      host: config.redisHost || 'localhost',
      port: config.redisPort || 6379,
      db: config.redisDB || 0,
      keyPrefix: 'mcpmem:cache:',
      enableOfflineQueue: false,
      maxRetriesPerRequest: 3,
      lazyConnect: true
    });
    
    // L3: S3 for large objects
    this.l3Cache = new S3CacheAdapter(config.s3Config);
    
    this.compressionEnabled = config.enableCompression || true;
    this.stats = new CacheStats();
    
    this.setupCacheWarming();
  }
  
  private calculateSize(value: any): number {
    if (typeof value === 'string') return value.length;
    return JSON.stringify(value).length;
  }
  
  // Intelligent get with promotion
  async get(key: string, options?: CacheGetOptions): Promise<any> {
    const startTime = Date.now();
    
    try {
      // Try L1 first
      const l1Result = this.l1Cache.get(key);
      if (l1Result !== undefined) {
        this.stats.recordHit('l1', Date.now() - startTime);
        return l1Result;
      }
      
      // Try L2
      const l2Data = await this.l2Cache.get(key);
      if (l2Data) {
        const parsed = this.deserialize(l2Data);
        
        // Promote to L1 if beneficial
        if (this.shouldPromoteToL1(key, parsed)) {
          this.l1Cache.set(key, parsed);
        }
        
        this.stats.recordHit('l2', Date.now() - startTime);
        return parsed;
      }
      
      // Try L3 for large objects
      if (options?.checkL3) {
        const l3Result = await this.l3Cache.get(key);
        if (l3Result) {
          // Promote to L2 if size permits
          if (this.calculateSize(l3Result) < 1024 * 1024) { // 1MB limit
            await this.l2Cache.setex(key, 3600, this.serialize(l3Result));
          }
          
          this.stats.recordHit('l3', Date.now() - startTime);
          return l3Result;
        }
      }
      
      // Cache miss
      this.stats.recordMiss(Date.now() - startTime);
      return null;
      
    } catch (error) {
      this.stats.recordError();
      throw error;
    }
  }
  
  // Intelligent set with tier selection
  async set(key: string, value: any, options?: CacheSetOptions): Promise<void> {
    const size = this.calculateSize(value);
    const ttl = options?.ttl || 3600;
    
    // Always try to set in L1 if size permits
    if (size < 10 * 1024) { // 10KB limit for L1
      this.l1Cache.set(key, value);
    }
    
    // Set in L2 for medium-sized objects
    if (size < 1024 * 1024) { // 1MB limit for L2
      await this.l2Cache.setex(key, ttl, this.serialize(value));
    } else {
      // Large objects go to L3
      await this.l3Cache.set(key, value, { ttl });
      
      // Store reference in L2
      await this.l2Cache.setex(
        key,
        ttl,
        this.serialize({ _ref: 'l3', size, key })
      );
    }
    
    this.stats.recordSet(size);
  }
  
  private shouldPromoteToL1(key: string, value: any): boolean {
    // Decision based on access frequency and value size
    const size = this.calculateSize(value);
    const accessCount = this.stats.getAccessCount(key);
    
    // Promote if frequently accessed and reasonably sized
    return accessCount > 3 && size < 10 * 1024;
  }
  
  private serialize(value: any): string {
    let serialized = JSON.stringify(value);
    
    if (this.compressionEnabled && serialized.length > 1024) {
      // Use compression for larger values
      serialized = this.compress(serialized);
    }
    
    return serialized;
  }
  
  private deserialize(data: string): any {
    try {
      if (data.startsWith('compressed:')) {
        data = this.decompress(data.slice(11));
      }
      return JSON.parse(data);
    } catch (error) {
      console.error('Cache deserialization error:', error);
      return null;
    }
  }
  
  private compress(data: string): string {
    // Simple compression using zlib (in real implementation, use proper compression)
    const compressed = Buffer.from(data).toString('base64');
    return `compressed:${compressed}`;
  }
  
  private decompress(data: string): string {
    return Buffer.from(data, 'base64').toString();
  }
  
  // Predictive cache warming
  private setupCacheWarming(): void {
    setInterval(async () => {
      const predictions = await this.predictFutureAccess();
      
      for (const prediction of predictions.slice(0, 20)) { // Top 20 predictions
        if (prediction.probability > 0.7) {
          await this.warmCache(prediction.key);
        }
      }
    }, 60000); // Every minute
  }
  
  private async predictFutureAccess(): Promise<AccessPrediction[]> {
    // Simple time-series prediction based on access patterns
    const recentPatterns = this.stats.getRecentAccessPatterns();
    const predictions: AccessPrediction[] = [];
    
    for (const [key, pattern] of recentPatterns) {
      const trend = this.calculateTrend(pattern);
      const seasonality = this.calculateSeasonality(pattern);
      
      const probability = Math.min(1, trend * seasonality);
      predictions.push({ key, probability });
    }
    
    return predictions.sort((a, b) => b.probability - a.probability);
  }
  
  private calculateTrend(pattern: AccessPattern): number {
    // Simple linear trend calculation
    if (pattern.accesses.length < 2) return 0;
    
    const recent = pattern.accesses.slice(-5); // Last 5 accesses
    const timeDiffs = [];
    
    for (let i = 1; i < recent.length; i++) {
      timeDiffs.push(recent[i].timestamp - recent[i-1].timestamp);
    }
    
    const avgTimeDiff = timeDiffs.reduce((a, b) => a + b, 0) / timeDiffs.length;
    
    // Convert to probability (shorter intervals = higher probability)
    return Math.max(0, 1 - (avgTimeDiff / (24 * 60 * 60 * 1000))); // 24 hours
  }
  
  private calculateSeasonality(pattern: AccessPattern): number {
    // Weekly seasonality calculation
    const now = new Date();
    const dayOfWeek = now.getDay();
    const hourOfDay = now.getHours();
    
    const sameTimeAccesses = pattern.accesses.filter(access => {
      const accessDate = new Date(access.timestamp);
      return accessDate.getDay() === dayOfWeek && 
             Math.abs(accessDate.getHours() - hourOfDay) <= 1;
    });
    
    return Math.min(1, sameTimeAccesses.length / 5); // Normalize to 0-1
  }
  
  private async warmCache(key: string): Promise<void> {
    // Check if already cached
    const cached = await this.get(key);
    if (cached) return;
    
    // Trigger cache population (implementation specific)
    this.emit('cacheWarmRequest', { key });
  }
  
  // Cache statistics and monitoring
  getStats(): any {
    return {
      l1: {
        size: this.l1Cache.size,
        calculatedSize: this.l1Cache.calculatedSize
      },
      performance: this.stats.getStats()
    };
  }
  
  // Cache invalidation
  async invalidate(pattern: string): Promise<void> {
    // Invalidate in all cache levels
    
    // L1: Clear matching keys
    for (const key of this.l1Cache.keys()) {
      if (this.matchesPattern(key, pattern)) {
        this.l1Cache.delete(key);
      }
    }
    
    // L2: Use Redis pattern deletion
    const keys = await this.l2Cache.keys(pattern);
    if (keys.length > 0) {
      await this.l2Cache.del(...keys);
    }
    
    // L3: Implementation specific
    await this.l3Cache.invalidatePattern(pattern);
  }
  
  private matchesPattern(key: string, pattern: string): boolean {
    // Simple wildcard matching
    const regex = new RegExp(pattern.replace(/\*/g, '.*'));
    return regex.test(key);
  }
}

// Worker thread implementation
if (!isMainThread) {
  // Worker thread code
  const { workerId, config } = workerData;
  
  // Mock embedding service for worker
  class WorkerEmbeddingService {
    async generateEmbeddings(texts: string[], model: string): Promise<EmbeddingResult[]> {
      // Simulate API call latency
      await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));
      
      return texts.map((text, index) => ({
        id: `embedding_${index}`,
        embedding: new Float32Array(1536).map(() => Math.random() - 0.5),
        model,
        tokens: Math.ceil(text.length / 4),
        latency: 150
      }));
    }
  }
  
  const embeddingService = new WorkerEmbeddingService();
  
  parentPort?.on('message', async (message) => {
    if (message.type === 'PROCESS_BATCH') {
      try {
        const startTime = Date.now();
        const results = await embeddingService.generateEmbeddings(
          message.batch.map((item: any) => item.text),
          message.config.model
        );
        
        const processingTime = Date.now() - startTime;
        
        parentPort?.postMessage({
          type: 'BATCH_COMPLETE',
          batchResults: results,
          metrics: {
            batchSize: message.batch.length,
            processingTime,
            model: message.config.model,
            workerId
          }
        });
        
      } catch (error) {
        parentPort?.postMessage({
          type: 'BATCH_ERROR',
          error: error instanceof Error ? error.message : 'Unknown error',
          workerId
        });
      }
    }
  });
}

// Supporting classes and interfaces
interface BatchMetrics {
  batchSize: number;
  totalTokens: number;
  processingTime: number;
  startTime: number;
  workerIndex: number;
  success: boolean;
}

class PerformanceStats {
  private batchMetrics: BatchMetrics[] = [];
  
  recordBatch(metrics: BatchMetrics): void {
    this.batchMetrics.push(metrics);
    this.trimMetrics();
  }
  
  private trimMetrics(): void {
    if (this.batchMetrics.length > 1000) {
      this.batchMetrics = this.batchMetrics.slice(-1000);
    }
  }
  
  getStats(): any {
    if (this.batchMetrics.length === 0) return {};
    
    const avgBatchSize = this.batchMetrics.reduce((sum, m) => sum + m.batchSize, 0) / this.batchMetrics.length;
    const avgProcessingTime = this.batchMetrics.reduce((sum, m) => sum + m.processingTime, 0) / this.batchMetrics.length;
    const successRate = this.batchMetrics.filter(m => m.success).length / this.batchMetrics.length;
    
    return {
      totalBatches: this.batchMetrics.length,
      avgBatchSize,
      avgProcessingTime,
      successRate,
      throughput: avgBatchSize / (avgProcessingTime / 1000) // items per second
    };
  }
}

class TokenEstimator {
  private cache = new Map<string, number>();
  
  estimate(text: string): number {
    if (this.cache.has(text)) {
      return this.cache.get(text)!;
    }
    
    // Simple estimation: ~4 characters per token
    const estimate = Math.ceil(text.length / 4);
    
    this.cache.set(text, estimate);
    
    // Limit cache size
    if (this.cache.size > 10000) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    return estimate;
  }
}

// Cache statistics
class CacheStats {
  private hits = new Map<string, number>();
  private misses = 0;
  private errors = 0;
  private accessCounts = new Map<string, number>();
  private accessPatterns = new Map<string, AccessPattern>();
  
  recordHit(level: string, latency: number): void {
    this.hits.set(level, (this.hits.get(level) || 0) + 1);
  }
  
  recordMiss(latency: number): void {
    this.misses++;
  }
  
  recordError(): void {
    this.errors++;
  }
  
  recordSet(size: number): void {
    // Track set operations
  }
  
  getAccessCount(key: string): number {
    return this.accessCounts.get(key) || 0;
  }
  
  getRecentAccessPatterns(): Map<string, AccessPattern> {
    return this.accessPatterns;
  }
  
  getStats(): any {
    const totalHits = Array.from(this.hits.values()).reduce((a, b) => a + b, 0);
    const total = totalHits + this.misses;
    
    return {
      hitRate: total > 0 ? totalHits / total : 0,
      hits: Object.fromEntries(this.hits),
      misses: this.misses,
      errors: this.errors,
      total
    };
  }
}

// S3 cache adapter (mock implementation)
class S3CacheAdapter {
  constructor(private config: any) {}
  
  async get(key: string): Promise<any> {
    // Mock S3 get implementation
    return null;
  }
  
  async set(key: string, value: any, options?: any): Promise<void> {
    // Mock S3 set implementation
  }
  
  async invalidatePattern(pattern: string): Promise<void> {
    // Mock S3 invalidation
  }
}

// Type definitions
interface CacheGetOptions {
  checkL3?: boolean;
  maxAge?: number;
}

interface CacheSetOptions {
  ttl?: number;
  tier?: 'l1' | 'l2' | 'l3';
}

interface CacheConfig {
  l1MaxItems?: number;
  l1TTL?: number;
  redisHost?: string;
  redisPort?: number;
  redisDB?: number;
  s3Config?: any;
  enableCompression?: boolean;
}

interface AccessPrediction {
  key: string;
  probability: number;
}

interface AccessPattern {
  accesses: Array<{ timestamp: number; type: string }>;
  lastAccess: Date;
  frequency: number;
}

export {
  AdaptiveBatchProcessor,
  MultiLevelCacheManager,
  type BatchProcessingConfig,
  type CacheConfig,
  type EmbeddingRequest,
  type EmbeddingResult
};