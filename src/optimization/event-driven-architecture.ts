/**
 * Event-Driven Architecture Implementation
 * Demonstrates CQRS pattern with event sourcing for the MCP Memory Server
 */

import { EventEmitter } from 'events';
import { Redis } from 'ioredis';
import { v4 as uuidv4 } from 'uuid';
import { Memory, MemoryType } from '../types/memory';

// Event types for memory operations
export enum MemoryEventType {
  MEMORY_CREATED = 'memory.created',
  MEMORY_UPDATED = 'memory.updated',
  MEMORY_DELETED = 'memory.deleted',
  MEMORY_ACCESSED = 'memory.accessed',
  MEMORY_ASSOCIATED = 'memory.associated',
  MEMORY_CHUNKED = 'memory.chunked',
  MEMORY_CONSOLIDATED = 'memory.consolidated',
  EMBEDDING_GENERATED = 'embedding.generated',
  SEARCH_PERFORMED = 'search.performed'
}

// Base event interface
export interface MemoryEvent {
  id: string;
  type: MemoryEventType;
  timestamp: Date;
  aggregateId: string; // Memory ID
  payload: any;
  metadata: {
    userId?: string;
    requestId: string;
    version: number;
    causationId?: string; // ID of the event that caused this event
    correlationId?: string; // ID to track related events
  };
}

// Command interface for CQRS
export interface Command {
  id: string;
  type: string;
  payload: any;
  timestamp: Date;
  metadata: {
    userId?: string;
    priority?: number;
  };
}

// Event Store for event sourcing
export class EventStore {
  private redis: Redis;
  private eventEmitter: EventEmitter;
  
  constructor(redisConfig?: any) {
    this.redis = new Redis(redisConfig || {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      db: 1, // Use separate DB for events
      keyPrefix: 'events:',
      enableOfflineQueue: false,
      maxRetriesPerRequest: 3,
      retryStrategy: (times: number) => Math.min(times * 50, 2000)
    });
    
    this.eventEmitter = new EventEmitter();
    this.eventEmitter.setMaxListeners(100); // Support many listeners
  }
  
  // Append event to the event stream
  async append(event: MemoryEvent): Promise<void> {
    const streamKey = `stream:${event.aggregateId}`;
    const globalStreamKey = 'stream:global';
    
    // Store event data
    const eventData = JSON.stringify(event);
    
    // Use Redis Streams for ordered event storage
    const pipeline = this.redis.pipeline();
    
    // Add to aggregate-specific stream
    pipeline.xadd(
      streamKey,
      '*', // Auto-generate ID
      'event', eventData,
      'type', event.type,
      'version', event.metadata.version.toString()
    );
    
    // Add to global stream for projections
    pipeline.xadd(
      globalStreamKey,
      '*',
      'event', eventData,
      'aggregateId', event.aggregateId,
      'type', event.type
    );
    
    // Add to type-specific sorted set for efficient querying
    pipeline.zadd(
      `events:byType:${event.type}`,
      event.timestamp.getTime(),
      event.id
    );
    
    // Store full event for retrieval
    pipeline.set(`event:${event.id}`, eventData);
    
    await pipeline.exec();
    
    // Emit for real-time processing
    this.eventEmitter.emit('event', event);
    this.eventEmitter.emit(event.type, event);
  }
  
  // Read events for an aggregate
  async readStream(
    aggregateId: string,
    fromVersion?: number,
    toVersion?: number
  ): Promise<MemoryEvent[]> {
    const streamKey = `stream:${aggregateId}`;
    
    // Read from Redis Stream
    const entries = await this.redis.xrange(
      streamKey,
      fromVersion ? `0-${fromVersion}` : '-',
      toVersion ? `0-${toVersion}` : '+'
    );
    
    return entries.map(([id, fields]) => {
      const eventData = fields.find((f, i) => i % 2 === 0 && f === 'event');
      const eventIndex = fields.indexOf(eventData!);
      return JSON.parse(fields[eventIndex + 1]);
    });
  }
  
  // Subscribe to events
  on(eventType: string | MemoryEventType, handler: (event: MemoryEvent) => void): void {
    this.eventEmitter.on(eventType, handler);
  }
  
  // Get events by time range
  async getEventsByTimeRange(
    from: Date,
    to: Date,
    eventTypes?: MemoryEventType[]
  ): Promise<MemoryEvent[]> {
    const types = eventTypes || Object.values(MemoryEventType);
    const events: MemoryEvent[] = [];
    
    for (const type of types) {
      const eventIds = await this.redis.zrangebyscore(
        `events:byType:${type}`,
        from.getTime(),
        to.getTime()
      );
      
      const eventPromises = eventIds.map(id => 
        this.redis.get(`event:${id}`)
      );
      
      const rawEvents = await Promise.all(eventPromises);
      events.push(...rawEvents.filter(e => e).map(e => JSON.parse(e!)));
    }
    
    return events.sort((a, b) => 
      a.timestamp.getTime() - b.timestamp.getTime()
    );
  }
  
  // Create a snapshot for faster aggregate rebuilding
  async createSnapshot(aggregateId: string, state: any): Promise<void> {
    const snapshot = {
      aggregateId,
      state,
      version: state.version || 0,
      timestamp: new Date()
    };
    
    await this.redis.set(
      `snapshot:${aggregateId}`,
      JSON.stringify(snapshot)
    );
  }
  
  // Get latest snapshot
  async getSnapshot(aggregateId: string): Promise<any | null> {
    const data = await this.redis.get(`snapshot:${aggregateId}`);
    return data ? JSON.parse(data) : null;
  }
}

// Command Bus for CQRS
export class CommandBus {
  private handlers: Map<string, CommandHandler<any>> = new Map();
  private eventStore: EventStore;
  private commandQueue: Redis;
  
  constructor(eventStore: EventStore) {
    this.eventStore = eventStore;
    this.commandQueue = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      db: 2, // Separate DB for commands
      keyPrefix: 'commands:'
    });
  }
  
  // Register command handler
  register<T extends Command>(
    commandType: string,
    handler: CommandHandler<T>
  ): void {
    this.handlers.set(commandType, handler);
  }
  
  // Execute command
  async execute<T extends Command>(command: T): Promise<void> {
    const handler = this.handlers.get(command.type);
    
    if (!handler) {
      throw new Error(`No handler registered for command type: ${command.type}`);
    }
    
    // Add to command log for audit
    await this.logCommand(command);
    
    // Validate command
    const isValid = await handler.validate(command);
    if (!isValid) {
      throw new Error(`Invalid command: ${command.type}`);
    }
    
    // Execute command and collect events
    const events = await handler.handle(command);
    
    // Store events
    for (const event of events) {
      await this.eventStore.append(event);
    }
  }
  
  // Queue command for async processing
  async queue<T extends Command>(command: T): Promise<string> {
    const jobId = uuidv4();
    
    await this.commandQueue.lpush(
      'queue:commands',
      JSON.stringify({
        jobId,
        command,
        queuedAt: new Date()
      })
    );
    
    return jobId;
  }
  
  // Process queued commands
  async processQueue(): Promise<void> {
    while (true) {
      const data = await this.commandQueue.brpop('queue:commands', 0);
      if (!data) continue;
      
      const [, rawJob] = data;
      const job = JSON.parse(rawJob);
      
      try {
        await this.execute(job.command);
        
        // Mark as completed
        await this.commandQueue.set(
          `job:${job.jobId}:status`,
          'completed'
        );
      } catch (error) {
        // Mark as failed
        await this.commandQueue.set(
          `job:${job.jobId}:status`,
          'failed'
        );
        await this.commandQueue.set(
          `job:${job.jobId}:error`,
          error instanceof Error ? error.message : 'Unknown error'
        );
      }
    }
  }
  
  private async logCommand(command: Command): Promise<void> {
    await this.commandQueue.zadd(
      'log:commands',
      Date.now(),
      JSON.stringify(command)
    );
  }
}

// Abstract command handler
export abstract class CommandHandler<T extends Command> {
  constructor(protected eventStore: EventStore) {}
  
  abstract validate(command: T): Promise<boolean>;
  abstract handle(command: T): Promise<MemoryEvent[]>;
}

// Example: Create Memory Command Handler
export class CreateMemoryCommand implements Command {
  readonly type = 'CreateMemory';
  readonly id: string;
  readonly timestamp: Date;
  
  constructor(
    public payload: {
      content: string;
      type: MemoryType;
      importance?: number;
      context?: any;
    },
    public metadata: {
      userId?: string;
      priority?: number;
    } = {}
  ) {
    this.id = uuidv4();
    this.timestamp = new Date();
  }
}

export class CreateMemoryCommandHandler extends CommandHandler<CreateMemoryCommand> {
  async validate(command: CreateMemoryCommand): Promise<boolean> {
    const { content, type } = command.payload;
    
    // Validation rules
    if (!content || content.trim().length === 0) {
      return false;
    }
    
    if (!Object.values(MemoryType).includes(type)) {
      return false;
    }
    
    if (command.payload.importance !== undefined) {
      if (command.payload.importance < 0 || command.payload.importance > 1) {
        return false;
      }
    }
    
    return true;
  }
  
  async handle(command: CreateMemoryCommand): Promise<MemoryEvent[]> {
    const memoryId = uuidv4();
    const events: MemoryEvent[] = [];
    
    // Create memory created event
    const memoryCreatedEvent: MemoryEvent = {
      id: uuidv4(),
      type: MemoryEventType.MEMORY_CREATED,
      timestamp: new Date(),
      aggregateId: memoryId,
      payload: {
        id: memoryId,
        content: command.payload.content,
        type: command.payload.type,
        importance: command.payload.importance || 0.5,
        context: command.payload.context || {},
        timestamp: new Date()
      },
      metadata: {
        userId: command.metadata.userId,
        requestId: command.id,
        version: 1
      }
    };
    
    events.push(memoryCreatedEvent);
    
    // If content is large, emit chunking event
    if (command.payload.content.length > 4000) {
      const chunkingEvent: MemoryEvent = {
        id: uuidv4(),
        type: MemoryEventType.MEMORY_CHUNKED,
        timestamp: new Date(),
        aggregateId: memoryId,
        payload: {
          reason: 'content_too_large',
          originalSize: command.payload.content.length,
          chunkCount: Math.ceil(command.payload.content.length / 1000)
        },
        metadata: {
          userId: command.metadata.userId,
          requestId: command.id,
          version: 2,
          causationId: memoryCreatedEvent.id
        }
      };
      
      events.push(chunkingEvent);
    }
    
    return events;
  }
}

// Read Model Projections
export class MemoryProjection {
  private redis: Redis;
  private eventStore: EventStore;
  
  constructor(eventStore: EventStore) {
    this.eventStore = eventStore;
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      db: 3, // Separate DB for projections
      keyPrefix: 'projection:'
    });
    
    this.subscribeToEvents();
  }
  
  private subscribeToEvents(): void {
    // Update projections based on events
    this.eventStore.on(MemoryEventType.MEMORY_CREATED, async (event) => {
      await this.handleMemoryCreated(event);
    });
    
    this.eventStore.on(MemoryEventType.MEMORY_UPDATED, async (event) => {
      await this.handleMemoryUpdated(event);
    });
    
    this.eventStore.on(MemoryEventType.MEMORY_ACCESSED, async (event) => {
      await this.updateAccessStatistics(event);
    });
    
    this.eventStore.on(MemoryEventType.SEARCH_PERFORMED, async (event) => {
      await this.updateSearchStatistics(event);
    });
  }
  
  private async handleMemoryCreated(event: MemoryEvent): Promise<void> {
    const { id, type, importance, timestamp } = event.payload;
    
    // Update various projections
    const pipeline = this.redis.pipeline();
    
    // Memory by type counter
    pipeline.hincrby('stats:memories:byType', type, 1);
    
    // Total memories counter
    pipeline.incr('stats:memories:total');
    
    // Recent memories list
    pipeline.lpush('recent:memories', id);
    pipeline.ltrim('recent:memories', 0, 99); // Keep last 100
    
    // High importance memories set
    if (importance > 0.8) {
      pipeline.zadd('important:memories', importance, id);
    }
    
    // Daily creation stats
    const dateKey = timestamp.toISOString().split('T')[0];
    pipeline.hincrby(`stats:daily:${dateKey}`, 'created', 1);
    
    await pipeline.exec();
  }
  
  private async handleMemoryUpdated(event: MemoryEvent): Promise<void> {
    const { id, changes } = event.payload;
    
    // Track update frequency
    await this.redis.hincrby(`memory:${id}:stats`, 'updates', 1);
    
    // Update importance index if changed
    if (changes.importance !== undefined) {
      await this.redis.zadd('important:memories', changes.importance, id);
    }
  }
  
  private async updateAccessStatistics(event: MemoryEvent): Promise<void> {
    const { memoryId, timestamp } = event.payload;
    
    const pipeline = this.redis.pipeline();
    
    // Increment access count
    pipeline.hincrby(`memory:${memoryId}:stats`, 'accesses', 1);
    
    // Update last access time
    pipeline.hset(`memory:${memoryId}:stats`, 'lastAccess', timestamp.toISOString());
    
    // Update hot memories set (decay over time)
    const score = Date.now();
    pipeline.zadd('hot:memories', score, memoryId);
    
    // Trim old entries from hot memories
    const cutoff = Date.now() - (24 * 60 * 60 * 1000); // 24 hours
    pipeline.zremrangebyscore('hot:memories', 0, cutoff);
    
    await pipeline.exec();
  }
  
  private async updateSearchStatistics(event: MemoryEvent): Promise<void> {
    const { query, resultCount, responseTime } = event.payload;
    
    const pipeline = this.redis.pipeline();
    
    // Track search queries
    pipeline.zincrby('stats:searches:queries', 1, query);
    
    // Track response times
    pipeline.lpush('stats:searches:responseTimes', responseTime);
    pipeline.ltrim('stats:searches:responseTimes', 0, 999); // Keep last 1000
    
    // Track result counts
    pipeline.hincrby('stats:searches:resultCounts', 
      this.getResultCountBucket(resultCount), 1
    );
    
    await pipeline.exec();
  }
  
  private getResultCountBucket(count: number): string {
    if (count === 0) return '0';
    if (count <= 5) return '1-5';
    if (count <= 10) return '6-10';
    if (count <= 20) return '11-20';
    return '20+';
  }
  
  // Query methods for read model
  async getMemoryStats(): Promise<any> {
    const [
      total,
      byType,
      recentMemories,
      importantCount,
      hotMemories
    ] = await Promise.all([
      this.redis.get('stats:memories:total'),
      this.redis.hgetall('stats:memories:byType'),
      this.redis.lrange('recent:memories', 0, 9),
      this.redis.zcard('important:memories'),
      this.redis.zrevrange('hot:memories', 0, 9, 'WITHSCORES')
    ]);
    
    return {
      total: parseInt(total || '0'),
      byType,
      recentMemories,
      importantCount,
      hotMemories: this.parseZSetWithScores(hotMemories)
    };
  }
  
  private parseZSetWithScores(data: string[]): Array<{ id: string; score: number }> {
    const result = [];
    for (let i = 0; i < data.length; i += 2) {
      result.push({
        id: data[i],
        score: parseFloat(data[i + 1])
      });
    }
    return result;
  }
}

// Saga for complex workflows
export abstract class Saga {
  protected eventStore: EventStore;
  protected commandBus: CommandBus;
  private state: any = {};
  
  constructor(eventStore: EventStore, commandBus: CommandBus) {
    this.eventStore = eventStore;
    this.commandBus = commandBus;
  }
  
  abstract handle(event: MemoryEvent): Promise<void>;
  
  protected async emit(command: Command): Promise<void> {
    await this.commandBus.execute(command);
  }
  
  protected setState(newState: any): void {
    this.state = { ...this.state, ...newState };
  }
  
  protected getState(): any {
    return this.state;
  }
}

// Example: Memory Association Saga
export class MemoryAssociationSaga extends Saga {
  async handle(event: MemoryEvent): Promise<void> {
    switch (event.type) {
      case MemoryEventType.MEMORY_CREATED:
        await this.handleMemoryCreated(event);
        break;
      case MemoryEventType.EMBEDDING_GENERATED:
        await this.handleEmbeddingGenerated(event);
        break;
    }
  }
  
  private async handleMemoryCreated(event: MemoryEvent): Promise<void> {
    // Start the association workflow
    this.setState({
      memoryId: event.aggregateId,
      status: 'pending_embedding'
    });
    
    // Command to generate embedding
    await this.emit(new GenerateEmbeddingCommand({
      memoryId: event.aggregateId,
      content: event.payload.content
    }));
  }
  
  private async handleEmbeddingGenerated(event: MemoryEvent): Promise<void> {
    if (this.getState().memoryId !== event.aggregateId) {
      return; // Not our saga
    }
    
    // Command to find associations
    await this.emit(new FindAssociationsCommand({
      memoryId: event.aggregateId,
      embedding: event.payload.embedding,
      threshold: 0.8
    }));
    
    this.setState({ status: 'complete' });
  }
}

// Aggregate Root for Memory
export class MemoryAggregate {
  private id: string;
  private version: number = 0;
  private events: MemoryEvent[] = [];
  private state: any = {};
  
  constructor(id?: string) {
    this.id = id || uuidv4();
  }
  
  // Apply events to rebuild state
  async loadFromEvents(events: MemoryEvent[]): Promise<void> {
    for (const event of events) {
      await this.apply(event);
    }
  }
  
  // Apply a single event
  private async apply(event: MemoryEvent): Promise<void> {
    switch (event.type) {
      case MemoryEventType.MEMORY_CREATED:
        this.state = {
          ...event.payload,
          version: event.metadata.version
        };
        break;
      case MemoryEventType.MEMORY_UPDATED:
        this.state = {
          ...this.state,
          ...event.payload.changes,
          version: event.metadata.version
        };
        break;
      case MemoryEventType.MEMORY_DELETED:
        this.state.deleted = true;
        this.state.deletedAt = event.timestamp;
        break;
    }
    
    this.version = event.metadata.version;
  }
  
  // Create a new memory
  create(content: string, type: MemoryType, metadata: any): MemoryEvent[] {
    if (this.state.id) {
      throw new Error('Memory already exists');
    }
    
    const event: MemoryEvent = {
      id: uuidv4(),
      type: MemoryEventType.MEMORY_CREATED,
      timestamp: new Date(),
      aggregateId: this.id,
      payload: {
        id: this.id,
        content,
        type,
        ...metadata
      },
      metadata: {
        requestId: uuidv4(),
        version: 1
      }
    };
    
    this.events.push(event);
    return [event];
  }
  
  // Update memory
  update(changes: Partial<Memory>): MemoryEvent[] {
    if (!this.state.id || this.state.deleted) {
      throw new Error('Cannot update non-existent or deleted memory');
    }
    
    const event: MemoryEvent = {
      id: uuidv4(),
      type: MemoryEventType.MEMORY_UPDATED,
      timestamp: new Date(),
      aggregateId: this.id,
      payload: {
        changes
      },
      metadata: {
        requestId: uuidv4(),
        version: this.version + 1
      }
    };
    
    this.events.push(event);
    return [event];
  }
  
  getState(): any {
    return { ...this.state };
  }
  
  getUncommittedEvents(): MemoryEvent[] {
    return [...this.events];
  }
  
  markEventsAsCommitted(): void {
    this.events = [];
  }
}

// Additional helper classes
class GenerateEmbeddingCommand implements Command {
  readonly type = 'GenerateEmbedding';
  readonly id = uuidv4();
  readonly timestamp = new Date();
  
  constructor(
    public payload: { memoryId: string; content: string },
    public metadata: any = {}
  ) {}
}

class FindAssociationsCommand implements Command {
  readonly type = 'FindAssociations';
  readonly id = uuidv4();
  readonly timestamp = new Date();
  
  constructor(
    public payload: { memoryId: string; embedding: number[]; threshold: number },
    public metadata: any = {}
  ) {}
}