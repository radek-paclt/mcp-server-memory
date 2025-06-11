import express from 'express';
import cors from 'cors';
import { randomUUID } from 'node:crypto';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import {
  isInitializeRequest,
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { InMemoryEventStore } from '@modelcontextprotocol/sdk/examples/shared/inMemoryEventStore.js';

import { MemoryService } from './services/memory';
import { MemoryType } from './types/memory';
import { tools } from './mcp/tools';
import { config } from './config/environment';
import { zodToJsonSchema } from './utils/zodToJsonSchema';
import { initializeTelemetry } from './config/telemetry';
import {
  memoryOperationsCounter,
  memoryOperationDuration,
  memoryErrorsCounter,
  mcpRequestsCounter,
  mcpRequestDuration,
  mcpActiveConnectionsGauge,
  mcpInferenceCounter,
  mcpInferenceDurationMs,
  mcpInferenceErrorsCounter,
  mcpInferenceTokensCounter,
} from './metrics/custom-metrics';

// Authentication middleware types
interface AuthOptions {
  enabled: boolean;
  apiKey?: string;
  bearerToken?: string;
}

class MemoryMCPHttpServer {
  private memoryService: MemoryService;
  private app: express.Application;
  private transports: Map<string, StreamableHTTPServerTransport | SSEServerTransport> = new Map();
  private authOptions: AuthOptions;

  constructor(authOptions: AuthOptions = { enabled: false }) {
    this.memoryService = new MemoryService();
    this.app = express();
    this.authOptions = authOptions;
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware() {
    // Enable CORS for cross-origin requests
    this.app.use(cors({
      origin: true,
      credentials: true,
      methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'mcp-session-id', 'last-event-id'],
    }));

    // Parse JSON requests
    this.app.use(express.json());

    // Note: Authentication middleware will be applied per route
  }

  private authenticateRequest(req: express.Request, res: express.Response, next: express.NextFunction): void {
    const authHeader = req.headers.authorization;
    
    if (!authHeader) {
      res.status(401).json({
        jsonrpc: '2.0',
        error: {
          code: -32000,
          message: 'Unauthorized: Missing authorization header',
        },
        id: null,
      });
      return;
    }

    // Support both API key and Bearer token authentication
    if (this.authOptions.apiKey && authHeader === `ApiKey ${this.authOptions.apiKey}`) {
      next();
      return;
    }

    if (this.authOptions.bearerToken && authHeader === `Bearer ${this.authOptions.bearerToken}`) {
      next();
      return;
    }

    res.status(401).json({
      jsonrpc: '2.0',
      error: {
        code: -32000,
        message: 'Unauthorized: Invalid credentials',
      },
      id: null,
    });
  }

  private createMCPServer(): Server {
    const server = new Server(
      {
        name: config.MCP_SERVER_NAME,
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupMCPHandlers(server);
    return server;
  }

  private setupMCPHandlers(server: Server) {
    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: Object.entries(tools).map(([name, tool]) => ({
        name,
        description: tool.description,
        inputSchema: zodToJsonSchema(tool.inputSchema),
      })),
    }));

    server.setRequestHandler(CallToolRequestSchema, async (request: any) => {
        const { name, arguments: args } = request.params;
        
        // Track MCP request metrics
        const startTime = Date.now();
        mcpRequestsCounter.add(1, { tool: name });
        
        // Track inference metrics (milliseconds)
        const inferenceStartTime = Date.now();
        mcpInferenceCounter.add(1, { tool: name });

        try {
          const result = await this.handleToolCall(name, args);
          return result;
        } catch (error) {
          // Track errors
          memoryErrorsCounter.add(1, { tool: name, error_type: error instanceof Error ? error.constructor.name : 'Unknown' });
          mcpInferenceErrorsCounter.add(1, { tool: name, error_type: error instanceof Error ? error.constructor.name : 'Unknown' });
          
          return {
            content: [
              {
                type: 'text',
                text: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
              },
            ],
          };
        } finally {
          // Track request duration (seconds)
          const duration = (Date.now() - startTime) / 1000;
          mcpRequestDuration.record(duration, { tool: name });
          
          // Track inference duration (milliseconds)
          const inferenceDurationMs = Date.now() - inferenceStartTime;
          mcpInferenceDurationMs.record(inferenceDurationMs, { tool: name });
        }
      }
    );
  }

  private async handleToolCall(name: string, args: any): Promise<any> {
    switch (name) {
      case 'store_memory': {
        const parsed = tools.store_memory.inputSchema.parse(args);
        const memory = await this.memoryService.createMemory(
          parsed.content,
          parsed.type as MemoryType,
          parsed.context,
          parsed.importance,
          parsed.summary
        );
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(memory, null, 2),
            },
          ],
        };
      }

      case 'store_memory_chunked': {
        const parsed = tools.store_memory_chunked.inputSchema.parse(args);
        const memories = await this.memoryService.createMemoryWithChunking(
          parsed.content,
          parsed.type as MemoryType,
          parsed.context,
          parsed.importance,
          parsed.chunkingOptions,
          parsed.summary
        );
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                message: `Created ${memories.length} memory chunks`,
                parentId: memories[0].id,
                chunkIds: memories.slice(1).map(m => m.id),
                memories: memories
              }, null, 2),
            },
          ],
        };
      }

      case 'search_memories': {
        const parsed = tools.search_memories.inputSchema.parse(args);
        const searchParams: any = {
          query: parsed.query,
          type: parsed.type as MemoryType | undefined,
          minImportance: parsed.minImportance,
          emotionalRange: parsed.emotionalRange,
          limit: parsed.limit,
          includeAssociations: parsed.includeAssociations,
          detailLevel: parsed.detailLevel,
          similarityThreshold: parsed.similarityThreshold,
          reconstructChunks: parsed.reconstructChunks,
          customer_id: parsed.customer_id,
          interaction_id: parsed.interaction_id,
        };
        
        // Add date range if provided
        if (parsed.dateRange) {
          searchParams.dateRange = {
            start: new Date(parsed.dateRange.start),
            end: new Date(parsed.dateRange.end),
          };
        }
        
        const memories = await this.memoryService.searchChunkedMemories(searchParams);
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(memories, null, 2),
            },
          ],
        };
      }

      case 'get_memory': {
        const parsed = tools.get_memory.inputSchema.parse(args);
        const memory = await this.memoryService.getMemory(parsed.id);
        return {
          content: [
            {
              type: 'text',
              text: memory ? JSON.stringify(memory, null, 2) : 'Memory not found',
            },
          ],
        };
      }

      case 'update_memory': {
        const parsed = tools.update_memory.inputSchema.parse(args);
        const memory = await this.memoryService.updateMemory(parsed.id, {
          content: parsed.content,
          importance: parsed.importance,
          ...(parsed.context && { context: parsed.context as any }),
          metadata: parsed.metadata,
        });
        return {
          content: [
            {
              type: 'text',
              text: memory ? JSON.stringify(memory, null, 2) : 'Memory not found',
            },
          ],
        };
      }

      case 'delete_memory': {
        const parsed = tools.delete_memory.inputSchema.parse(args);
        await this.memoryService.deleteMemory(parsed.id);
        return {
          content: [
            {
              type: 'text',
              text: `Memory ${parsed.id} deleted successfully`,
            },
          ],
        };
      }

      case 'analyze_memories': {
        const parsed = tools.analyze_memories.inputSchema.parse(args);
        const memories = await this.memoryService.searchMemories({
          query: '',
          type: parsed.type as MemoryType | undefined,
          limit: 100,
        });

        const analysis = {
          totalMemories: memories.length,
          byType: memories.reduce((acc, m) => {
            acc[m.type] = (acc[m.type] || 0) + 1;
            return acc;
          }, {} as Record<string, number>),
          averageImportance: memories.reduce((sum, m) => sum + m.importance, 0) / memories.length,
          averageEmotionalValence: memories.reduce((sum, m) => sum + m.emotionalValence, 0) / memories.length,
          mostCommonTags: this.getMostCommonTags(memories),
        };

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(analysis, null, 2),
            },
          ],
        };
      }

      case 'connect_memories': {
        const parsed = tools.connect_memories.inputSchema.parse(args);
        await this.memoryService.connectMemories(
          parsed.sourceId,
          parsed.targetId,
          parsed.bidirectional
        );
        return {
          content: [
            {
              type: 'text',
              text: `Connected memories ${parsed.sourceId} and ${parsed.targetId}`,
            },
          ],
        };
      }

      case 'find_memory_paths': {
        const parsed = tools.find_memory_paths.inputSchema.parse(args);
        const paths = await this.memoryService.findMemoryPaths(
          parsed.startId,
          parsed.endId,
          parsed.maxDepth,
          parsed.includeContent
        );
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(paths, null, 2),
            },
          ],
        };
      }

      case 'get_association_graph': {
        const parsed = tools.get_association_graph.inputSchema.parse(args);
        const graph = await this.memoryService.getAssociationGraph(
          parsed.centerMemoryId,
          parsed.depth,
          parsed.minImportance,
          parsed.includeContent
        );
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(graph, null, 2),
            },
          ],
        };
      }

      case 'consolidate_memories': {
        const parsed = tools.consolidate_memories.inputSchema.parse(args);
        const consolidated = await this.memoryService.consolidateMemories(
          parsed.memoryIds,
          parsed.strategy,
          parsed.keepOriginals
        );
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(consolidated, null, 2),
            },
          ],
        };
      }

      case 'remove_association': {
        const parsed = tools.remove_association.inputSchema.parse(args);
        await this.memoryService.removeAssociation(
          parsed.sourceId,
          parsed.targetId,
          parsed.bidirectional
        );
        return {
          content: [
            {
              type: 'text',
              text: `Removed association between ${parsed.sourceId} and ${parsed.targetId}`,
            },
          ],
        };
      }

      case 'delete_memories_bulk': {
        const parsed = tools.delete_memories_bulk.inputSchema.parse(args);
        const result = await this.memoryService.deleteMemories(parsed.ids);
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                message: `Deleted ${result.deleted} memories`,
                deleted: result.deleted,
                failed: result.failed,
                totalRequested: parsed.ids.length
              }, null, 2),
            },
          ],
        };
      }

      case 'delete_all_memories': {
        const parsed = tools.delete_all_memories.inputSchema.parse(args);
        if (parsed.confirm !== true) {
          throw new Error('Confirmation required to delete all memories');
        }
        const result = await this.memoryService.deleteAllMemories();
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                message: 'All memories have been deleted',
                deleted: result.deleted
              }, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  private getMostCommonTags(memories: any[]): Record<string, number> {
    const tagCounts: Record<string, number> = {};
    memories.forEach(m => {
      m.context.tags?.forEach((tag: string) => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
    });
    return Object.fromEntries(
      Object.entries(tagCounts)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
    );
  }

  private setupRoutes() {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({ status: 'ok', timestamp: new Date().toISOString() });
    });

    // Streamable HTTP Transport (protocol version 2025-03-26)
    this.app.all('/mcp', 
      ...(this.authOptions.enabled ? [(req: express.Request, res: express.Response, next: express.NextFunction) => this.authenticateRequest(req, res, next)] : []),
      async (req, res) => {
      console.log(`Received ${req.method} request to /mcp`);
      
      try {
        const sessionId = req.headers['mcp-session-id'] as string;
        let transport: StreamableHTTPServerTransport;

        if (sessionId && this.transports.has(sessionId)) {
          const existingTransport = this.transports.get(sessionId);
          if (existingTransport instanceof StreamableHTTPServerTransport) {
            transport = existingTransport;
          } else {
            res.status(400).json({
              jsonrpc: '2.0',
              error: {
                code: -32000,
                message: 'Bad Request: Session exists but uses a different transport protocol',
              },
              id: null,
            });
            return;
          }
        } else if (!sessionId && req.method === 'POST' && isInitializeRequest(req.body)) {
          const eventStore = new InMemoryEventStore();
          transport = new StreamableHTTPServerTransport({
            sessionIdGenerator: () => randomUUID(),
            eventStore,
            onsessioninitialized: (sessionId) => {
              console.log(`StreamableHTTP session initialized with ID: ${sessionId}`);
              this.transports.set(sessionId, transport);
              mcpActiveConnectionsGauge.add(1);
            }
          });

          transport.onclose = () => {
            const sid = transport.sessionId;
            if (sid && this.transports.has(sid)) {
              console.log(`Transport closed for session ${sid}`);
              this.transports.delete(sid);
              mcpActiveConnectionsGauge.add(-1);
            }
          };

          const server = this.createMCPServer();
          await server.connect(transport);
        } else {
          res.status(400).json({
            jsonrpc: '2.0',
            error: {
              code: -32000,
              message: 'Bad Request: No valid session ID provided',
            },
            id: null,
          });
          return;
        }

        await transport.handleRequest(req, res, req.body);
      } catch (error) {
        console.error('Error handling MCP request:', error);
        if (!res.headersSent) {
          res.status(500).json({
            jsonrpc: '2.0',
            error: {
              code: -32603,
              message: 'Internal server error',
            },
            id: null,
          });
        }
      }
    });

    // Deprecated HTTP+SSE Transport (protocol version 2024-11-05)
    this.app.get('/sse', 
      ...(this.authOptions.enabled ? [(req: express.Request, res: express.Response, next: express.NextFunction) => this.authenticateRequest(req, res, next)] : []),
      async (req, res) => {
      console.log('Received GET request to /sse (deprecated SSE transport)');
      
      const transport = new SSEServerTransport('/messages', res);
      this.transports.set(transport.sessionId, transport);
      mcpActiveConnectionsGauge.add(1);
      
      res.on('close', () => {
        this.transports.delete(transport.sessionId);
        mcpActiveConnectionsGauge.add(-1);
      });

      const server = this.createMCPServer();
      await server.connect(transport);
    });

    this.app.post('/messages', 
      ...(this.authOptions.enabled ? [(req: express.Request, res: express.Response, next: express.NextFunction) => this.authenticateRequest(req, res, next)] : []),
      async (req, res) => {
      const sessionId = req.query.sessionId as string;
      
      if (!sessionId || !this.transports.has(sessionId)) {
        res.status(400).json({
          jsonrpc: '2.0',
          error: {
            code: -32000,
            message: 'Bad Request: Invalid or missing session ID',
          },
          id: null,
        });
        return;
      }

      const transport = this.transports.get(sessionId);
      if (!(transport instanceof SSEServerTransport)) {
        res.status(400).json({
          jsonrpc: '2.0',
          error: {
            code: -32000,
            message: 'Bad Request: Session exists but uses a different transport protocol',
          },
          id: null,
        });
        return;
      }

      await transport.handlePostMessage(req, res, req.body);
    });
  }

  async start(port: number = 3000): Promise<void> {
    // Initialize telemetry first
    initializeTelemetry();
    
    // Initialize memory service
    await this.memoryService.initialize();

    // Start the HTTP server
    this.app.listen(port, () => {
      console.log(`MCP Memory HTTP Server listening on port ${port}`);
      console.log(`
==============================================
SUPPORTED TRANSPORT OPTIONS:

1. Streamable HTTP (Protocol version: 2025-03-26)
   Endpoint: /mcp
   Methods: GET, POST, DELETE
   Usage: 
     - Initialize with POST to /mcp
     - Establish SSE stream with GET to /mcp
     - Send requests with POST to /mcp
     - Terminate session with DELETE to /mcp

2. HTTP + SSE (Protocol version: 2024-11-05)
   Endpoints: /sse (GET) and /messages (POST)
   Usage:
     - Establish SSE stream with GET to /sse
     - Send requests with POST to /messages?sessionId=<id>

3. Health Check: GET /health

4. Metrics: GET /metrics (on port ${process.env.METRICS_PORT || 9090})

Authentication: ${this.authOptions.enabled ? 'Enabled' : 'Disabled'}
${this.authOptions.enabled ? `  - API Key: Authorization: ApiKey <key>
  - Bearer Token: Authorization: Bearer <token>` : ''}
==============================================
      `);
    });

    // Handle graceful shutdown
    process.on('SIGINT', this.shutdown.bind(this));
    process.on('SIGTERM', this.shutdown.bind(this));
  }

  private async shutdown(): Promise<void> {
    console.log('Shutting down HTTP server...');
    
    // Close all active transports
    for (const [sessionId, transport] of this.transports) {
      try {
        console.log(`Closing transport for session ${sessionId}`);
        await transport.close();
        this.transports.delete(sessionId);
      } catch (error) {
        console.error(`Error closing transport for session ${sessionId}:`, error);
      }
    }

    console.log('Server shutdown complete');
    process.exit(0);
  }
}

// Export for programmatic use
export { MemoryMCPHttpServer, AuthOptions };

// CLI usage when run directly
if (process.argv[1] && process.argv[1].endsWith('http-server.ts') || process.argv[1] && process.argv[1].endsWith('http-server.js')) {
  const port = parseInt(process.env.HTTP_PORT || '3000');
  
  // Parse authentication options from environment
  const authOptions: AuthOptions = {
    enabled: process.env.MCP_AUTH_ENABLED === 'true',
    apiKey: process.env.MCP_API_KEY,
    bearerToken: process.env.MCP_BEARER_TOKEN,
  };

  const server = new MemoryMCPHttpServer(authOptions);
  server.start(port).catch(console.error);
}