import { z } from 'zod';
import { MemoryType } from '../types/memory';

export const tools = {
  store_memory: {
    description: 'Store a new memory with automatic type detection. The system will intelligently determine whether this is an episodic (personal experience), semantic (factual knowledge), procedural (instructions), emotional, sensory, or working memory based on content analysis. You can optionally specify a type to override the automatic detection. Summary is auto-generated if not provided.',
    inputSchema: z.object({
      summary: z.string().optional().describe('Short 1-2 sentence summary capturing the essence of the memory. If not provided, auto-generated from content.'),
      content: z.string().describe('Full detailed memory content'),
      type: z.enum([
        'episodic',
        'semantic',
        'procedural',
        'emotional',
        'sensory',
        'working',
      ]).optional().describe('Memory type (auto-detected if not specified)'),
      context: z.object({
        location: z.string().optional().describe('Where this happened'),
        people: z.array(z.string()).optional().describe('People involved'),
        mood: z.string().optional().describe('Emotional state'),
        activity: z.string().optional().describe('What was happening'),
        tags: z.array(z.string()).optional().describe('Additional tags'),
        source: z.string().optional().describe('Source of information'),
      }).optional().describe('Context of the memory'),
      importance: z.number().min(0).max(1).optional().describe('Importance (0-1, auto-calculated if not specified)'),
    }),
  },

  store_memory_chunked: {
    description: 'Store a large memory with automatic semantic chunking and type detection. The system will analyze the content to determine the appropriate memory type and chunk large content intelligently.',
    inputSchema: z.object({
      summary: z.string().optional().describe('Short 1-2 sentence summary of the entire memory. If not provided, auto-generated.'),
      content: z.string().describe('The memory content (can be very long)'),
      type: z.enum([
        'episodic',
        'semantic',
        'procedural',
        'emotional',
        'sensory',
        'working',
      ]).optional().describe('Memory type (auto-detected if not specified)'),
      context: z.object({
        location: z.string().optional().describe('Where this happened'),
        people: z.array(z.string()).optional().describe('People involved'),
        mood: z.string().optional().describe('Emotional state'),
        activity: z.string().optional().describe('What was happening'),
        tags: z.array(z.string()).optional().describe('Additional tags'),
        source: z.string().optional().describe('Source of information'),
      }).optional().describe('Context of the memory'),
      importance: z.number().min(0).max(1).optional().describe('Importance (0-1, auto-calculated if not specified)'),
      chunkingMethod: z.enum(['semantic', 'sentence', 'paragraph', 'fixed']).optional().describe('Chunking method (default: semantic)'),
      maxChunkSize: z.number().optional().describe('Maximum chunk size in tokens (default: 1000)'),
    }),
  },

  search_memories: {
    description: 'Search through stored memories using natural language queries. Optionally filter by memory type, importance, date range, or emotional content.',
    inputSchema: z.object({
      query: z.string().optional().describe('Natural language search query'),
      type: z.enum([
        'episodic',
        'semantic',
        'procedural',
        'emotional',
        'sensory',
        'working',
      ]).optional().describe('Filter by memory type'),
      limit: z.number().optional().describe('Maximum number of results (default: 10)'),
      similarityThreshold: z.number().min(0).max(1).optional().describe('Similarity threshold (0-1, default: 0.3)'),
      importance: z.object({
        min: z.number().min(0).max(1).optional(),
        max: z.number().min(0).max(1).optional(),
      }).optional().describe('Filter by importance range'),
      dateRange: z.object({
        start: z.string().optional().describe('Start date (ISO format)'),
        end: z.string().optional().describe('End date (ISO format)'),
      }).optional().describe('Filter by date range'),
      emotionalRange: z.object({
        min: z.number().min(-1).max(1).optional(),
        max: z.number().min(-1).max(1).optional(),
      }).optional().describe('Filter by emotional valence range (-1 to 1)'),
      includeAssociations: z.boolean().optional().describe('Include associated memories in results'),
      format: z.enum(['compact', 'full']).optional().describe('Result format (default: compact)'),
    }),
  },

  get_memory: {
    description: 'Retrieve a specific memory by its ID with full details and associations.',
    inputSchema: z.object({
      id: z.string().describe('Memory ID'),
      includeAssociations: z.boolean().optional().describe('Include associated memories'),
    }),
  },

  update_memory: {
    description: 'Update an existing memory. Type will be re-detected automatically if content changes significantly.',
    inputSchema: z.object({
      id: z.string().describe('Memory ID to update'),
      content: z.string().optional().describe('New memory content'),
      summary: z.string().optional().describe('New summary'),
      type: z.enum([
        'episodic',
        'semantic',
        'procedural',
        'emotional',
        'sensory',
        'working',
      ]).optional().describe('Memory type (re-detected automatically if not specified)'),
      context: z.object({
        location: z.string().optional(),
        people: z.array(z.string()).optional(),
        mood: z.string().optional(),
        activity: z.string().optional(),
        tags: z.array(z.string()).optional(),
        source: z.string().optional(),
      }).optional().describe('Updated context'),
      importance: z.number().min(0).max(1).optional().describe('Updated importance'),
    }),
  },

  delete_memory: {
    description: 'Delete a specific memory by ID',
    inputSchema: z.object({
      id: z.string().describe('Memory ID to delete'),
    }),
  },

  delete_memories_bulk: {
    description: 'Delete multiple memories by their IDs',
    inputSchema: z.object({
      ids: z.array(z.string()).describe('Array of memory IDs to delete'),
    }),
  },

  delete_all_memories: {
    description: 'Delete ALL memories from the system. Use with extreme caution!',
    inputSchema: z.object({
      confirmation: z.literal('DELETE_ALL_MEMORIES').describe('Must be exactly "DELETE_ALL_MEMORIES" to confirm'),
    }),
  },

  analyze_memories: {
    description: 'Analyze stored memories to find patterns, statistics, and insights.',
    inputSchema: z.object({
      type: z.enum([
        'episodic',
        'semantic',
        'procedural',
        'emotional',
        'sensory',
        'working',
      ]).optional().describe('Filter analysis by memory type'),
      dateRange: z.object({
        start: z.string().optional(),
        end: z.string().optional(),
      }).optional().describe('Date range for analysis'),
      includeEmotionalAnalysis: z.boolean().optional().describe('Include emotional pattern analysis'),
      includeTemporalAnalysis: z.boolean().optional().describe('Include temporal pattern analysis'),
    }),
  },

  get_association_graph: {
    description: 'Get the association graph showing how memories are connected to each other.',
    inputSchema: z.object({
      memoryId: z.string().optional().describe('Focus on associations for a specific memory'),
      maxDepth: z.number().optional().describe('Maximum depth of associations to traverse'),
      type: z.enum([
        'episodic',
        'semantic',
        'procedural',
        'emotional',
        'sensory',
        'working',
      ]).optional().describe('Filter associations by memory type'),
    }),
  },

  find_memory_paths: {
    description: 'Find connection paths between two memories through their associations.',
    inputSchema: z.object({
      fromMemoryId: z.string().describe('Starting memory ID'),
      toMemoryId: z.string().describe('Target memory ID'),
      maxDepth: z.number().optional().describe('Maximum path length (default: 5)'),
    }),
  },

  connect_memories: {
    description: 'Manually create an association between two memories',
    inputSchema: z.object({
      fromMemoryId: z.string().describe('First memory ID'),
      toMemoryId: z.string().describe('Second memory ID'),
      bidirectional: z.boolean().optional().describe('Create bidirectional connection (default: true)'),
    }),
  },

  remove_association: {
    description: 'Remove an association between two memories',
    inputSchema: z.object({
      fromMemoryId: z.string().describe('First memory ID'),
      toMemoryId: z.string().describe('Second memory ID'),
      bidirectional: z.boolean().optional().describe('Remove bidirectional connection (default: true)'),
    }),
  },

  consolidate_memories: {
    description: 'Consolidate multiple related memories into a single memory using various strategies.',
    inputSchema: z.object({
      memoryIds: z.array(z.string()).describe('Array of memory IDs to consolidate'),
      strategy: z.enum(['merge_content', 'summarize', 'keep_most_important', 'create_composite']).describe('Consolidation strategy'),
      keepOriginals: z.boolean().optional().describe('Keep original memories (default: false)'),
      newType: z.enum([
        'episodic',
        'semantic',
        'procedural',
        'emotional',
        'sensory',
        'working',
      ]).optional().describe('Type for consolidated memory (auto-detected if not specified)'),
    }),
  },

  debug_memory_classification: {
    description: 'Debug the automatic memory type classification system by analyzing how content would be classified.',
    inputSchema: z.object({
      content: z.string().describe('Content to analyze for type classification'),
      showAnalysis: z.boolean().optional().describe('Show detailed analysis breakdown (default: true)'),
    }),
  },
};