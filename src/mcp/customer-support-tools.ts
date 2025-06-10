/**
 * MCP Tools for Customer Support Multi-Level Memory System
 * Supports Company → Customer → Interaction hierarchy
 */

import { z } from 'zod';

export const customerSupportTools = {
  // ============= SCOPED MEMORY STORAGE =============
  
  store_support_memory: {
    description: 'Store memory with customer support scoping. Automatically detects memory type and determines scope level based on provided IDs. Company level (no IDs), Customer level (customer_id only), or Interaction level (both IDs).',
    inputSchema: z.object({
      content: z.string().describe('Full memory content'),
      summary: z.string().optional().describe('Brief summary (auto-generated if not provided)'),
      type: z.enum([
        'episodic',
        'semantic', 
        'procedural',
        'emotional',
        'sensory',
        'working'
      ]).optional().describe('Memory type (auto-detected if not specified)'),
      
      // Scoping identifiers
      customer_id: z.string().optional().describe('Customer ID for customer/interaction scoped memories'),
      interaction_id: z.string().optional().describe('Interaction ID for conversation-specific memories'),
      
      // Customer support context
      support_agent: z.string().optional().describe('Support agent handling the interaction'),
      ticket_id: z.string().optional().describe('Support ticket reference'),
      channel: z.enum(['phone', 'email', 'chat', 'in-person']).optional().describe('Communication channel'),
      priority: z.enum(['low', 'medium', 'high', 'urgent']).optional().describe('Priority level'),
      category: z.string().optional().describe('Issue category (e.g., billing, technical, account)'),
      subcategory: z.string().optional().describe('Specific subcategory'),
      resolution_status: z.enum(['open', 'in-progress', 'resolved', 'escalated']).optional().describe('Current status'),
      
      // General context
      tags: z.array(z.string()).optional().describe('Additional tags for organization'),
      importance: z.number().min(0).max(1).optional().describe('Importance score (auto-calculated if not provided)'),
      location: z.string().optional().describe('Physical or virtual location'),
      people: z.array(z.string()).optional().describe('People involved'),
      mood: z.string().optional().describe('Emotional context'),
    }),
  },

  // ============= HIERARCHICAL SEARCH =============

  search_support_memories: {
    description: 'Search memories with customer support scoping and inheritance. Can search across all levels (company → customer → interaction) or focus on specific scope levels.',
    inputSchema: z.object({
      query: z.string().optional().describe('Natural language search query'),
      
      // Scoping parameters
      customer_id: z.string().optional().describe('Limit to specific customer'),
      interaction_id: z.string().optional().describe('Limit to specific interaction'),
      scope_level: z.enum(['company', 'customer', 'interaction']).optional().describe('Search only at specific scope level'),
      inherit_from_parent_scopes: z.boolean().default(true).describe('Include memories from broader scopes (company → customer → interaction)'),
      
      // Standard search filters
      type: z.enum(['episodic', 'semantic', 'procedural', 'emotional', 'sensory', 'working']).optional().describe('Filter by memory type'),
      limit: z.number().min(1).max(100).default(10).describe('Maximum results'),
      similarity_threshold: z.number().min(0).max(1).optional().describe('Minimum similarity score'),
      importance_range: z.object({
        min: z.number().min(0).max(1).optional(),
        max: z.number().min(0).max(1).optional()
      }).optional().describe('Filter by importance range'),
      date_range: z.object({
        start: z.string().optional().describe('Start date (ISO format)'),
        end: z.string().optional().describe('End date (ISO format)')
      }).optional().describe('Filter by date range'),
      
      // Customer support filters
      support_agent: z.string().optional().describe('Filter by support agent'),
      channel: z.enum(['phone', 'email', 'chat', 'in-person']).optional().describe('Filter by communication channel'),
      priority: z.enum(['low', 'medium', 'high', 'urgent']).optional().describe('Filter by priority'),
      category: z.string().optional().describe('Filter by issue category'),
      resolution_status: z.enum(['open', 'in-progress', 'resolved', 'escalated']).optional().describe('Filter by status'),
      
      // Result formatting
      include_associations: z.boolean().default(false).describe('Include related memories'),
      include_scope_breakdown: z.boolean().default(true).describe('Show count by scope level'),
      format: z.enum(['compact', 'full']).default('compact').describe('Detail level of results'),
    }),
  },

  // ============= CUSTOMER-SPECIFIC OPERATIONS =============

  get_customer_profile: {
    description: 'Retrieve complete customer knowledge profile including preferences, history, communication patterns, and interaction summary.',
    inputSchema: z.object({
      customer_id: z.string().describe('Customer ID'),
      include_interactions: z.boolean().default(false).describe('Include interaction-level memories'),
      max_memories_per_category: z.number().min(1).max(50).default(10).describe('Limit per category'),
      summary_format: z.enum(['brief', 'detailed']).default('brief').describe('Level of detail in summary'),
    }),
  },

  get_interaction_context: {
    description: 'Get comprehensive context for live customer support including current interaction, customer history, and relevant company knowledge.',
    inputSchema: z.object({
      customer_id: z.string().describe('Customer ID'),
      interaction_id: z.string().describe('Current interaction ID'),
      include_company_knowledge: z.boolean().default(true).describe('Include relevant company policies/procedures'),
      context_depth: z.enum(['minimal', 'standard', 'comprehensive']).default('standard').describe('Amount of context to retrieve'),
      prioritize_recent: z.boolean().default(true).describe('Prioritize recent customer interactions'),
    }),
  },

  search_customer_history: {
    description: 'Search across all interactions for a specific customer, including their complete history with the company.',
    inputSchema: z.object({
      customer_id: z.string().describe('Customer ID'),
      query: z.string().optional().describe('Search query within customer history'),
      include_company_knowledge: z.boolean().default(true).describe('Include company-level relevant knowledge'),
      time_range: z.object({
        start: z.string().optional().describe('Start date for history search'),
        end: z.string().optional().describe('End date for history search')
      }).optional().describe('Limit search to specific time period'),
      interaction_types: z.array(z.enum(['phone', 'email', 'chat', 'in-person'])).optional().describe('Filter by interaction types'),
      limit: z.number().min(1).max(100).default(20).describe('Maximum results'),
    }),
  },

  // ============= COMPANY KNOWLEDGE MANAGEMENT =============

  search_company_knowledge: {
    description: 'Search company-wide knowledge base including policies, procedures, product information, and best practices.',
    inputSchema: z.object({
      query: z.string().describe('Search query'),
      category: z.string().optional().describe('Filter by knowledge category'),
      type: z.enum(['episodic', 'semantic', 'procedural', 'emotional', 'sensory', 'working']).optional().describe('Filter by memory type'),
      department: z.string().optional().describe('Filter by originating department'),
      recency: z.enum(['any', 'last_week', 'last_month', 'last_quarter']).default('any').describe('Filter by recency'),
      limit: z.number().min(1).max(50).default(15).describe('Maximum results'),
    }),
  },

  store_company_knowledge: {
    description: 'Store company-wide knowledge that applies to all customers and interactions.',
    inputSchema: z.object({
      content: z.string().describe('Knowledge content'),
      summary: z.string().optional().describe('Brief summary'),
      category: z.string().describe('Knowledge category (e.g., policy, procedure, product)'),
      subcategory: z.string().optional().describe('Specific subcategory'),
      department: z.string().optional().describe('Originating department'),
      applies_to: z.array(z.string()).optional().describe('What this knowledge applies to'),
      effective_date: z.string().optional().describe('When this knowledge becomes effective'),
      expiry_date: z.string().optional().describe('When this knowledge expires'),
      tags: z.array(z.string()).optional().describe('Tags for organization'),
      importance: z.number().min(0).max(1).optional().describe('Importance level'),
    }),
  },

  // ============= ANALYTICS AND INSIGHTS =============

  analyze_customer_patterns: {
    description: 'Analyze patterns in customer interactions, satisfaction trends, and common issues.',
    inputSchema: z.object({
      customer_id: z.string().optional().describe('Analyze specific customer (or all customers if not provided)'),
      analysis_type: z.enum(['satisfaction', 'issues', 'communication', 'escalations', 'resolution_time']).describe('Type of analysis'),
      time_period: z.object({
        start: z.string().optional(),
        end: z.string().optional()
      }).optional().describe('Analysis time window'),
      group_by: z.enum(['day', 'week', 'month', 'quarter']).optional().describe('Grouping for temporal analysis'),
      include_trends: z.boolean().default(true).describe('Include trend analysis'),
    }),
  },

  get_memory_scope_stats: {
    description: 'Get statistics about memory distribution across company, customer, and interaction scopes.',
    inputSchema: z.object({
      include_customer_breakdown: z.boolean().default(true).describe('Include per-customer statistics'),
      include_category_breakdown: z.boolean().default(true).describe('Include breakdown by categories'),
      time_range: z.object({
        start: z.string().optional(),
        end: z.string().optional()
      }).optional().describe('Limit stats to specific time period'),
    }),
  },

  // ============= MEMORY MANAGEMENT =============

  consolidate_customer_memories: {
    description: 'Consolidate related memories for a customer, creating summary memories while preserving important details.',
    inputSchema: z.object({
      customer_id: z.string().describe('Customer ID'),
      consolidation_type: z.enum(['interactions', 'issues', 'preferences', 'all']).describe('What type of memories to consolidate'),
      time_window_days: z.number().min(1).max(365).default(30).describe('Consider memories within this time window'),
      similarity_threshold: z.number().min(0).max(1).default(0.7).describe('Minimum similarity for consolidation'),
      keep_originals: z.boolean().default(true).describe('Keep original memories after consolidation'),
      dry_run: z.boolean().default(false).describe('Preview consolidation without executing'),
    }),
  },

  transfer_interaction_memories: {
    description: 'Transfer memories from one interaction to another (e.g., when escalating or transferring calls).',
    inputSchema: z.object({
      source_interaction_id: z.string().describe('Source interaction ID'),
      target_interaction_id: z.string().describe('Target interaction ID'),
      customer_id: z.string().describe('Customer ID (must match for both interactions)'),
      memory_filter: z.object({
        types: z.array(z.enum(['episodic', 'semantic', 'procedural', 'emotional', 'sensory', 'working'])).optional(),
        importance_min: z.number().min(0).max(1).optional(),
        tags: z.array(z.string()).optional()
      }).optional().describe('Filter which memories to transfer'),
      copy_mode: z.enum(['copy', 'move']).default('copy').describe('Copy or move memories'),
    }),
  },

  // ============= VALIDATION AND DEBUGGING =============

  validate_memory_access: {
    description: 'Validate access permissions for memory operations in customer support context.',
    inputSchema: z.object({
      memory_id: z.string().describe('Memory ID to check'),
      customer_id: z.string().optional().describe('Customer context for access check'),
      interaction_id: z.string().optional().describe('Interaction context for access check'),
      agent_id: z.string().optional().describe('Agent requesting access'),
      operation: z.enum(['read', 'write', 'delete']).describe('Type of operation'),
    }),
  },

  debug_memory_scoping: {
    description: 'Debug memory scoping and inheritance for development and troubleshooting.',
    inputSchema: z.object({
      customer_id: z.string().optional().describe('Customer ID for scope testing'),
      interaction_id: z.string().optional().describe('Interaction ID for scope testing'),
      show_inheritance_path: z.boolean().default(true).describe('Show memory inheritance hierarchy'),
      show_access_rules: z.boolean().default(true).describe('Show access restriction rules'),
      test_query: z.string().optional().describe('Test query to see scoping in action'),
    }),
  },

  // ============= BULK OPERATIONS =============

  bulk_update_customer_memories: {
    description: 'Bulk update memories for a customer (e.g., updating customer tier, adding tags).',
    inputSchema: z.object({
      customer_id: z.string().describe('Customer ID'),
      updates: z.object({
        tags_to_add: z.array(z.string()).optional(),
        tags_to_remove: z.array(z.string()).optional(),
        category_update: z.string().optional(),
        importance_adjustment: z.number().min(-1).max(1).optional(),
        metadata_updates: z.record(z.any()).optional()
      }).describe('Updates to apply'),
      filter: z.object({
        memory_types: z.array(z.enum(['episodic', 'semantic', 'procedural', 'emotional', 'sensory', 'working'])).optional(),
        date_range: z.object({
          start: z.string().optional(),
          end: z.string().optional()
        }).optional(),
        interaction_ids: z.array(z.string()).optional()
      }).optional().describe('Filter which memories to update'),
      dry_run: z.boolean().default(false).describe('Preview changes without executing'),
    }),
  },

  export_customer_data: {
    description: 'Export customer memory data for compliance, analysis, or backup purposes.',
    inputSchema: z.object({
      customer_id: z.string().describe('Customer ID'),
      export_format: z.enum(['json', 'csv', 'markdown']).default('json').describe('Export format'),
      include_interaction_memories: z.boolean().default(true).describe('Include interaction-specific memories'),
      include_metadata: z.boolean().default(true).describe('Include all metadata'),
      anonymize_agents: z.boolean().default(false).describe('Remove agent identifiers'),
      date_range: z.object({
        start: z.string().optional(),
        end: z.string().optional()
      }).optional().describe('Limit export to date range'),
    }),
  },
};

// Type definitions for tool responses
export interface ScopedMemoryResult {
  memory: {
    id: string;
    content: string;
    summary: string;
    type: string;
    scope_level: 'company' | 'customer' | 'interaction';
    customer_id?: string;
    interaction_id?: string;
    importance: number;
    timestamp: string;
  };
  scope_info: {
    level: 'company' | 'customer' | 'interaction';
    inheritance_path: string[];
    access_restrictions: string[];
  };
}

export interface ScopedSearchResult {
  memories: any[];
  scope_breakdown: {
    company_level: number;
    customer_level: number;
    interaction_level: number;
  };
  relevance_scores: Record<string, number>;
  total_found: number;
  search_time_ms: number;
}

export interface CustomerProfileResult {
  customer_id: string;
  profile_summary: string;
  categories: {
    preferences: any[];
    issues_history: any[];
    communication_style: any[];
    product_usage: any[];
    satisfaction_trends: any[];
    escalation_patterns: any[];
  };
  interaction_stats: {
    total_interactions: number;
    avg_resolution_time: number;
    satisfaction_score: number;
    common_channels: string[];
  };
}

export interface InteractionContextResult {
  current_interaction: {
    interaction_id: string;
    customer_id: string;
    agent_notes: any[];
    customer_statements: any[];
    resolution_steps: any[];
    context_summary: string;
  };
  customer_history: any[];
  relevant_company_knowledge: any[];
  recommended_actions: string[];
  similar_cases: any[];
}

export default customerSupportTools;