/**
 * Enhanced Memory Types for Customer Support Multi-Level Scoping
 * Supports Company → Customer → Interaction hierarchy
 */

export enum MemoryType {
  EPISODIC = 'episodic',
  SEMANTIC = 'semantic',
  PROCEDURAL = 'procedural',
  EMOTIONAL = 'emotional',
  SENSORY = 'sensory',
  WORKING = 'working',
}

export type MemoryScopeLevel = 'company' | 'customer' | 'interaction';

export interface EnhancedMemoryContext {
  // Original context fields
  location?: string;
  people?: string[];
  mood?: string;
  activity?: string;
  tags?: string[];
  source?: string;
  
  // NEW: Multi-level scoping fields
  customer_id?: string;
  interaction_id?: string;
  scope_level?: MemoryScopeLevel;
  
  // Customer Support specific fields
  support_agent?: string;
  ticket_id?: string;
  channel?: 'phone' | 'email' | 'chat' | 'in-person';
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  category?: string;
  subcategory?: string;
  resolution_status?: 'open' | 'in-progress' | 'resolved' | 'escalated';
  
  // Original chunking fields
  isParentChunk?: boolean;
  chunkIndex?: number;
  chunkOf?: string;
  totalChunks?: number;
  semanticDensity?: number;
}

export interface EnhancedMemory {
  id: string;
  content: string;
  summary: string;
  type: MemoryType;
  importance: number;
  emotionalValence: number;
  timestamp: Date;
  lastAccessed: Date;
  accessCount: number;
  associations: string[];
  context: EnhancedMemoryContext;
  embedding?: number[];
  
  // NEW: Computed fields for customer support
  scope_level: MemoryScopeLevel;
  is_customer_specific: boolean;
  is_interaction_specific: boolean;
  customer_context?: CustomerContext;
}

export interface CustomerContext {
  customer_id: string;
  customer_name?: string;
  customer_tier?: 'basic' | 'premium' | 'enterprise';
  account_status?: 'active' | 'suspended' | 'closed';
  interaction_count?: number;
  last_interaction?: Date;
  preferred_language?: string;
  communication_preference?: 'email' | 'phone' | 'chat';
}

export interface InteractionContext {
  interaction_id: string;
  customer_id: string;
  start_time: Date;
  end_time?: Date;
  channel: 'phone' | 'email' | 'chat' | 'in-person';
  support_agent: string;
  status: 'active' | 'completed' | 'abandoned';
  satisfaction_score?: number;
  resolution_time?: number; // minutes
}

export interface MemoryScopeFilter {
  customer_id?: string;
  interaction_id?: string;
  scope_level?: MemoryScopeLevel;
  inherit_from_parent_scopes?: boolean;
  include_company_knowledge?: boolean;
  exclude_other_customers?: boolean;
}

export interface EnhancedMemorySearchParams {
  query?: string;
  type?: MemoryType;
  limit?: number;
  similarityThreshold?: number;
  importance?: { min?: number; max?: number };
  dateRange?: { start?: Date; end?: Date };
  emotionalRange?: { min?: number; max?: number };
  includeAssociations?: boolean;
  format?: 'compact' | 'full';
  
  // NEW: Scoping parameters
  customer_id?: string;
  interaction_id?: string;
  scope_level?: MemoryScopeLevel;
  inherit_from_parent_scopes?: boolean;
  
  // Customer Support filters
  support_agent?: string;
  channel?: 'phone' | 'email' | 'chat' | 'in-person';
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  category?: string;
  resolution_status?: 'open' | 'in-progress' | 'resolved' | 'escalated';
  customer_tier?: 'basic' | 'premium' | 'enterprise';
}

export interface MemorySearchResult {
  memories: EnhancedMemory[];
  scope_breakdown: {
    company_level: number;
    customer_level: number;
    interaction_level: number;
  };
  relevance_scores: Record<string, number>;
  customer_context?: CustomerContext;
  interaction_context?: InteractionContext;
}

export interface CreateScopedMemoryParams {
  content: string;
  summary?: string;
  type?: MemoryType;
  importance?: number;
  
  // Scoping
  customer_id?: string;
  interaction_id?: string;
  
  // Context
  context?: Partial<EnhancedMemoryContext>;
  
  // Customer Support specific
  support_agent?: string;
  ticket_id?: string;
  channel?: 'phone' | 'email' | 'chat' | 'in-person';
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  category?: string;
  resolution_status?: 'open' | 'in-progress' | 'resolved' | 'escalated';
}

export interface MemoryScopeStats {
  total_memories: number;
  by_scope: Record<MemoryScopeLevel, number>;
  by_customer: Record<string, {
    customer_level: number;
    interaction_level: number;
    total_interactions: number;
  }>;
  company_knowledge_base: {
    total: number;
    by_category: Record<string, number>;
    by_type: Record<MemoryType, number>;
  };
}

// Utility types for complex operations
export type MemoryInheritanceRule = {
  scope: MemoryScopeLevel;
  weight: number;
  max_results: number;
};

export interface MemoryConsolidationOptions {
  scope: MemoryScopeLevel;
  customer_id?: string;
  interaction_id?: string;
  strategy: 'merge_similar' | 'summarize_interactions' | 'create_customer_profile';
  similarity_threshold?: number;
  time_window_days?: number;
}

// Customer Support Workflow Types
export interface SupportInteractionMemory {
  interaction_id: string;
  customer_id: string;
  agent_notes: EnhancedMemory[];
  customer_statements: EnhancedMemory[];
  resolution_steps: EnhancedMemory[];
  outcome_summary: EnhancedMemory;
  satisfaction_feedback?: EnhancedMemory;
}

export interface CustomerKnowledgeProfile {
  customer_id: string;
  preferences: EnhancedMemory[];
  issues_history: EnhancedMemory[];
  communication_style: EnhancedMemory[];
  product_usage: EnhancedMemory[];
  satisfaction_trends: EnhancedMemory[];
  escalation_patterns: EnhancedMemory[];
}

export interface CompanyKnowledgeBase {
  product_documentation: EnhancedMemory[];
  troubleshooting_guides: EnhancedMemory[];
  policy_information: EnhancedMemory[];
  training_materials: EnhancedMemory[];
  best_practices: EnhancedMemory[];
  common_solutions: EnhancedMemory[];
}

// Export compatibility with existing types
export type { Memory } from './memory';
export { MemoryType as OriginalMemoryType } from './memory';

// Type guards for scope detection
export function isCompanyScoped(memory: EnhancedMemory): boolean {
  return !memory.context.customer_id && !memory.context.interaction_id;
}

export function isCustomerScoped(memory: EnhancedMemory): boolean {
  return !!memory.context.customer_id && !memory.context.interaction_id;
}

export function isInteractionScoped(memory: EnhancedMemory): boolean {
  return !!memory.context.customer_id && !!memory.context.interaction_id;
}

export function getScopeLevel(context: EnhancedMemoryContext): MemoryScopeLevel {
  if (context.interaction_id && context.customer_id) return 'interaction';
  if (context.customer_id) return 'customer';
  return 'company';
}

export function canAccessMemory(
  memory: EnhancedMemory,
  requestContext: { customer_id?: string; interaction_id?: string }
): boolean {
  // Company-level memories are accessible to everyone
  if (isCompanyScoped(memory)) return true;
  
  // Customer-level memories are accessible to the same customer
  if (isCustomerScoped(memory)) {
    return memory.context.customer_id === requestContext.customer_id;
  }
  
  // Interaction-level memories are accessible within the same interaction
  if (isInteractionScoped(memory)) {
    return memory.context.customer_id === requestContext.customer_id &&
           memory.context.interaction_id === requestContext.interaction_id;
  }
  
  return false;
}