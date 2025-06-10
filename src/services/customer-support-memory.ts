/**
 * Customer Support Memory Service
 * Implements hierarchical memory scoping: Company → Customer → Interaction
 */

import { 
  EnhancedMemory, 
  EnhancedMemoryContext, 
  MemoryScopeLevel, 
  EnhancedMemorySearchParams,
  MemorySearchResult,
  CreateScopedMemoryParams,
  MemoryScopeStats,
  CustomerKnowledgeProfile,
  SupportInteractionMemory,
  getScopeLevel,
  canAccessMemory
} from '../types/memory-enhanced';
import { MemoryType } from '../types/memory';
import { MemoryService } from './memory';
import { MemoryTypeClassifier } from './memory-type-classifier';
import { QdrantService } from './qdrant';

export class CustomerSupportMemoryService extends MemoryService {
  private typeClassifier: MemoryTypeClassifier;

  constructor() {
    super();
    this.typeClassifier = new MemoryTypeClassifier();
  }

  /**
   * Create memory with automatic scoping and type detection
   */
  async createScopedMemory(params: CreateScopedMemoryParams): Promise<{
    memory: EnhancedMemory;
    scope_info: {
      level: MemoryScopeLevel;
      inheritance_path: string[];
      access_restrictions: string[];
    };
  }> {
    const startTime = Date.now();

    // Step 1: Determine scope level
    const scopeLevel = this.determineScopeLevel(params.customer_id, params.interaction_id);
    
    // Step 2: Auto-detect memory type if not provided
    let finalType = params.type;
    if (!finalType) {
      const classification = await this.typeClassifier.classifyMemoryType(params.content, params.context);
      finalType = classification.type;
      console.log(`Auto-detected type: ${finalType} (${(classification.confidence * 100).toFixed(1)}% confidence)`);
    }

    // Step 3: Build enhanced context
    const enhancedContext: EnhancedMemoryContext = {
      ...params.context,
      customer_id: params.customer_id,
      interaction_id: params.interaction_id,
      scope_level: scopeLevel,
      support_agent: params.support_agent,
      ticket_id: params.ticket_id,
      channel: params.channel,
      priority: params.priority,
      category: params.category,
      resolution_status: params.resolution_status,
    };

    // Step 4: Calculate importance based on scope and context
    const finalImportance = params.importance || this.calculateScopedImportance(
      params.content,
      finalType,
      scopeLevel,
      enhancedContext
    );

    // Step 5: Create memory
    const memory = await this.createMemory(
      params.content,
      finalType,
      enhancedContext,
      finalImportance,
      params.summary
    );

    // Step 6: Build enhanced memory object
    const enhancedMemory: EnhancedMemory = {
      ...memory,
      context: enhancedContext,
      scope_level: scopeLevel,
      is_customer_specific: !!params.customer_id,
      is_interaction_specific: !!params.interaction_id,
    };

    const scopeInfo = {
      level: scopeLevel,
      inheritance_path: this.getInheritancePath(params.customer_id, params.interaction_id),
      access_restrictions: this.getAccessRestrictions(scopeLevel, params.customer_id, params.interaction_id)
    };

    console.log(`Created ${scopeLevel}-scoped memory in ${Date.now() - startTime}ms`);

    return { memory: enhancedMemory, scope_info: scopeInfo };
  }

  /**
   * Search memories with hierarchical scoping and inheritance
   */
  async searchScopedMemories(params: EnhancedMemorySearchParams): Promise<MemorySearchResult> {
    const startTime = Date.now();

    // Step 1: Build scope filters
    const scopeFilters = this.buildScopeFilters(params);

    // Step 2: Execute searches for each scope level
    const searches = await Promise.all([
      this.searchCompanyLevel(params, scopeFilters),
      this.searchCustomerLevel(params, scopeFilters),
      this.searchInteractionLevel(params, scopeFilters)
    ]);

    const [companyMemories, customerMemories, interactionMemories] = searches;

    // Step 3: Combine and rank results
    const combinedMemories = this.combineAndRankMemories({
      company: companyMemories,
      customer: customerMemories,
      interaction: interactionMemories
    }, params);

    // Step 4: Apply final limit
    const finalMemories = combinedMemories.slice(0, params.limit || 10);

    // Step 5: Calculate relevance scores
    const relevanceScores = this.calculateRelevanceScores(finalMemories, params);

    const result: MemorySearchResult = {
      memories: finalMemories,
      scope_breakdown: {
        company_level: companyMemories.length,
        customer_level: customerMemories.length,
        interaction_level: interactionMemories.length
      },
      relevance_scores: relevanceScores
    };

    console.log(`Scoped search completed in ${Date.now() - startTime}ms`);
    return result;
  }

  /**
   * Get complete customer knowledge profile
   */
  async getCustomerProfile(customer_id: string): Promise<CustomerKnowledgeProfile> {
    const memories = await this.searchScopedMemories({
      customer_id,
      inherit_from_parent_scopes: false, // Only customer-level memories
      limit: 1000,
      format: 'full'
    });

    // Categorize memories by type and content
    const profile: CustomerKnowledgeProfile = {
      customer_id,
      preferences: [],
      issues_history: [],
      communication_style: [],
      product_usage: [],
      satisfaction_trends: [],
      escalation_patterns: []
    };

    for (const memory of memories.memories) {
      // Categorize based on content analysis and tags
      const category = this.categorizeCustomerMemory(memory);
      
      switch (category) {
        case 'preference':
          profile.preferences.push(memory);
          break;
        case 'issue':
          profile.issues_history.push(memory);
          break;
        case 'communication':
          profile.communication_style.push(memory);
          break;
        case 'product':
          profile.product_usage.push(memory);
          break;
        case 'satisfaction':
          profile.satisfaction_trends.push(memory);
          break;
        case 'escalation':
          profile.escalation_patterns.push(memory);
          break;
      }
    }

    return profile;
  }

  /**
   * Get interaction context for live support
   */
  async getInteractionContext(
    customer_id: string, 
    interaction_id: string
  ): Promise<{
    current_interaction: SupportInteractionMemory;
    customer_history: EnhancedMemory[];
    relevant_company_knowledge: EnhancedMemory[];
  }> {
    // Get all relevant context with inheritance
    const contextSearch = await this.searchScopedMemories({
      customer_id,
      interaction_id,
      inherit_from_parent_scopes: true,
      limit: 50,
      format: 'full'
    });

    // Separate by scope level
    const currentInteraction = contextSearch.memories.filter(m => 
      m.context.interaction_id === interaction_id
    );
    
    const customerHistory = contextSearch.memories.filter(m => 
      m.context.customer_id === customer_id && !m.context.interaction_id
    );
    
    const companyKnowledge = contextSearch.memories.filter(m => 
      !m.context.customer_id
    );

    // Build interaction memory structure
    const interactionMemory: SupportInteractionMemory = {
      interaction_id,
      customer_id,
      agent_notes: currentInteraction.filter(m => 
        m.context.tags?.includes('agent-note') || m.context.source === 'agent'
      ),
      customer_statements: currentInteraction.filter(m => 
        m.context.tags?.includes('customer-statement') || m.context.source === 'customer'
      ),
      resolution_steps: currentInteraction.filter(m => 
        m.context.tags?.includes('resolution') || m.type === MemoryType.PROCEDURAL
      ),
      outcome_summary: currentInteraction.find(m => 
        m.context.tags?.includes('outcome') || m.context.tags?.includes('summary')
      ) || currentInteraction[0] // fallback to first memory
    };

    return {
      current_interaction: interactionMemory,
      customer_history,
      relevant_company_knowledge: companyKnowledge
    };
  }

  /**
   * Analyze memory distribution across scopes
   */
  async getScopeStatistics(): Promise<MemoryScopeStats> {
    const allMemories = await this.searchMemories({ limit: 10000, format: 'compact' });
    
    const stats: MemoryScopeStats = {
      total_memories: allMemories.length,
      by_scope: {
        company: 0,
        customer: 0,
        interaction: 0
      },
      by_customer: {},
      company_knowledge_base: {
        total: 0,
        by_category: {},
        by_type: {
          [MemoryType.EPISODIC]: 0,
          [MemoryType.SEMANTIC]: 0,
          [MemoryType.PROCEDURAL]: 0,
          [MemoryType.EMOTIONAL]: 0,
          [MemoryType.SENSORY]: 0,
          [MemoryType.WORKING]: 0
        }
      }
    };

    for (const memory of allMemories) {
      const context = memory.context as EnhancedMemoryContext;
      const scopeLevel = getScopeLevel(context);
      
      stats.by_scope[scopeLevel]++;

      if (scopeLevel === 'company') {
        stats.company_knowledge_base.total++;
        stats.company_knowledge_base.by_type[memory.type]++;
        
        const category = context.category || 'uncategorized';
        stats.company_knowledge_base.by_category[category] = 
          (stats.company_knowledge_base.by_category[category] || 0) + 1;
      }

      if (context.customer_id) {
        if (!stats.by_customer[context.customer_id]) {
          stats.by_customer[context.customer_id] = {
            customer_level: 0,
            interaction_level: 0,
            total_interactions: 0
          };
        }

        if (scopeLevel === 'customer') {
          stats.by_customer[context.customer_id].customer_level++;
        } else if (scopeLevel === 'interaction') {
          stats.by_customer[context.customer_id].interaction_level++;
        }
      }
    }

    // Calculate unique interactions per customer
    for (const customer_id of Object.keys(stats.by_customer)) {
      const customerMemories = allMemories.filter(m => 
        (m.context as EnhancedMemoryContext).customer_id === customer_id
      );
      
      const uniqueInteractions = new Set(
        customerMemories
          .map(m => (m.context as EnhancedMemoryContext).interaction_id)
          .filter(Boolean)
      );
      
      stats.by_customer[customer_id].total_interactions = uniqueInteractions.size;
    }

    return stats;
  }

  /**
   * Validate access permissions for memory operations
   */
  validateAccess(
    memory: EnhancedMemory,
    requestContext: { customer_id?: string; interaction_id?: string; agent_id?: string }
  ): { allowed: boolean; reason?: string } {
    // Company-level memories are accessible to all agents
    if (memory.scope_level === 'company') {
      return { allowed: true };
    }

    // Customer-level memories require matching customer_id
    if (memory.scope_level === 'customer') {
      if (!requestContext.customer_id) {
        return { allowed: false, reason: 'Customer ID required for customer-scoped memory' };
      }
      if (memory.context.customer_id !== requestContext.customer_id) {
        return { allowed: false, reason: 'Cannot access other customer memories' };
      }
      return { allowed: true };
    }

    // Interaction-level memories require matching customer_id and interaction_id
    if (memory.scope_level === 'interaction') {
      if (!requestContext.customer_id || !requestContext.interaction_id) {
        return { allowed: false, reason: 'Customer and interaction IDs required' };
      }
      if (memory.context.customer_id !== requestContext.customer_id ||
          memory.context.interaction_id !== requestContext.interaction_id) {
        return { allowed: false, reason: 'Cannot access other interaction memories' };
      }
      return { allowed: true };
    }

    return { allowed: false, reason: 'Unknown scope level' };
  }

  // Private helper methods

  private determineScopeLevel(customer_id?: string, interaction_id?: string): MemoryScopeLevel {
    if (interaction_id && customer_id) return 'interaction';
    if (customer_id) return 'customer';
    return 'company';
  }

  private calculateScopedImportance(
    content: string,
    type: MemoryType,
    scope: MemoryScopeLevel,
    context: EnhancedMemoryContext
  ): number {
    let importance = 0.5; // Base importance

    // Scope-based adjustments
    switch (scope) {
      case 'company':
        importance += 0.1; // Company knowledge is valuable
        break;
      case 'customer':
        importance += 0.15; // Customer-specific info is important
        break;
      case 'interaction':
        importance += 0.05; // Interaction context is situational
        break;
    }

    // Type-based adjustments
    switch (type) {
      case MemoryType.EPISODIC:
        importance += scope === 'interaction' ? 0.1 : 0.2;
        break;
      case MemoryType.PROCEDURAL:
        importance += scope === 'company' ? 0.2 : 0.1;
        break;
      case MemoryType.EMOTIONAL:
        importance += 0.15;
        break;
    }

    // Context-based adjustments
    if (context.priority) {
      const priorityBonus = {
        'low': 0,
        'medium': 0.05,
        'high': 0.1,
        'urgent': 0.2
      };
      importance += priorityBonus[context.priority] || 0;
    }

    if (context.resolution_status === 'escalated') importance += 0.15;
    if (context.tags?.includes('important')) importance += 0.1;

    return Math.max(0, Math.min(1, importance));
  }

  private buildScopeFilters(params: EnhancedMemorySearchParams) {
    return {
      include_company: params.inherit_from_parent_scopes !== false,
      include_customer: !!params.customer_id,
      include_interaction: !!params.interaction_id,
      customer_id: params.customer_id,
      interaction_id: params.interaction_id
    };
  }

  private async searchCompanyLevel(params: EnhancedMemorySearchParams, filters: any): Promise<EnhancedMemory[]> {
    if (!filters.include_company) return [];

    const companyParams = {
      ...params,
      // Add filter to exclude customer-specific memories
    };

    const results = await this.searchMemories(companyParams);
    return results.filter(m => !(m.context as EnhancedMemoryContext).customer_id) as EnhancedMemory[];
  }

  private async searchCustomerLevel(params: EnhancedMemorySearchParams, filters: any): Promise<EnhancedMemory[]> {
    if (!filters.include_customer || !filters.customer_id) return [];

    const customerParams = {
      ...params,
      // Add customer filter logic
    };

    const results = await this.searchMemories(customerParams);
    return results.filter(m => {
      const ctx = m.context as EnhancedMemoryContext;
      return ctx.customer_id === filters.customer_id && !ctx.interaction_id;
    }) as EnhancedMemory[];
  }

  private async searchInteractionLevel(params: EnhancedMemorySearchParams, filters: any): Promise<EnhancedMemory[]> {
    if (!filters.include_interaction || !filters.interaction_id) return [];

    const interactionParams = {
      ...params,
      // Add interaction filter logic
    };

    const results = await this.searchMemories(interactionParams);
    return results.filter(m => {
      const ctx = m.context as EnhancedMemoryContext;
      return ctx.customer_id === filters.customer_id && 
             ctx.interaction_id === filters.interaction_id;
    }) as EnhancedMemory[];
  }

  private combineAndRankMemories(
    scopedResults: {
      company: EnhancedMemory[];
      customer: EnhancedMemory[];
      interaction: EnhancedMemory[];
    },
    params: EnhancedMemorySearchParams
  ): EnhancedMemory[] {
    const all = [
      ...scopedResults.interaction,
      ...scopedResults.customer,
      ...scopedResults.company
    ];

    // Sort by relevance (interaction > customer > company) and importance
    return all.sort((a, b) => {
      const scopePriority = { interaction: 3, customer: 2, company: 1 };
      const aPriority = scopePriority[a.scope_level];
      const bPriority = scopePriority[b.scope_level];
      
      if (aPriority !== bPriority) return bPriority - aPriority;
      return b.importance - a.importance;
    });
  }

  private calculateRelevanceScores(memories: EnhancedMemory[], params: EnhancedMemorySearchParams): Record<string, number> {
    const scores: Record<string, number> = {};
    
    for (const memory of memories) {
      let score = memory.importance;
      
      // Boost score based on scope relevance
      if (memory.scope_level === 'interaction') score += 0.3;
      else if (memory.scope_level === 'customer') score += 0.2;
      else score += 0.1;
      
      scores[memory.id] = Math.min(1, score);
    }
    
    return scores;
  }

  private categorizeCustomerMemory(memory: EnhancedMemory): string {
    const content = memory.content.toLowerCase();
    const tags = memory.context.tags || [];
    
    if (tags.includes('preference') || content.includes('prefer') || content.includes('like')) {
      return 'preference';
    }
    if (tags.includes('issue') || content.includes('problem') || content.includes('error')) {
      return 'issue';
    }
    if (tags.includes('communication') || content.includes('email') || content.includes('call')) {
      return 'communication';
    }
    if (tags.includes('satisfaction') || content.includes('happy') || content.includes('satisfied')) {
      return 'satisfaction';
    }
    if (tags.includes('escalation') || content.includes('escalate') || memory.context.priority === 'urgent') {
      return 'escalation';
    }
    
    return 'product'; // default
  }

  private getInheritancePath(customer_id?: string, interaction_id?: string): string[] {
    const path = ['company'];
    if (customer_id) path.push(`customer:${customer_id}`);
    if (interaction_id) path.push(`interaction:${interaction_id}`);
    return path;
  }

  private getAccessRestrictions(scope: MemoryScopeLevel, customer_id?: string, interaction_id?: string): string[] {
    const restrictions: string[] = [];
    
    if (scope === 'customer') {
      restrictions.push(`Limited to customer ${customer_id}`);
    }
    if (scope === 'interaction') {
      restrictions.push(`Limited to interaction ${interaction_id}`);
      restrictions.push(`Limited to customer ${customer_id}`);
    }
    
    return restrictions;
  }
}

export const customerSupportMemoryService = new CustomerSupportMemoryService();