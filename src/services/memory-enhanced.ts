/**
 * Enhanced Memory Service with Automatic Type Detection
 * Integrates the MemoryTypeClassifier for intelligent memory type inference
 */

import { Memory, MemoryType } from '../types/memory';
import { MemoryService } from './memory';
import { MemoryTypeClassifier } from './memory-type-classifier';
import { OpenAIService } from './openai';
import { QdrantService } from './qdrant';
import { ChunkingService } from './chunking';

interface CreateMemoryParams {
  content: string;
  summary?: string;
  type?: MemoryType;
  context?: any;
  importance?: number;
}

interface CreateMemoryResult {
  memory: Memory;
  typeDetection: {
    detectedType: MemoryType;
    confidence: number;
    reasoning: string;
    wasOverridden: boolean;
  };
}

export class EnhancedMemoryService extends MemoryService {
  private typeClassifier: MemoryTypeClassifier;

  constructor() {
    super();
    this.typeClassifier = new MemoryTypeClassifier();
  }

  /**
   * Enhanced memory creation with automatic type detection
   */
  async createMemoryEnhanced(params: CreateMemoryParams): Promise<CreateMemoryResult> {
    const startTime = Date.now();
    
    // Step 1: Determine memory type
    let finalType: MemoryType;
    let typeDetection: any;
    let wasOverridden = false;

    if (params.type) {
      // User specified type - use it but still analyze for comparison
      finalType = params.type;
      wasOverridden = true;
      
      // Run type detection in background for learning/validation
      this.typeClassifier.classifyMemoryType(params.content, params.context)
        .then(result => {
          if (result.type !== params.type) {
            console.log(`Type override: User specified ${params.type}, but analysis suggests ${result.type} (confidence: ${(result.confidence * 100).toFixed(1)}%)`);
          }
        })
        .catch(console.error);

      typeDetection = {
        detectedType: finalType,
        confidence: 1.0,
        reasoning: 'User-specified type',
        wasOverridden: true
      };
    } else {
      // Auto-detect type
      const classification = await this.typeClassifier.classifyMemoryType(params.content, params.context);
      finalType = classification.type;
      
      typeDetection = {
        detectedType: classification.type,
        confidence: classification.confidence,
        reasoning: classification.reasoning,
        wasOverridden: false
      };

      console.log(`Auto-detected memory type: ${finalType} (confidence: ${(classification.confidence * 100).toFixed(1)}%) - ${classification.reasoning}`);
    }

    // Step 2: Auto-calculate importance if not provided
    let finalImportance = params.importance;
    if (finalImportance === undefined) {
      finalImportance = this.calculateImportanceEnhanced(params.content, finalType, params.context);
    }

    // Step 3: Create memory using parent class method
    const memory = await this.createMemory(
      params.content,
      finalType,
      params.context,
      finalImportance,
      params.summary
    );

    const totalTime = Date.now() - startTime;
    console.log(`Enhanced memory creation completed in ${totalTime}ms`);

    return {
      memory,
      typeDetection
    };
  }

  /**
   * Enhanced chunked memory creation with type detection
   */
  async createMemoryWithChunkingEnhanced(
    content: string,
    summary?: string,
    type?: MemoryType,
    context?: any,
    importance?: number,
    chunkingMethod: 'semantic' | 'sentence' | 'paragraph' | 'fixed' = 'semantic',
    maxChunkSize: number = 1000
  ): Promise<{
    parentMemory: Memory;
    chunks: Memory[];
    typeDetection: {
      detectedType: MemoryType;
      confidence: number;
      reasoning: string;
      wasOverridden: boolean;
    };
  }> {
    const startTime = Date.now();

    // Step 1: Determine memory type for parent memory
    let finalType: MemoryType;
    let typeDetection: any;

    if (type) {
      finalType = type;
      typeDetection = {
        detectedType: finalType,
        confidence: 1.0,
        reasoning: 'User-specified type',
        wasOverridden: true
      };
    } else {
      // Auto-detect type based on full content
      const classification = await this.typeClassifier.classifyMemoryType(content, context);
      finalType = classification.type;
      
      typeDetection = {
        detectedType: classification.type,
        confidence: classification.confidence,
        reasoning: classification.reasoning,
        wasOverridden: false
      };

      console.log(`Auto-detected memory type for chunked content: ${finalType} (confidence: ${(classification.confidence * 100).toFixed(1)}%)`);
    }

    // Step 2: Auto-calculate importance if not provided
    let finalImportance = importance;
    if (finalImportance === undefined) {
      finalImportance = this.calculateImportanceEnhanced(content, finalType, context);
    }

    // Step 3: Create chunked memory using parent class method
    const result = await this.createMemoryWithChunking(
      content,
      finalType,
      context,
      finalImportance,
      summary,
      chunkingMethod,
      maxChunkSize
    );

    const totalTime = Date.now() - startTime;
    console.log(`Enhanced chunked memory creation completed in ${totalTime}ms`);

    return {
      parentMemory: result.parentMemory,
      chunks: result.memories.slice(1), // First memory is parent
      typeDetection
    };
  }

  /**
   * Enhanced memory update with type re-detection
   */
  async updateMemoryEnhanced(
    id: string,
    updates: {
      content?: string;
      summary?: string;
      type?: MemoryType;
      context?: any;
      importance?: number;
    }
  ): Promise<{
    memory: Memory;
    typeChanged: boolean;
    typeDetection?: {
      detectedType: MemoryType;
      confidence: number;
      reasoning: string;
      wasOverridden: boolean;
    };
  }> {
    const existingMemory = await this.getMemory(id);
    if (!existingMemory) {
      throw new Error(`Memory with ID ${id} not found`);
    }

    let typeDetection: any = undefined;
    let typeChanged = false;
    let finalType = existingMemory.type;

    // If content is being updated, re-analyze type
    if (updates.content && updates.content !== existingMemory.content) {
      if (updates.type) {
        // User specified new type
        finalType = updates.type;
        typeChanged = finalType !== existingMemory.type;
        
        typeDetection = {
          detectedType: finalType,
          confidence: 1.0,
          reasoning: 'User-specified type during update',
          wasOverridden: true
        };
      } else {
        // Auto-detect new type
        const classification = await this.typeClassifier.classifyMemoryType(
          updates.content, 
          updates.context || existingMemory.context
        );
        
        finalType = classification.type;
        typeChanged = finalType !== existingMemory.type;
        
        typeDetection = {
          detectedType: classification.type,
          confidence: classification.confidence,
          reasoning: classification.reasoning,
          wasOverridden: false
        };

        if (typeChanged) {
          console.log(`Type changed during update: ${existingMemory.type} → ${finalType} (confidence: ${(classification.confidence * 100).toFixed(1)}%)`);
        }
      }
    } else if (updates.type && updates.type !== existingMemory.type) {
      // Type specified but content not changed
      finalType = updates.type;
      typeChanged = true;
      
      typeDetection = {
        detectedType: finalType,
        confidence: 1.0,
        reasoning: 'User-specified type change without content change',
        wasOverridden: true
      };
    }

    // Recalculate importance if content or type changed
    let finalImportance = updates.importance;
    if (finalImportance === undefined && (updates.content || typeChanged)) {
      finalImportance = this.calculateImportanceEnhanced(
        updates.content || existingMemory.content,
        finalType,
        updates.context || existingMemory.context
      );
    }

    // Update memory using parent class method
    const updatedMemory = await this.updateMemory(id, {
      ...updates,
      type: finalType,
      importance: finalImportance
    });

    return {
      memory: updatedMemory,
      typeChanged,
      typeDetection
    };
  }

  /**
   * Enhanced importance calculation considering type and context
   */
  private calculateImportanceEnhanced(content: string, type: MemoryType, context?: any): number {
    let importance = 0.5; // Base importance

    // Type-based importance adjustments
    switch (type) {
      case MemoryType.EPISODIC:
        importance += 0.2; // Personal experiences are important
        break;
      case MemoryType.EMOTIONAL:
        importance += 0.15; // Emotional memories are significant
        break;
      case MemoryType.PROCEDURAL:
        importance += 0.1; // How-to information is valuable
        break;
      case MemoryType.WORKING:
        importance -= 0.1; // Working memory is temporary
        break;
      case MemoryType.SEMANTIC:
      case MemoryType.SENSORY:
        // No adjustment - base importance
        break;
    }

    // Content-based adjustments
    if (content.length > 500) importance += 0.1; // Longer content might be more detailed
    if (content.length > 1000) importance += 0.1; // Very long content is often important
    
    // Emotional intensity (if we have emotional analysis)
    const emotionalWords = ['important', 'critical', 'urgent', 'significant', 'crucial', 'vital', 'essential'];
    const lowerContent = content.toLowerCase();
    const emotionalCount = emotionalWords.filter(word => lowerContent.includes(word)).length;
    importance += emotionalCount * 0.05; // Boost for emotional markers

    // Context-based adjustments
    if (context) {
      if (context.tags && context.tags.length > 0) importance += 0.05; // Tagged content is more organized
      if (context.people && context.people.length > 0) importance += 0.05; // Social memories are important
      if (context.source === 'meeting' || context.source === 'conversation') importance += 0.1;
    }

    // Ensure importance stays within bounds
    return Math.max(0, Math.min(1, importance));
  }

  /**
   * Debug method to test type classification without storing
   */
  async debugTypeClassification(content: string, context?: any): Promise<{
    quick: any;
    enhanced: any;
    final: any;
    analysis: any;
  }> {
    return this.typeClassifier.debugClassification(content);
  }

  /**
   * Batch type re-detection for existing memories
   */
  async batchRetypeMemories(filter?: { type?: MemoryType; limit?: number }): Promise<{
    analyzed: number;
    changed: number;
    results: Array<{
      id: string;
      oldType: MemoryType;
      newType: MemoryType;
      confidence: number;
      reasoning: string;
    }>;
  }> {
    console.log('Starting batch re-typing of memories...');
    
    // Get memories to analyze
    const memories = await this.searchMemories({
      type: filter?.type,
      limit: filter?.limit || 100,
      format: 'full'
    });

    const results: any[] = [];
    let changed = 0;

    for (const memory of memories) {
      try {
        const classification = await this.typeClassifier.classifyMemoryType(
          memory.content,
          memory.context
        );

        if (classification.type !== memory.type && classification.confidence > 0.7) {
          // High confidence that type should change
          results.push({
            id: memory.id,
            oldType: memory.type,
            newType: classification.type,
            confidence: classification.confidence,
            reasoning: classification.reasoning
          });

          // Optionally auto-update (uncomment if desired)
          // await this.updateMemory(memory.id, { type: classification.type });
          changed++;
        }

      } catch (error) {
        console.error(`Failed to re-type memory ${memory.id}:`, error);
      }
    }

    console.log(`Batch re-typing complete: ${results.length}/${memories.length} memories analyzed, ${changed} suggested changes`);

    return {
      analyzed: memories.length,
      changed,
      results
    };
  }

  /**
   * Get type distribution statistics
   */
  async getTypeStatistics(): Promise<{
    total: number;
    byType: Record<MemoryType, number>;
    averageConfidence?: number;
  }> {
    const memories = await this.searchMemories({ limit: 1000, format: 'compact' });
    
    const stats = {
      total: memories.length,
      byType: {
        [MemoryType.EPISODIC]: 0,
        [MemoryType.SEMANTIC]: 0,
        [MemoryType.PROCEDURAL]: 0,
        [MemoryType.EMOTIONAL]: 0,
        [MemoryType.SENSORY]: 0,
        [MemoryType.WORKING]: 0
      }
    };

    memories.forEach(memory => {
      stats.byType[memory.type]++;
    });

    return stats;
  }
}

export const enhancedMemoryService = new EnhancedMemoryService();