/**
 * Automatic Memory Type Classification System
 * Intelligently determines memory type based on content analysis
 */

import { MemoryType } from '../types/memory';
import { OpenAIService } from './openai';

interface TypeClassificationResult {
  type: MemoryType;
  confidence: number;
  reasoning: string;
  alternatives?: Array<{ type: MemoryType; confidence: number }>;
}

interface ContentAnalysis {
  hasPersonalPronouns: boolean;
  hasTimeReferences: boolean;
  hasEmotionalWords: boolean;
  hasInstructions: boolean;
  hasSensoryWords: boolean;
  hasFactualStatements: boolean;
  hasQuestions: boolean;
  temporalContext: 'past' | 'present' | 'future' | 'timeless';
  emotionalIntensity: number;
  instructionalWords: string[];
  personalMarkers: string[];
}

export class MemoryTypeClassifier {
  private openai: OpenAIService;
  
  // Pattern recognition for different memory types
  private readonly patterns = {
    episodic: {
      personalPronouns: ['I', 'me', 'my', 'we', 'us', 'our'],
      timeIndicators: ['yesterday', 'today', 'last week', 'when I', 'after I', 'before I', 'during', 'while I'],
      experienceWords: ['happened', 'experienced', 'felt', 'saw', 'heard', 'went', 'visited', 'met', 'did'],
      contexts: ['meeting', 'conversation', 'event', 'trip', 'experience', 'incident']
    },
    
    semantic: {
      factualIndicators: ['is', 'are', 'was', 'were', 'fact', 'research shows', 'studies indicate', 'definition'],
      knowledgeWords: ['concept', 'theory', 'principle', 'rule', 'law', 'formula', 'algorithm'],
      generalStatements: ['always', 'never', 'typically', 'generally', 'usually', 'often'],
      domains: ['science', 'mathematics', 'history', 'literature', 'technology', 'business']
    },
    
    procedural: {
      instructionalWords: ['how to', 'step', 'first', 'then', 'next', 'finally', 'process', 'method'],
      actionVerbs: ['create', 'build', 'make', 'configure', 'setup', 'install', 'run', 'execute'],
      sequenceWords: ['before', 'after', 'during', 'while', 'until', 'when'],
      imperativeVerbs: ['do', 'use', 'click', 'type', 'select', 'choose', 'enter', 'save']
    },
    
    emotional: {
      emotionWords: ['happy', 'sad', 'angry', 'frustrated', 'excited', 'worried', 'anxious', 'proud'],
      intensifiers: ['very', 'extremely', 'incredibly', 'really', 'so', 'quite', 'rather'],
      feelingExpressions: ['I feel', 'feeling', 'emotion', 'mood', 'heartbroken', 'overjoyed'],
      relationships: ['love', 'hate', 'friendship', 'family', 'relationship', 'conflict']
    },
    
    sensory: {
      visualWords: ['see', 'look', 'watch', 'color', 'bright', 'dark', 'beautiful', 'ugly'],
      auditoryWords: ['hear', 'listen', 'sound', 'noise', 'music', 'voice', 'loud', 'quiet'],
      tactileWords: ['feel', 'touch', 'soft', 'hard', 'warm', 'cold', 'smooth', 'rough'],
      olfactoryWords: ['smell', 'scent', 'fragrance', 'odor', 'aroma'],
      gustoryWords: ['taste', 'flavor', 'sweet', 'sour', 'bitter', 'salty', 'delicious']
    },
    
    working: {
      temporaryIndicators: ['currently', 'right now', 'at the moment', 'temporarily', 'for now'],
      taskWords: ['task', 'todo', 'reminder', 'note', 'quick', 'temporary'],
      urgencyWords: ['urgent', 'asap', 'immediately', 'soon', 'deadline', 'due'],
      shortTermWords: ['brief', 'quick', 'short', 'momentary', 'fleeting']
    }
  };

  constructor() {
    this.openai = new OpenAIService();
  }

  /**
   * Main classification method - determines memory type automatically
   */
  async classifyMemoryType(content: string, context?: any): Promise<TypeClassificationResult> {
    // Step 1: Fast pattern-based classification (no API calls)
    const quickClassification = this.quickClassify(content);
    
    // Step 2: If confidence is high enough, return quickly
    if (quickClassification.confidence > 0.8) {
      return quickClassification;
    }
    
    // Step 3: Enhanced classification with AI analysis
    const enhancedClassification = await this.enhancedClassify(content, context);
    
    // Step 4: Combine results for final decision
    return this.combineClassifications(quickClassification, enhancedClassification);
  }

  /**
   * Fast pattern-based classification using rule-based analysis
   */
  private quickClassify(content: string): TypeClassificationResult {
    const analysis = this.analyzeContent(content);
    const scores = this.calculateTypeScores(analysis, content);
    
    // Find the type with highest score
    const sortedTypes = Object.entries(scores).sort(([,a], [,b]) => b - a);
    const [bestType, bestScore] = sortedTypes[0];
    
    return {
      type: bestType as MemoryType,
      confidence: Math.min(bestScore / 10, 1.0), // Normalize to 0-1
      reasoning: this.generateReasoning(bestType as MemoryType, analysis),
      alternatives: sortedTypes.slice(1, 3).map(([type, score]) => ({
        type: type as MemoryType,
        confidence: Math.min(score / 10, 1.0)
      }))
    };
  }

  /**
   * Enhanced classification using AI analysis for edge cases
   */
  private async enhancedClassify(content: string, context?: any): Promise<TypeClassificationResult> {
    try {
      const prompt = `
Analyze this text and classify it into one of these memory types:

1. EPISODIC - Personal experiences, events that happened to someone
2. SEMANTIC - Facts, knowledge, general information  
3. PROCEDURAL - Instructions, how-to information, processes
4. EMOTIONAL - Emotionally charged content, feelings, relationships
5. SENSORY - Descriptions involving senses (sight, sound, touch, taste, smell)
6. WORKING - Temporary information, reminders, short-term notes

Text to analyze: "${content}"

Context: ${context ? JSON.stringify(context) : 'None provided'}

Respond with ONLY a JSON object in this format:
{
  "type": "episodic|semantic|procedural|emotional|sensory|working",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
`;

      const response = await this.openai.client.chat.completions.create({
        model: 'gpt-4o-mini', // Faster and cheaper for classification
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1, // Low temperature for consistent classification
        max_tokens: 200
      });

      const result = JSON.parse(response.choices[0].message.content || '{}');
      
      return {
        type: result.type as MemoryType,
        confidence: result.confidence || 0.5,
        reasoning: result.reasoning || 'AI classification',
        alternatives: []
      };

    } catch (error) {
      console.error('Enhanced classification failed:', error);
      
      // Fallback to semantic type
      return {
        type: MemoryType.SEMANTIC,
        confidence: 0.3,
        reasoning: 'Fallback classification due to AI analysis failure'
      };
    }
  }

  /**
   * Analyze content for various patterns and characteristics
   */
  private analyzeContent(content: string): ContentAnalysis {
    const lowerContent = content.toLowerCase();
    const words = lowerContent.split(/\s+/);

    return {
      hasPersonalPronouns: this.containsAny(words, this.patterns.episodic.personalPronouns.map(p => p.toLowerCase())),
      hasTimeReferences: this.containsAny(lowerContent, this.patterns.episodic.timeIndicators),
      hasEmotionalWords: this.containsAny(words, this.patterns.emotional.emotionWords),
      hasInstructions: this.containsAny(lowerContent, this.patterns.procedural.instructionalWords),
      hasSensoryWords: this.containsAnySensory(words),
      hasFactualStatements: this.containsAny(lowerContent, this.patterns.semantic.factualIndicators),
      hasQuestions: content.includes('?'),
      temporalContext: this.determineTemporalContext(lowerContent),
      emotionalIntensity: this.calculateEmotionalIntensity(words),
      instructionalWords: this.findMatches(lowerContent, this.patterns.procedural.instructionalWords),
      personalMarkers: this.findMatches(lowerContent, this.patterns.episodic.personalPronouns.map(p => p.toLowerCase()))
    };
  }

  /**
   * Calculate confidence scores for each memory type
   */
  private calculateTypeScores(analysis: ContentAnalysis, content: string): Record<string, number> {
    const scores = {
      [MemoryType.EPISODIC]: 0,
      [MemoryType.SEMANTIC]: 0,
      [MemoryType.PROCEDURAL]: 0,
      [MemoryType.EMOTIONAL]: 0,
      [MemoryType.SENSORY]: 0,
      [MemoryType.WORKING]: 0
    };

    // Episodic scoring
    if (analysis.hasPersonalPronouns) scores[MemoryType.EPISODIC] += 3;
    if (analysis.hasTimeReferences) scores[MemoryType.EPISODIC] += 2;
    if (analysis.temporalContext === 'past') scores[MemoryType.EPISODIC] += 2;
    if (analysis.personalMarkers.length > 0) scores[MemoryType.EPISODIC] += analysis.personalMarkers.length;

    // Semantic scoring
    if (analysis.hasFactualStatements) scores[MemoryType.SEMANTIC] += 3;
    if (analysis.temporalContext === 'timeless') scores[MemoryType.SEMANTIC] += 2;
    if (!analysis.hasPersonalPronouns && !analysis.hasTimeReferences) scores[MemoryType.SEMANTIC] += 1;
    if (this.containsAny(content.toLowerCase(), this.patterns.semantic.knowledgeWords)) scores[MemoryType.SEMANTIC] += 2;

    // Procedural scoring
    if (analysis.hasInstructions) scores[MemoryType.PROCEDURAL] += 4;
    if (analysis.instructionalWords.length > 0) scores[MemoryType.PROCEDURAL] += analysis.instructionalWords.length * 2;
    if (this.containsAny(content.toLowerCase(), this.patterns.procedural.actionVerbs)) scores[MemoryType.PROCEDURAL] += 2;

    // Emotional scoring
    if (analysis.hasEmotionalWords) scores[MemoryType.EMOTIONAL] += 3;
    if (analysis.emotionalIntensity > 0.7) scores[MemoryType.EMOTIONAL] += 3;
    if (this.containsAny(content.toLowerCase(), this.patterns.emotional.feelingExpressions)) scores[MemoryType.EMOTIONAL] += 2;

    // Sensory scoring
    if (analysis.hasSensoryWords) scores[MemoryType.SENSORY] += 3;
    const sensoryCount = this.countSensoryWords(content.toLowerCase());
    scores[MemoryType.SENSORY] += sensoryCount * 0.5;

    // Working memory scoring  
    if (content.length < 100) scores[MemoryType.WORKING] += 1; // Short content
    if (this.containsAny(content.toLowerCase(), this.patterns.working.temporaryIndicators)) scores[MemoryType.WORKING] += 3;
    if (this.containsAny(content.toLowerCase(), this.patterns.working.taskWords)) scores[MemoryType.WORKING] += 2;
    if (this.containsAny(content.toLowerCase(), this.patterns.working.urgencyWords)) scores[MemoryType.WORKING] += 2;

    return scores;
  }

  /**
   * Combine quick and enhanced classifications
   */
  private combineClassifications(
    quick: TypeClassificationResult, 
    enhanced: TypeClassificationResult
  ): TypeClassificationResult {
    // If both agree and confidence is high, use that
    if (quick.type === enhanced.type && enhanced.confidence > 0.7) {
      return {
        type: quick.type,
        confidence: Math.max(quick.confidence, enhanced.confidence),
        reasoning: `Both analyses agree: ${enhanced.reasoning}`,
        alternatives: quick.alternatives
      };
    }

    // If enhanced has much higher confidence, use that
    if (enhanced.confidence > quick.confidence + 0.3) {
      return enhanced;
    }

    // Otherwise, use quick classification (faster, no API cost)
    return quick;
  }

  /**
   * Utility methods
   */
  private containsAny(text: string, patterns: string[]): boolean {
    return patterns.some(pattern => text.includes(pattern.toLowerCase()));
  }

  private containsAnySensory(words: string[]): boolean {
    const allSensoryWords = [
      ...this.patterns.sensory.visualWords,
      ...this.patterns.sensory.auditoryWords,
      ...this.patterns.sensory.tactileWords,
      ...this.patterns.sensory.olfactoryWords,
      ...this.patterns.sensory.gustoryWords
    ];
    
    return words.some(word => allSensoryWords.includes(word));
  }

  private countSensoryWords(text: string): number {
    const allSensoryWords = [
      ...this.patterns.sensory.visualWords,
      ...this.patterns.sensory.auditoryWords,
      ...this.patterns.sensory.tactileWords,
      ...this.patterns.sensory.olfactoryWords,
      ...this.patterns.sensory.gustoryWords
    ];
    
    return allSensoryWords.filter(word => text.includes(word)).length;
  }

  private determineTemporalContext(content: string): 'past' | 'present' | 'future' | 'timeless' {
    const pastIndicators = ['was', 'were', 'had', 'did', 'went', 'saw', 'yesterday', 'last', 'ago', 'before', 'earlier'];
    const presentIndicators = ['am', 'is', 'are', 'do', 'does', 'now', 'today', 'currently', 'at the moment'];
    const futureIndicators = ['will', 'shall', 'going to', 'plan to', 'tomorrow', 'next', 'future', 'later'];

    const pastCount = pastIndicators.filter(word => content.includes(word)).length;
    const presentCount = presentIndicators.filter(word => content.includes(word)).length;
    const futureCount = futureIndicators.filter(word => content.includes(word)).length;

    if (pastCount > presentCount && pastCount > futureCount) return 'past';
    if (futureCount > presentCount && futureCount > pastCount) return 'future';
    if (presentCount > 0) return 'present';
    return 'timeless';
  }

  private calculateEmotionalIntensity(words: string[]): number {
    const emotionalWords = this.patterns.emotional.emotionWords;
    const intensifiers = this.patterns.emotional.intensifiers;
    
    let intensity = 0;
    let emotionCount = 0;

    words.forEach(word => {
      if (emotionalWords.includes(word)) {
        emotionCount++;
        intensity += 0.5;
      }
      if (intensifiers.includes(word)) {
        intensity += 0.3;
      }
    });

    return Math.min(intensity / words.length * 100, 1.0); // Normalize to 0-1
  }

  private findMatches(content: string, patterns: string[]): string[] {
    return patterns.filter(pattern => content.includes(pattern.toLowerCase()));
  }

  private generateReasoning(type: MemoryType, analysis: ContentAnalysis): string {
    switch (type) {
      case MemoryType.EPISODIC:
        return `Personal experience indicators: ${analysis.hasPersonalPronouns ? 'personal pronouns, ' : ''}${analysis.hasTimeReferences ? 'time references, ' : ''}${analysis.temporalContext} context`;
      
      case MemoryType.SEMANTIC:
        return `Factual content indicators: ${analysis.hasFactualStatements ? 'factual statements, ' : ''}${analysis.temporalContext} context, ${analysis.hasPersonalPronouns ? '' : 'no '}personal markers`;
      
      case MemoryType.PROCEDURAL:
        return `Instructional content: ${analysis.instructionalWords.length} instruction words, ${analysis.hasInstructions ? 'how-to patterns' : 'process descriptions'}`;
      
      case MemoryType.EMOTIONAL:
        return `Emotional content: ${analysis.hasEmotionalWords ? 'emotion words, ' : ''}intensity ${(analysis.emotionalIntensity * 100).toFixed(0)}%`;
      
      case MemoryType.SENSORY:
        return `Sensory descriptions: ${analysis.hasSensoryWords ? 'sensory words detected' : 'sensory patterns'}`;
      
      case MemoryType.WORKING:
        return `Temporary/task content: ${analysis.hasTimeReferences ? 'immediate timeframe, ' : ''}short-term indicators`;
      
      default:
        return 'Pattern-based classification';
    }
  }

  /**
   * Public method for testing/debugging classifications
   */
  async debugClassification(content: string): Promise<{
    quick: TypeClassificationResult;
    enhanced: TypeClassificationResult;
    final: TypeClassificationResult;
    analysis: ContentAnalysis;
  }> {
    const analysis = this.analyzeContent(content);
    const quick = this.quickClassify(content);
    const enhanced = await this.enhancedClassify(content);
    const final = this.combineClassifications(quick, enhanced);

    return { quick, enhanced, final, analysis };
  }
}

// Example usage and testing
export const memoryTypeClassifier = new MemoryTypeClassifier();

// Test cases for validation
export const testClassifications = async () => {
  const testCases = [
    {
      content: "I went to the store yesterday and bought groceries. The cashier was very friendly.",
      expected: MemoryType.EPISODIC
    },
    {
      content: "The capital of France is Paris. It is located in Western Europe and has a population of over 2 million.",
      expected: MemoryType.SEMANTIC
    },
    {
      content: "To install Node.js: 1. Download from nodejs.org 2. Run the installer 3. Verify with node --version",
      expected: MemoryType.PROCEDURAL
    },
    {
      content: "I'm feeling so frustrated with this project. It's making me really anxious and stressed.",
      expected: MemoryType.EMOTIONAL
    },
    {
      content: "The roses smelled incredibly sweet. I could hear birds chirping and feel the warm sunshine on my face.",
      expected: MemoryType.SENSORY
    },
    {
      content: "Remember to call John at 3pm today. Also need to pick up milk.",
      expected: MemoryType.WORKING
    }
  ];

  console.log('Testing memory type classification...');
  
  for (const testCase of testCases) {
    const result = await memoryTypeClassifier.classifyMemoryType(testCase.content);
    const isCorrect = result.type === testCase.expected;
    
    console.log(`\nContent: "${testCase.content}"`);
    console.log(`Expected: ${testCase.expected}, Got: ${result.type} (${(result.confidence * 100).toFixed(1)}%)`);
    console.log(`Correct: ${isCorrect ? '✅' : '❌'}`);
    console.log(`Reasoning: ${result.reasoning}`);
  }
};