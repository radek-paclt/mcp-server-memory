import OpenAI from 'openai';
import { config } from '../config/environment';
import {
  openaiEmbeddingRequestsCounter,
  openaiEmbeddingDuration,
  openaiEmbeddingDurationMs,
  openaiEmbeddingTokensCounter,
  openaiChatCompletionCounter,
  openaiChatCompletionDurationMs,
} from '../metrics/custom-metrics';

export class OpenAIService {
  private client: OpenAI;

  constructor() {
    this.client = new OpenAI({
      apiKey: config.OPENAI_API_KEY,
    });
  }

  async createEmbedding(text: string): Promise<number[]> {
    const startTime = Date.now();
    openaiEmbeddingRequestsCounter.add(1, { operation: 'single_embedding' });
    
    try {
      const response = await this.client.embeddings.create({
        model: config.OPENAI_MODEL,
        input: text,
      });

      openaiEmbeddingTokensCounter.add(response.usage?.total_tokens || 0);
      return response.data[0].embedding;
    } catch (error) {
      throw error;
    } finally {
      const duration = (Date.now() - startTime) / 1000;
      openaiEmbeddingDurationMs.record(Date.now() - startTime, { operation: 'single_embedding' });
    }
  }

  async createEmbeddings(texts: string[]): Promise<number[][]> {
    const startTime = Date.now();
    openaiEmbeddingRequestsCounter.add(1, { operation: 'batch_embedding' });
    
    try {
      const response = await this.client.embeddings.create({
        model: config.OPENAI_MODEL,
        input: texts,
      });

      openaiEmbeddingTokensCounter.add(response.usage?.total_tokens || 0);
      return response.data.map((item) => item.embedding);
    } catch (error) {
      throw error;
    } finally {
      const duration = (Date.now() - startTime) / 1000;
      openaiEmbeddingDurationMs.record(Date.now() - startTime, { operation: 'batch_embedding' });
    }
  }

  async extractKeywords(text: string): Promise<string[]> {
    const startTime = Date.now();
    openaiChatCompletionCounter.add(1, { operation: 'extract_keywords' });
    
    try {
      const response = await this.client.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: 'Extract 3-5 key words or phrases from the text. Return only comma-separated keywords.',
          },
          {
            role: 'user',
            content: text,
          },
        ],
        temperature: 0.3,
        max_tokens: 50,
      });

      const keywords = response.choices[0].message.content?.split(',').map(k => k.trim()) || [];
      return keywords;
    } catch (error) {
      return [];
    } finally {
      openaiChatCompletionDurationMs.record(Date.now() - startTime, { operation: 'extract_keywords' });
    }
  }

  async generateSummary(text: string): Promise<string> {
    const startTime = Date.now();
    openaiChatCompletionCounter.add(1, { operation: 'generate_summary' });
    
    try {
      const response = await this.client.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: 'Create a concise 1-2 sentence summary capturing the essence of the text. Be specific and informative.',
          },
          {
            role: 'user',
            content: text,
          },
        ],
        temperature: 0.3,
        max_tokens: 60,
      });

      return response.choices[0].message.content?.trim() || text.substring(0, 150) + '...';
    } catch (error) {
      // Fallback to truncation
      return text.length > 150 ? text.substring(0, 147) + '...' : text;
    } finally {
      openaiChatCompletionDurationMs.record(Date.now() - startTime, { operation: 'generate_summary' });
    }
  }

  async analyzeEmotion(text: string): Promise<number> {
    const startTime = Date.now();
    openaiChatCompletionCounter.add(1, { operation: 'analyze_emotion' });
    
    try {
      const response = await this.client.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: 'Analyze the emotional valence of this text. Return only a number between -1 (very negative) and 1 (very positive).',
          },
          {
            role: 'user',
            content: text,
          },
        ],
        temperature: 0.3,
        max_tokens: 10,
      });

      const valence = parseFloat(response.choices[0].message.content || '0');
      return Math.max(-1, Math.min(1, valence));
    } catch (error) {
      return 0;
    } finally {
      openaiChatCompletionDurationMs.record(Date.now() - startTime, { operation: 'analyze_emotion' });
    }
  }

  async summarizeTexts(texts: string[]): Promise<string> {
    const startTime = Date.now();
    openaiChatCompletionCounter.add(1, { operation: 'summarize_texts' });
    
    try {
      const response = await this.client.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant that creates concise summaries of multiple related texts.',
          },
          {
            role: 'user',
            content: `Please create a coherent summary of these ${texts.length} related memories:\n\n${texts.map((t, i) => `Memory ${i + 1}: ${t}`).join('\n\n')}`,
          },
        ],
        max_tokens: 300,
        temperature: 0.7,
      });

      return response.choices[0].message.content || 'Summary generation failed';
    } catch (error) {
      return `Combined memories: ${texts.map(t => t.substring(0, 50)).join('; ')}...`;
    } finally {
      openaiChatCompletionDurationMs.record(Date.now() - startTime, { operation: 'summarize_texts' });
    }
  }
}