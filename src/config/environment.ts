import dotenv from 'dotenv';
import { z } from 'zod';

// Load environment-specific config file
const envFile = process.env.NODE_ENV === 'development' ? '.env.development' : '.env';
dotenv.config({ path: envFile });

const envSchema = z.object({
  QDRANT_URL: z.string().url(),
  QDRANT_API_KEY: z.string().optional(),
  QDRANT_COLLECTION_NAME: z.string(),
  OPENAI_API_KEY: z.string(),
  OPENAI_MODEL: z.string(),
  MCP_SERVER_NAME: z.string(),
  MCP_SERVER_PORT: z.string().transform(Number),
  SIMILARITY_THRESHOLD: z.string().transform(Number).default('0.7'),
});

export const config = envSchema.parse(process.env);

export const VECTOR_DIMENSION = 1536; // for text-embedding-3-small
export const DEFAULT_SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD || 0.3; // Use configured value or fallback