import { metrics } from '@opentelemetry/api';

// Get meter - this will be available after telemetry is initialized
function getMeter() {
  return metrics.getMeter('mcp-memory-server', '1.0.0');
}

// Memory operation metrics
export const memoryOperationsCounter = getMeter().createCounter('mcp_memory_operations_total', {
  description: 'Total number of memory operations',
  unit: '1',
});

export const memoryOperationDuration = getMeter().createHistogram('mcp_memory_operation_duration_seconds', {
  description: 'Duration of memory operations in seconds',
  unit: 's',
});

export const memoryErrorsCounter = getMeter().createCounter('mcp_memory_errors_total', {
  description: 'Total number of memory operation errors',
  unit: '1',
});

// Qdrant metrics
export const qdrantConnectionsGauge = getMeter().createUpDownCounter('mcp_qdrant_connections', {
  description: 'Number of active Qdrant connections',
  unit: '1',
});

export const qdrantOperationsCounter = getMeter().createCounter('mcp_qdrant_operations_total', {
  description: 'Total number of Qdrant operations',
  unit: '1',
});

export const qdrantOperationDuration = getMeter().createHistogram('mcp_qdrant_operation_duration_seconds', {
  description: 'Duration of Qdrant operations in seconds',
  unit: 's',
});

// OpenAI metrics
export const openaiEmbeddingRequestsCounter = getMeter().createCounter('mcp_openai_embedding_requests_total', {
  description: 'Total number of OpenAI embedding requests',
  unit: '1',
});

export const openaiEmbeddingDuration = getMeter().createHistogram('mcp_openai_embedding_duration_seconds', {
  description: 'Duration of OpenAI embedding requests in seconds',
  unit: 's',
});

export const openaiEmbeddingTokensCounter = getMeter().createCounter('mcp_openai_embedding_tokens_total', {
  description: 'Total number of tokens processed for embeddings',
  unit: '1',
});

// MCP server metrics
export const mcpRequestsCounter = getMeter().createCounter('mcp_requests_total', {
  description: 'Total number of MCP requests',
  unit: '1',
});

export const mcpRequestDuration = getMeter().createHistogram('mcp_request_duration_seconds', {
  description: 'Duration of MCP requests in seconds',
  unit: 's',
});

export const mcpActiveConnectionsGauge = getMeter().createUpDownCounter('mcp_active_connections', {
  description: 'Number of active MCP connections',
  unit: '1',
});