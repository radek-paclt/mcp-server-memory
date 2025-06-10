# MCP Memory Server Documentation

Welcome to the MCP Memory Server documentation. This system provides human-like memory capabilities for Claude through the Model Context Protocol (MCP).

## Quick Start

1. **[Installation Guide](./installation.md)** - Step-by-step setup instructions
2. **[Development Setup](./development-setup.md)** - Local development environment setup
3. **[Configuration Guide](./configuration.md)** - Detailed configuration options  
4. **[Usage Guide](./usage.md)** - How to use the memory server with Claude Code
5. **[Claude Desktop Integration](./claude-usage.md)** - Using with Claude Desktop

## Core Features

- **[Customer Support Scoping](./customer-support-scoping.md)** - Hierarchical memory isolation for customer support teams
- **[API Reference](./api-reference.md)** - Complete tool reference
- **[Architecture](./architecture.md)** - System design and components
- **[Examples](./examples.md)** - Real-world usage examples

## Configuration & Setup

- **[Claude Configuration](./claude-configuration.md)** - Example CLAUDE.md configuration for optimal memory usage
- **[Troubleshooting](./troubleshooting.md)** - Common issues and solutions

## Advanced Topics

- **[Performance Analysis](./similarity-threshold-analysis.md)** - Similarity threshold optimization
- **[Optimization Analysis](./optimization-analysis.md)** - Ultra-deep performance optimization strategies

## What is MCP?

The Model Context Protocol (MCP) is a standard for extending AI assistants with custom tools and capabilities. This memory server implements MCP to give Claude human-like memory abilities including:

- **Episodic memories** - Personal experiences
- **Semantic memories** - Facts and knowledge  
- **Procedural memories** - How-to information
- **Emotional memories** - Feelings and emotions
- **Sensory memories** - Sensory experiences
- **Working memories** - Short-term information

## Key Features

- 🧠 **Human-like memory types** with automatic categorization
- 🔍 **Vector similarity search** using Qdrant database
- 🎯 **Contextual understanding** with OpenAI embeddings
- 😊 **Emotional analysis** of memory content
- 🔗 **Memory associations** automatically created
- 📊 **Memory analytics** and pattern recognition
- 🚀 **Fast retrieval** with vector search
- 🔐 **Secure storage** with environment-based configuration

## Customer Support Features ✨

The memory system now includes **hierarchical scoping** for customer support teams:

- **Company Level**: Global knowledge accessible to all agents (no customer_id)
- **Customer Level**: Customer-specific information across all interactions (customer_id only)
- **Interaction Level**: Conversation-specific notes and context (customer_id + interaction_id)

This enables automatic data isolation while maintaining contextual relevance and **100% backward compatibility**.

## Quick Start Example

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Build the project
npm run build

# 4. Test with Claude Code
claude -p "Store a memory: Had a great meeting today" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions

# 5. Test customer support scoping
claude -p "Store customer memory with customer_id: cust_123" \
  --mcp-config claude-code-mcp.json
```

## Architecture Overview

The system uses:
- **Qdrant** for vector database storage
- **OpenAI** for embeddings and text analysis
- **TypeScript** for type safety and development experience
- **MCP Protocol** for Claude integration
- **Hierarchical Scoping** for multi-tenant memory isolation