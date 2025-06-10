<<<<<<< HEAD
# MCP Memory Server Documentation

Welcome to the MCP Memory Server documentation. This system provides human-like memory capabilities for Claude through the Model Context Protocol (MCP).

## Documentation Contents

1. **[Installation Guide](./installation.md)** - Step-by-step setup instructions
2. **[Configuration Guide](./configuration.md)** - Detailed configuration options
3. **[Usage Guide](./usage.md)** - How to use the memory server with Claude Code
4. **[API Reference](./api-reference.md)** - Complete tool reference
5. **[Architecture](./architecture.md)** - System design and components
6. **[Examples](./examples.md)** - Real-world usage examples
7. **[Troubleshooting](./troubleshooting.md)** - Common issues and solutions

## Quick Start

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
```

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
=======
# MCP Memory Server Documentation

Welcome to the MCP Memory Server documentation. This system provides human-like memory capabilities for Claude through the Model Context Protocol (MCP).

## Documentation Contents

1. **[Installation Guide](./installation.md)** - Step-by-step setup instructions
2. **[Configuration Guide](./configuration.md)** - Detailed configuration options
3. **[Usage Guide](./usage.md)** - How to use the memory server with Claude Code
4. **[API Reference](./api-reference.md)** - Complete tool reference
5. **[Architecture](./architecture.md)** - System design and components
6. **[Examples](./examples.md)** - Real-world usage examples
7. **[Troubleshooting](./troubleshooting.md)** - Common issues and solutions

## Quick Start

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
```

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
>>>>>>> origin/main
- 🔐 **Secure storage** with environment-based configuration