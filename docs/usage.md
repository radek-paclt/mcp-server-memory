# Usage Guide

This guide explains how to use the MCP Memory Server with Claude Code for storing and retrieving human-like memories.

## Basic Usage

### Starting Claude Code with MCP

```bash
# Basic command
claude -p "Your request here" --mcp-config claude-code-mcp.json

# Skip permissions for automation
claude -p "Your request here" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions
```

## Storing Memories

### Simple Memory Storage

```bash
# Store a basic memory
claude -p "Store this memory: Had a productive meeting with the team about Q4 planning" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions
```

### Storing Different Memory Types

```bash
# Episodic memory (personal experience)
claude -p "Store an episodic memory: Went to Sarah's birthday party at the beach. The sunset was beautiful and everyone had a great time." \
  --mcp-config claude-code-mcp.json

# Semantic memory (fact/knowledge)
claude -p "Store a semantic memory: Python's GIL (Global Interpreter Lock) prevents true multi-threading for CPU-bound tasks" \
  --mcp-config claude-code-mcp.json

# Procedural memory (how-to)
claude -p "Store a procedural memory: To make perfect coffee, use 1:15 coffee to water ratio, water at 200°F, and brew for 4 minutes" \
  --mcp-config claude-code-mcp.json

# Emotional memory
claude -p "Store an emotional memory: Felt incredibly proud when my daughter graduated from university" \
  --mcp-config claude-code-mcp.json
```

### Storing Memories with Context

```bash
# With location and people
claude -p 'Store this memory with context: "Had lunch at Bella Italia with John and Mary. Discussed the new project timeline." Include location: "Bella Italia Restaurant", people: ["John", "Mary"], mood: "optimistic", activity: "business lunch"' \
  --mcp-config claude-code-mcp.json

# With custom importance
claude -p 'Store this important memory: "Signed the contract for the new house today!" Set importance to 0.9' \
  --mcp-config claude-code-mcp.json
```

## Searching Memories

### Basic Search

```bash
# Simple keyword search
claude -p "Search my memories for 'meeting'" \
  --mcp-config claude-code-mcp.json

# Natural language search
claude -p "What do I remember about last week's project discussions?" \
  --mcp-config claude-code-mcp.json
```

### Advanced Search

```bash
# Search by memory type
claude -p "Search for all my procedural memories about cooking" \
  --mcp-config claude-code-mcp.json

# Search by importance
claude -p "Show me my most important memories (importance > 0.7)" \
  --mcp-config claude-code-mcp.json

# Search by emotional valence
claude -p "Find my happy memories (positive emotional valence)" \
  --mcp-config claude-code-mcp.json

# Search with associations
claude -p "Search for memories about 'vacation' and include related memories" \
  --mcp-config claude-code-mcp.json
```

### Time-based Search

```bash
# Recent memories
claude -p "Show me memories from the last 7 days" \
  --mcp-config claude-code-mcp.json

# Date range search
claude -p "Find memories from January 2024 about work projects" \
  --mcp-config claude-code-mcp.json
```

## Managing Memories

### Retrieving Specific Memories

```bash
# Get memory by ID
claude -p "Get the memory with ID: 12345-abcde-67890" \
  --mcp-config claude-code-mcp.json
```

### Updating Memories

```bash
# Update memory content
claude -p "Update memory 12345-abcde: Add that the meeting was rescheduled to next Tuesday" \
  --mcp-config claude-code-mcp.json

# Update importance
claude -p "Increase the importance of memory 12345-abcde to 0.9" \
  --mcp-config claude-code-mcp.json

# Add context
claude -p "Add location 'Conference Room B' to memory 12345-abcde" \
  --mcp-config claude-code-mcp.json
```

### Deleting Memories

```bash
# Delete specific memory
claude -p "Delete the memory with ID: 12345-abcde-67890" \
  --mcp-config claude-code-mcp.json

# Delete old memories
claude -p "Delete all working memories older than 7 days" \
  --mcp-config claude-code-mcp.json
```

## Analyzing Memories

### Basic Analysis

```bash
# Get memory statistics
claude -p "Analyze all my memories and show statistics" \
  --mcp-config claude-code-mcp.json

# Analyze by type
claude -p "Analyze my episodic memories from this month" \
  --mcp-config claude-code-mcp.json
```

### Pattern Recognition

```bash
# Find memory patterns
claude -p "What patterns do you see in my work-related memories?" \
  --mcp-config claude-code-mcp.json

# Emotional patterns
claude -p "Analyze the emotional patterns in my memories over the last month" \
  --mcp-config claude-code-mcp.json
```

## Practical Examples

### Daily Journal

```bash
# Morning reflection
claude -p "Store this morning reflection: Woke up feeling energized. Goals for today: finish the report, call mom, gym at 6pm" \
  --mcp-config claude-code-mcp.json

# Evening review
claude -p "Store evening review: Completed the report successfully, had a nice chat with mom, missed gym due to rain. Overall productive day." \
  --mcp-config claude-code-mcp.json
```

### Learning Tracker

```bash
# Store learning
claude -p "Store what I learned: Redux Toolkit simplifies Redux boilerplate significantly. Key concepts: slices, createAsyncThunk, RTK Query" \
  --mcp-config claude-code-mcp.json

# Review learning
claude -p "What have I learned about React and Redux in the past month?" \
  --mcp-config claude-code-mcp.json
```

### Meeting Notes

```bash
# Store meeting notes
claude -p 'Store meeting memory: "Q4 Planning Meeting - Decided to focus on mobile app, postpone desktop version. Sarah leads mobile, John handles backend. Timeline: 3 months." Tag as important work memory' \
  --mcp-config claude-code-mcp.json

# Retrieve meeting history
claude -p "Show me all memories from planning meetings this quarter" \
  --mcp-config claude-code-mcp.json
```

### Personal CRM

```bash
# Store personal information
claude -p 'Store memory: "John mentioned his daughter Emma is starting college next month at Stanford studying Computer Science"' \
  --mcp-config claude-code-mcp.json

# Retrieve before meeting
claude -p "What do I remember about John's family?" \
  --mcp-config claude-code-mcp.json
```

## Tips and Best Practices

1. **Be Descriptive**: More detailed memories are easier to search later
2. **Use Consistent Naming**: Use consistent names for people and places
3. **Tag Important Memories**: Set higher importance for crucial information
4. **Regular Reviews**: Periodically search and review your memories
5. **Context Matters**: Always include context when relevant
6. **Emotional Tagging**: The system automatically analyzes emotion, but you can be explicit

## Automation Scripts

### Daily Memory Backup

```bash
#!/bin/bash
# backup-memories.sh
claude -p "Search all memories from today and export as JSON" \
  --mcp-config claude-code-mcp.json \
  --output-format json > "memories-$(date +%Y%m%d).json"
```

### Weekly Summary

```bash
#!/bin/bash
# weekly-summary.sh
claude -p "Analyze my memories from the past week and create a summary" \
  --mcp-config claude-code-mcp.json > "weekly-summary-$(date +%Y%W).txt"
```

## Troubleshooting Common Issues

### Permission Errors

If you get permission prompts, either:
1. Accept them interactively
2. Use `--dangerously-skip-permissions` for scripts
3. Use `--allowedTools` with specific tool names

### No Results Found

- Check your search terms
- Try broader queries
- Verify memories exist with "analyze all memories"

### Slow Responses

- Reduce search limit
- Use more specific queries
- Check Qdrant performance