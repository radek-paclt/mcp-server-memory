# Examples

Real-world examples of using the MCP Memory Server with Claude Code.

## Personal Knowledge Management

### Daily Journaling

```bash
# Morning planning
claude -p "Store this morning plan: Focus on completing the API documentation, prepare for 2pm client call, review Sarah's PR. Feeling energetic and motivated." \
  --mcp-config claude-code-mcp.json

# End of day reflection
claude -p "Store evening reflection: Completed API docs ahead of schedule. Client call went well - they approved the proposal. Sarah's PR needed minor fixes. Overall very productive day!" \
  --mcp-config claude-code-mcp.json

# Weekly review
claude -p "Search my memories from this week and summarize my accomplishments and challenges" \
  --mcp-config claude-code-mcp.json
```

### Learning and Study Notes

```bash
# Store learning
claude -p "Store this semantic memory: React Server Components (RSC) render on the server and send HTML to client. Benefits: smaller bundles, better SEO, direct database access. Drawbacks: no client state, learning curve." \
  --mcp-config claude-code-mcp.json

# Create study connections
claude -p "Store procedural memory: To implement RSC: 1) Mark component as async, 2) Fetch data directly in component, 3) Return JSX, 4) Use 'use client' directive for interactive parts" \
  --mcp-config claude-code-mcp.json

# Review learning
claude -p "What have I learned about React this month? Include all related memories" \
  --mcp-config claude-code-mcp.json
```

## Professional Use Cases

### Meeting Management

```bash
# Pre-meeting prep
claude -p "What do I remember about previous meetings with the Johnson Industries team?" \
  --mcp-config claude-code-mcp.json

# Store meeting notes
claude -p 'Store this episodic memory: "Johnson Industries quarterly review - They are happy with progress but want faster delivery. Budget increased by 20%. New point of contact: Lisa Chen (lisa@johnson.com). Next steps: revise timeline, add 2 developers." Context: {location: "Zoom call", people: ["Tom Johnson", "Lisa Chen", "Mike Ross"], mood: "positive", activity: "client meeting"} Importance: 0.9' \
  --mcp-config claude-code-mcp.json

# Action items
claude -p "Store working memory: TODO from Johnson meeting - 1) Send revised timeline by Friday, 2) Schedule intro call with Lisa, 3) Post job listings for 2 developers" \
  --mcp-config claude-code-mcp.json
```

### Project Documentation

```bash
# Architecture decisions
claude -p "Store semantic memory: Decided to use PostgreSQL over MongoDB for user data due to complex relationships and need for ACID compliance. Trade-off: less flexibility but better data integrity." \
  --mcp-config claude-code-mcp.json

# Problem solutions
claude -p "Store procedural memory: Fixed memory leak in React app by: 1) Clean up useEffect subscriptions, 2) Clear timers in cleanup, 3) Abort fetch requests on unmount, 4) Use weak maps for caches" \
  --mcp-config claude-code-mcp.json

# Search solutions
claude -p "Search for all memories about performance optimization in our React app" \
  --mcp-config claude-code-mcp.json
```

## Personal CRM

### Networking and Relationships

```bash
# Store personal info
claude -p 'Store memory: "Met Jane Rodriguez at ReactConf. She works at Meta on React core team, specifically on Suspense. Interested in our error boundary implementation. Exchanged LinkedIns." Context: {location: "ReactConf 2024", people: ["Jane Rodriguez"], activity: "conference networking"}' \
  --mcp-config claude-code-mcp.json

# Add follow-up
claude -p "Store memory: Sent Jane Rodriguez the error boundary article as promised. She replied with helpful feedback about using Error Boundary composition patterns." \
  --mcp-config claude-code-mcp.json

# Before reconnecting
claude -p "What do I remember about Jane Rodriguez?" \
  --mcp-config claude-code-mcp.json
```

### Client Relationship Management

```bash
# Client preferences
claude -p "Store memory about client: Acme Corp prefers morning meetings (9-11am), decisions made by committee, very detail-oriented. Key stakeholder: David Park values data-driven presentations." \
  --mcp-config claude-code-mcp.json

# Project history
claude -p "Search all memories related to Acme Corp projects and client interactions" \
  --mcp-config claude-code-mcp.json
```

## Technical Knowledge Base

### Code Patterns and Solutions

```bash
# Store solution
claude -p 'Store procedural memory: "Optimized database queries by implementing connection pooling: pgPool config: {max: 20, idleTimeoutMillis: 30000, connectionTimeoutMillis: 2000}. Reduced response time by 40%."' \
  --mcp-config claude-code-mcp.json

# Debugging notes
claude -p "Store memory: Mysterious production bug - users randomly logged out. Root cause: Redis session store memory limit hit. Solution: Increased memory and implemented LRU eviction policy." \
  --mcp-config claude-code-mcp.json

# Search for solutions
claude -p "Search memories for database optimization techniques we've used" \
  --mcp-config claude-code-mcp.json
```

### Configuration Management

```bash
# Store configurations
claude -p "Store semantic memory: Production Kubernetes limits - API pods: 2CPU/4Gi memory, Worker pods: 4CPU/8Gi memory, Database: 8CPU/32Gi memory. Autoscaling: 2-10 pods based on CPU>70%." \
  --mcp-config claude-code-mcp.json

# Environment details
claude -p "What are our production environment configurations?" \
  --mcp-config claude-code-mcp.json
```

## Creative and Personal

### Idea Capture

```bash
# Random ideas
claude -p "Store memory: App idea - 'MoodMap' - tracks emotional states throughout the day using voice analysis and creates beautiful visualizations. Could help with mental health awareness." \
  --mcp-config claude-code-mcp.json

# Creative inspirations
claude -p "Store sensory memory: Saw amazing generative art installation at museum - flowing particles responding to viewer movement, created sense of connection between people. Colors shifted from cool to warm based on proximity." \
  --mcp-config claude-code-mcp.json

# Review ideas
claude -p "Search all my app ideas and creative inspirations from the last 6 months" \
  --mcp-config claude-code-mcp.json
```

### Personal Milestones

```bash
# Life events
claude -p "Store emotional memory: Daughter took her first steps today! She was so proud of herself, huge smile. We all cheered. One of those perfect family moments. Importance: 1.0" \
  --mcp-config claude-code-mcp.json

# Achievements
claude -p "Store episodic memory: Successfully defended my thesis today! Committee was impressed with the novel approach to distributed systems consensus. Dr. Johnson said it was one of the best defenses he's seen. Celebrating tonight!" \
  --mcp-config claude-code-mcp.json

# Memory lane
claude -p "Show me my most important positive memories from this year" \
  --mcp-config claude-code-mcp.json
```

## Automation Examples

### Daily Standup Prep

```bash
#!/bin/bash
# standup-prep.sh
echo "=== Daily Standup Prep ==="
echo "Yesterday's work:"
claude -p "Search my memories from yesterday about work tasks and accomplishments" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions

echo -e "\nToday's plans:"
claude -p "Search my working memories for todos and today's plans" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions
```

### Weekly Report Generator

```bash
#!/bin/bash
# weekly-report.sh
WEEK_START=$(date -d 'last monday' '+%Y-%m-%d')
WEEK_END=$(date '+%Y-%m-%d')

claude -p "Generate a weekly report by analyzing my memories from $WEEK_START to $WEEK_END. Include key accomplishments, meetings, decisions made, and challenges faced." \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions > weekly-report-$(date +%Y-W%V).md
```

### Memory Maintenance

```bash
#!/bin/bash
# memory-maintenance.sh
# Clean up old working memories
claude -p "Delete all working memories older than 7 days" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions

# Archive important memories
claude -p "Search for memories with importance > 0.8 from last month and create a summary" \
  --mcp-config claude-code-mcp.json \
  --dangerously-skip-permissions > important-memories-$(date -d 'last month' +%Y-%m).md
```

## Advanced Patterns

### Memory Chains

```bash
# Create connected memories
claude -p "Store memory: Started new feature branch for user authentication" \
  --mcp-config claude-code-mcp.json

claude -p "Store memory: Implemented JWT token generation for user auth feature. Tokens expire in 24h, refresh tokens in 30 days." \
  --mcp-config claude-code-mcp.json

claude -p "Store memory: Added password reset flow to auth feature. Send email with 6-digit code, expires in 10 minutes." \
  --mcp-config claude-code-mcp.json

# Retrieve the chain
claude -p "Show me all memories related to the user authentication feature in chronological order" \
  --mcp-config claude-code-mcp.json
```

### Contextual Memory Search

```bash
# Multi-criteria search
claude -p 'Search memories with these criteria: type: "episodic", minimum importance: 0.7, positive emotional valence, from last 30 days, include associations' \
  --mcp-config claude-code-mcp.json

# Pattern analysis
claude -p "Analyze my work memories to identify my most productive times and conditions" \
  --mcp-config claude-code-mcp.json
```

These examples demonstrate the versatility of the memory system for personal knowledge management, professional use, and creative applications.