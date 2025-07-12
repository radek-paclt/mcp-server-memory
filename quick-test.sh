#!/bin/bash

echo "🧠 Memory System Quick Test"
echo "=========================="

# Initialize session
echo "📡 Initializing MCP session..."
SESSION_RESPONSE=$(curl -s http://localhost:3000/sse -H "Accept: text/event-stream" | head -2 | tail -1)
SESSION_ID=$(echo $SESSION_RESPONSE | grep -o 'sessionId=[^"]*' | cut -d'=' -f2)

if [ -z "$SESSION_ID" ]; then
    echo "❌ Failed to get session ID"
    exit 1
fi

echo "✅ Session ID: $SESSION_ID"

# Function to search memory
search_memory() {
    local query="$1"
    echo ""
    echo "🔍 Searching for: '$query'"
    echo "--------------------------------"
    
    curl -s -X POST "http://localhost:3000/messages?sessionId=$SESSION_ID" \
        -H "Content-Type: application/json" \
        -d "{
            \"jsonrpc\": \"2.0\",
            \"id\": 1,
            \"method\": \"tools/call\",
            \"params\": {
                \"name\": \"search_memories\",
                \"arguments\": {
                    \"query\": \"$query\",
                    \"limit\": 3,
                    \"detailLevel\": \"compact\"
                }
            }
        }" && echo ""
}

echo ""
echo "🎯 Příklady dotazů, které můžete zkusit:"
echo "1. search_memory 'Chrome browser freezing'"
echo "2. search_memory 'agent desktop problems'"
echo "3. search_memory 'status change delay'"
echo "4. search_memory 'Genesys Cloud issues'"
echo ""
echo "💡 Nebo použijte funkci přímo:"
echo "   search_memory 'váš dotaz zde'"
echo ""

# Example searches
echo "🚀 Demo vyhledávání:"
search_memory "Chrome freezing issues"
sleep 1
search_memory "agent status problems"