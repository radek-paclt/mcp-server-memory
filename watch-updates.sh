#!/bin/bash

echo "👁️  MCP Memory Server Update Watcher"
echo "===================================="
echo "Checking for updates every 5 minutes..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Fetch without output
    git fetch origin main -q
    
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)
    
    if [ $LOCAL != $REMOTE ]; then
        echo ""
        echo "🔔 $(date '+%Y-%m-%d %H:%M:%S') - New update available!"
        echo "   Run './update.sh' to update"
        
        # Optional: Send desktop notification
        if command -v notify-send &> /dev/null; then
            notify-send "MCP Memory Update" "New version available! Run ./update.sh"
        fi
    fi
    
    sleep 300 # 5 minutes
done