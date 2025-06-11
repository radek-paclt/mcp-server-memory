#!/bin/bash

echo "🐉 MCP Memory Server Auto-Update 🐉"
echo "===================================="

# Save current branch
CURRENT_BRANCH=$(git branch --show-current)

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "⚠️  You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

echo "📥 Fetching latest updates..."
git fetch origin main

# Check if update is needed
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ $LOCAL = $REMOTE ]; then
    echo "✅ Already up to date!"
    exit 0
fi

echo "🔄 Updates available! Pulling latest changes..."
git pull origin main

echo "📦 Installing dependencies..."
npm install

echo "🔨 Building project..."
npm run build

echo "✅ Update complete!"
echo ""
echo "🚀 To restart the MCP server in Claude:"
echo "   1. Close Claude (Ctrl+C)"
echo "   2. Start Claude again"
echo ""
echo "🐉 Your memory server is ready to fly! 🐉"