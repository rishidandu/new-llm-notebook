#!/bin/bash

# Start ASU RAG Frontend and API Server
echo "🚀 Starting ASU RAG System..."

# Function to cleanup background processes on exit
cleanup() {
    echo "🛑 Shutting down servers..."
    pkill -f "python.*api_server"
    pkill -f "next.*dev"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start API server in background
echo "📡 Starting API server on port 3000..."
source venv/bin/activate
python scripts/start_api_server.py &
API_PID=$!

# Wait a moment for API server to start
sleep 3

# Start Next.js frontend in background
echo "🎨 Starting Next.js frontend on port 3001..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ ASU RAG System is running!"
echo "📡 API Server: http://localhost:3000"
echo "🎨 Frontend:   http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait $API_PID $FRONTEND_PID 