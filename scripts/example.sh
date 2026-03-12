#!/usr/bin/env bash
# example.sh - Demonstrates how to call the /chat endpoint.
# The server must be running locally on port 8000.
#   uvicorn src.app:app --reload

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Sending a math question to the agent..."

curl -s -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Please add 37 and 56 together. Run add_two_numbers tool to get the answer",
    "session_id": "demo-001"
  }' | python3 -m json.tool
