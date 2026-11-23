#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
NC="\033[0m"

echo -e "${BOLD}${CYAN}=== Launching Gemma-Talk Voice Assistant ===${NC}"

trap 'echo -e "\nStopping all services..."; kill 0' SIGINT SIGTERM EXIT

# Start backend server
"$SCRIPT_DIR/start-server.sh" &
SERVER_PID=$!

# Wait for server /health to be ready
echo -e "Waiting for server warmup to complete..."
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 0.5
done
echo -e "${GREEN}✓ Backend is warmed and running on http://localhost:8000${NC}"

# Start frontend
"$SCRIPT_DIR/start-client.sh" &
CLIENT_PID=$!

echo -e "\n${BOLD}${GREEN}===================================================================${NC}"
echo -e "${BOLD}${GREEN}✓ Gemma-Talk is live!${NC}"
echo -e "Open your browser at: ${BOLD}${CYAN}http://localhost:5173${NC}"
echo -e "${BOLD}${GREEN}===================================================================${NC}\n"

wait
