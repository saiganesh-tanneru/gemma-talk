#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "Starting Gemma-Talk Backend Server on port 8000..."
export PYTHONPATH="$SCRIPT_DIR"
exec "$SCRIPT_DIR/server/.venv/bin/python3" -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000
