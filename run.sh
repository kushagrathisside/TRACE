#!/usr/bin/env bash
# Start the TRACE server (Trustworthy Retrieval with Automated Continuous Evaluation).
# Usage:
#   ./run.sh               # production (port 8000)
#   ./run.sh --reload      # dev mode with auto-reload
#   PORT=9000 ./run.sh     # custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
    echo "Loaded .env"
fi

# Verify venv exists
PYTHON="$SCRIPT_DIR/venv/bin/python3"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: venv not found. Run:  python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Verify Ollama is reachable (non-fatal warning)
if ! curl -sf "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" > /dev/null 2>&1; then
    echo "WARNING: Ollama is not reachable at ${OLLAMA_BASE_URL:-http://localhost:11434}"
    echo "         Start Ollama and ensure '${LLM_MODEL_NAME:-llama3.2}' is pulled."
fi

cd "$SCRIPT_DIR/backend"
exec "$SCRIPT_DIR/venv/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    "$@"
