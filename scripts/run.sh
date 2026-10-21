#!/usr/bin/env bash
# Start the UI (and the selected model sidecar, if data/model_profile.json exists).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv found. Run ./scripts/setup.sh first." >&2
  exit 1
fi
if [[ ! -f frontend/dist/index.html ]]; then
  echo "building frontend ..."
  (cd frontend && npm run build)
fi

model_pid=""
cleanup() {
  if [[ -n "$model_pid" ]]; then
    kill "$model_pid" 2>/dev/null || true
    wait "$model_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -f data/model_profile.json && "${WITH_MODEL:-1}" != "0" ]]; then
  echo "starting llama-server from data/model_profile.json ..."
  .venv/bin/python scripts/serve_model.py data/model_profile.json &
  model_pid=$!
fi

if [[ "${NO_BROWSER:-0}" != "1" ]] && command -v xdg-open >/dev/null; then
  (sleep 1; xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true) &
fi

.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT" "$@"
