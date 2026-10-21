#!/usr/bin/env bash
# Linux/POSIX setup for the twitch simulator (M0). Windows uses setup.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Twitch Simulator setup =="

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Python 3.12+ is required" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "creating .venv ..."
  if ! "$PYTHON" -m venv .venv; then
    echo "python venv module unavailable; trying venv --without-pip" >&2
    rm -rf .venv
    "$PYTHON" -m venv --without-pip .venv
    "$PYTHON" -m pip --python .venv install --upgrade pip
  fi
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

if [[ "${TWITCHSIM_SKIP_FRONTEND:-0}" != "1" ]]; then
  command -v npm >/dev/null || { echo "Node.js/npm is required (Node 20+)" >&2; exit 1; }
  (
    cd frontend
    if [[ -f package-lock.json ]]; then npm ci; else npm install; fi
    npm run build
  )
fi

.venv/bin/python scripts/make_sample.py

if command -v ffmpeg >/dev/null; then
  ffmpeg -version | head -n 1
else
  echo "warning: ffmpeg is not on PATH" >&2
fi
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "warning: nvidia-smi not found; M0.5 cannot measure VRAM" >&2
fi

chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true
echo
echo "Setup complete. Run: ./scripts/run.sh"
echo "Gate: ./scripts/fetch_models.sh 9b, then read README.md (M0.5)."
