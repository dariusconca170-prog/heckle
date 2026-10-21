#!/usr/bin/env bash
# Download model candidates into data/models. Verify repos/file names first.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="${MODELS_DIR:-data/models}"
mkdir -p "$MODELS_DIR"
PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

if ! command -v hf >/dev/null; then
  echo "installing huggingface_hub CLI ..."
  "$PYTHON" -m pip install "huggingface_hub[cli]"
fi
HF="$(command -v hf || true)"
if [[ -z "$HF" ]]; then
  HF="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts") + "/hf")')"
fi

if [[ $# -eq 0 ]]; then
  wanted=(9b)
else
  wanted=("$@")
fi
for name in "${wanted[@]}"; do
  case "$name" in
    9b)
      echo "== 9B default candidate =="
      "$HF" download HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
        --include "*Q4_K_M*" --include "*mmproj*" --local-dir "$MODELS_DIR"
      ;;
    35b)
      echo "== 35B challenger =="
      "$HF" download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive \
        --include "*Q4_K_P*" --include "*mmproj*" --local-dir "$MODELS_DIR"
      ;;
    4b)
      echo "== 4B fit fallback =="
      "$HF" download prithivMLmods/Qwen3.5-abliterated-MAX-AIO-GGUF \
        --include "*4B*Q4*" --include "*4B*mmproj*" --local-dir "$MODELS_DIR"
      ;;
    all)
      "$0" 9b 35b 4b
      ;;
    *)
      echo "unknown profile: $name (use 9b, 35b, 4b, or all)" >&2
      exit 2
      ;;
  esac
done

echo
echo "Downloaded GGUF files:"
find "$MODELS_DIR" -name '*.gguf' -type f -print0 | while IFS= read -r -d '' file; do
  size=$(du -h "$file" | cut -f1)
  sum=$(sha256sum "$file" | cut -d' ' -f1)
  printf '%8s  %s\n          sha256 %s\n' "$size" "$file" "$sum"
done
echo
echo "Check that config/models/<profile>.toml points at these exact filenames."
echo "Do not start M1 until data/bakeoff.md names a winner."
