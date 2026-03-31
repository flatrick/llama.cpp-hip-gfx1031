#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
MODEL_REF="${MODEL_REF:-unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  LLAMA_SERVER_BIN="$(command -v llama-server || true)"
fi

if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  echo "ERROR: llama-server not found in PATH; set LLAMA_SERVER_BIN=/path/to/llama-server" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "ERROR: LLAMA_SERVER_BIN is not executable: $LLAMA_SERVER_BIN" >&2
  exit 1
fi

if [[ ! -d /dev/dri ]]; then
  echo "ERROR: /dev/dri is not present on the host; Vulkan is unavailable" >&2
  exit 1
fi

if [[ ! -e /dev/dri/renderD128 ]] && ! compgen -G "/dev/dri/renderD*" > /dev/null; then
  echo "ERROR: no /dev/dri/renderD* nodes were found; Vulkan compute access is likely unavailable" >&2
  exit 1
fi

exec "$LLAMA_SERVER_BIN" \
  -hf "$MODEL_REF" \
  --ctx-size 53248 \
  --n-gpu-layers -1 \
  --batch-size 4096 \
  --ubatch-size 512 \
  --parallel 1 \
  --flash-attn on \
  --no-mmproj \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --top-k 20 \
  --top-p 0.8 \
  --temp 0.7 \
  --presence-penalty 1.5 \
  --jinja \
  --no-warmup \
  -cram 2048 \
  --host "$HOST" \
  --port "$PORT"
