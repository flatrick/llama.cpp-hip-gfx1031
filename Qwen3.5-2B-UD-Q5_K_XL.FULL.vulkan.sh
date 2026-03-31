#!/usr/bin/env bash
set -euo pipefail

VULKAN_IMAGE="${VULKAN_IMAGE:-llama-cpp-vulkan:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNTIME=$(command -v podman || command -v docker)
if [ -z "$RUNTIME" ]; then
  echo "ERROR: neither podman nor docker found in PATH" >&2; exit 1
fi

if ! "$RUNTIME" image inspect "$VULKAN_IMAGE" > /dev/null 2>&1; then
  echo "ERROR: image $VULKAN_IMAGE not found — run ./build.docker-vulkan.sh first" >&2; exit 1
fi

if [[ ! -d /dev/dri ]]; then
  echo "ERROR: /dev/dri is not present on the host; Vulkan device passthrough is unavailable" >&2
  exit 1
fi

DEVICE_FLAGS=()
GROUP_FLAGS=()
declare -A SEEN_GIDS=()

for node in /dev/dri/renderD* /dev/dri/card*; do
  [[ -e "$node" ]] || continue

  DEVICE_FLAGS+=(--device "$node:$node")

  gid="$(stat -c '%g' "$node")"
  if [[ -n "$gid" && -z "${SEEN_GIDS[$gid]:-}" ]]; then
    GROUP_FLAGS+=(--group-add "$gid")
    SEEN_GIDS["$gid"]=1
  fi
done

if [[ ${#DEVICE_FLAGS[@]} -eq 0 ]]; then
  echo "ERROR: no /dev/dri render or card nodes were found for Vulkan passthrough" >&2
  exit 1
fi

"$RUNTIME" run --rm \
  "${DEVICE_FLAGS[@]}" \
  "${GROUP_FLAGS[@]}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/llama.cpp:/root/.cache/llama.cpp" \
  -p 8080:8080 \
  "$VULKAN_IMAGE" \
    -hf "unsloth/Qwen3.5-2B-GGUF:UD-Q5_K_XL" \
    --ctx-size 262144 \
    --n-gpu-layers -1 \
    --batch-size 1024 \
    --ubatch-size 256 \
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
    -cram 2048 \
    --host 0.0.0.0 \
    --port 8080
