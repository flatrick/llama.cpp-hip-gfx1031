ROCM_IMAGE="${ROCM_IMAGE:-llama-cpp-gfx1031:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNTIME=$(command -v podman || command -v docker)
if [ -z "$RUNTIME" ]; then
  echo "ERROR: neither podman nor docker found in PATH" >&2; exit 1
fi

if ! "$RUNTIME" image inspect "$ROCM_IMAGE" > /dev/null 2>&1; then
  echo "ERROR: image $ROCM_IMAGE not found — run ./build.docker-rocm.sh first" >&2; exit 1
fi

"$RUNTIME" run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/llama.cpp:/root/.cache/llama.cpp" \
  -p 8080:8080 \
  "$ROCM_IMAGE" \
    -hf "unsloth/Qwen3.5-2B-GGUF:UD-Q5_K_XL" \
    --ctx-size 209715 \
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
