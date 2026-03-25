ROCM_IMAGE="llama-cpp-gfx1031:latest"
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
    --model-url "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf" \
    --ctx-size 131072 \
    --n-gpu-layers -1 \
    --batch-size 1024 \
    --parallel 1 \
    --flash-attn on \
    --defrag-thold 0.1 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --jinja \
    --no-warmup \
    -cram 2048 \
    --host 0.0.0.0 \
    --port 8080
