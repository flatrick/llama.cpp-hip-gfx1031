ROCM_IMAGE="${ROCM_IMAGE:-llama-cpp-gfx1031:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNTIME=$(command -v podman || command -v docker)
if [ -z "$RUNTIME" ]; then
  echo "ERROR: neither podman nor docker found in PATH" >&2; exit 1
fi

if ! "$RUNTIME" image inspect "$ROCM_IMAGE" > /dev/null 2>&1; then
  echo "ERROR: image $ROCM_IMAGE not found — run ./build.docker-rocm.sh first" >&2; exit 1
fi

# ctx-size budget (ROCm f16/f16 KV cache):
#   Fixed VRAM (weights + compute + overhead): ~6.05 GB
#   f16 KV per token: ~33 KiB  →  rate: 0.03319 MiB/token
#   11 GB ceiling → max ctx: (11264 - 6202) / 0.03319 ≈ 152,500 tokens
#
#   131,072 (~128k): ~10.3 GB — desktop-safe (~700 MB headroom for compositor)
#   147,456 (~144k): ~10.8 GB — headless only (~200 MB headroom, compositor will OOM)
#
# f16 KV cache is used instead of q8_0 because ROCm dequantizes any quantized
# KV type to float32 internally; those buffers accumulate in the GGML HIP pool
# and are never freed, causing VRAM to grow to ~11.5 GB at full context with
# q8_0. With f16 there is no dequantization, VRAM stays flat at ~8.1 GB at
# 66k and scales linearly with ctx-size.

"$RUNTIME" run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/llama.cpp:/root/.cache/llama.cpp" \
  -p 8080:8080 \
  "$ROCM_IMAGE" \
    -hf "unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL" \
    --ctx-size 131072 \
    --n-gpu-layers -1 \
    --batch-size 1024 \
    --ubatch-size 256 \
    --parallel 1 \
    --flash-attn on \
    --no-mmproj \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --top-k 20 \
    --top-p 0.8 \
    --temp 0.7 \
    --presence-penalty 1.5 \
    --jinja \
    --no-warmup \
    -cram 2048 \
    --host 0.0.0.0 \
    --port 8080
