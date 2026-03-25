# How I run OpenCode with a local LLM

## Environment

- OS: EndeavourOS/Arch Linux
- CPU: AMD Ryzen 5 5600X
- RAM: 32GB
- GPU: AMD RX 6700 XT 12GB (gfx1031)

## Prerequisites

No ROCm or llama.cpp packages needed on the host. The container brings its own ROCm userspace and llama.cpp binary. You only need:

- Docker or Podman (the `amdgpu` kernel driver is already built into the Arch kernel)

## LLM models

Currently using [Qwen3.5-9B-GGUF by unsloth.ai](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)

Qwen3.5-9B is a hybrid SSM/attention model (Gated Delta Net). Only 8 of 32 layers
use a KV cache, so the KV cache stays small relative to context size.

## Starting the server

Build the image once (or after `Dockerfile.rocm` changes, use `--force` to rebuild):

```bash
bash build.docker-rocm.sh
```

Then start the server:

```bash
bash llama.server.Q4_K_M.docker-rocm.sh 2>&1 | tee llama.server.log
```

```bash
bash llama.server.UD-Q4_K_XL.docker-rocm.sh 2>&1 | tee llama.server.log
```

## Configuration notes

- `--no-mmproj` is required. As of build b8495, llama.cpp auto-downloads a multimodal
  projector when using `-hf` if one is present in the repo. This model is text-only.
- `-cram 2048` enables a 2GB prompt cache in RAM, speeding up repeated context
  (e.g. system prompts).
- Model files are cached at `~/.cache/huggingface/` on the host and mounted into the
  container, so they persist across restarts and are only downloaded once.
- At 131072 ctx with f16 KV, VRAM usage is ~9500 MiB (Q4_K_M) or ~9772 MiB (UD-Q4_K_XL)
  out of 12272 MiB available.
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` is passed into the container at runtime. See the
  gfx1031 section below for why this is required.
- `--swa-full` does not apply to this model (`n_swa = 0`) and should be omitted.
- Qwen3.5 is a thinking model. Use `max_tokens` of at least 400 to give it room to
  reason before producing a final answer. The response arrives in `reasoning_content`
  (thinking) and `content` (answer) fields.

## Verify setup is running

**Check model is loaded:**
```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
```

**Basic inference — note: needs enough tokens for thinking + answer:**
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [
      {"role": "user", "content": "What is 17 * 23? Answer with just the number."}
    ],
    "max_tokens": 400
  }' | jq '{answer: .choices[0].message.content, tps: .timings.predicted_per_second}'
```

Expected: answer `391`, generation ~45 tok/sec (confirms GPU is active; CPU-only
would be ~5 tok/sec).

**Code generation test:**
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [
      {"role": "user", "content": "Write a short C# function that sums an integer array and explain it briefly."}
    ],
    "max_tokens": 600,
    "stream": false
  }' | jq '.choices[0].message.content'
```

**Prompt cache test** (second request should be faster due to cached system prompt):
```bash
for i in 1 2; do
  time curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen3.5-9B",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."}
      ],
      "max_tokens": 400
    }' > /dev/null
done
```

## gfx1031 (RX 6700 XT) — why this works and what was tried

### Root cause

gfx1031 is not officially supported by AMD ROCm. The rocBLAS library ships
**broken gfx1031 kernel binaries** (`.hsaco` files) that fail to load at the HIP
runtime level with:

```
hip_code_object.cpp:400: Assertion `err == hipSuccess' failed
```

This causes `CUBLAS_STATUS_INTERNAL_ERROR` from every hipBLAS GEMM call, making
all GPU inference fail regardless of model or settings.

gfx1030 (RX 6800/6900 XT) is officially supported and has working, tuned rocBLAS
kernels. gfx1030 and gfx1031 are ISA-identical (same RDNA2 architecture); the only
difference is CU count and memory bus width.

### The fix

Two things are required together:

1. **Compile llama.cpp with `-DAMDGPU_TARGETS="gfx1030;gfx1031"`** so the binary
   contains GPU kernels for both targets.

2. **Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` at runtime** so the HIP runtime presents
   the GPU as gfx1030. This makes both the llama.cpp kernels (gfx1030 variant) and
   the rocBLAS kernels (gfx1030, which actually load and work) get used.

Both are needed. Without the dual AMDGPU_TARGETS, the llama.cpp binary only has
gfx1031 kernels and crashes when the override makes HIP look for gfx1030. Without
the override, rocBLAS loads the broken gfx1031 binaries and crashes.

### What does NOT work

- `GGML_CUDA_FORCE_MMQ=1` — only redirects quantized-weight GEMMs; F16-weight
  tensors still route to hipBLAS and crash.
- `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` — switches compute type but rocBLAS still
  fails to load the gfx1031 kernel binaries regardless.
- Symlinking `TensileLibrary_lazy_gfx1031.dat` → `TensileLibrary_lazy_gfx1030.dat`
  inside the container — HIP won't load gfx1030 code objects on a gfx1031 device
  without `HSA_OVERRIDE_GFX_VERSION`.
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` alone with a gfx1031-only llama.cpp binary —
  the binary has no gfx1030 kernels so HIP crashes looking for them.

### Inside Docker

The base `rocm/dev-ubuntu-24.04` image contains only the standard rocBLAS with
working gfx1030 kernels and no broken gfx1031 files. The override is injected via
`-e HSA_OVERRIDE_GFX_VERSION=10.3.0` in the `docker run` command. No host-side
ROCm packages (e.g. `rocblas-gfx1031-backend`) are needed or wanted.
