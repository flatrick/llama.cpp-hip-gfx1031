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

Two quantizations are available — both are the same model weights at different
precision:

| File | Weight size | Quality |
|------|------------|---------|
| `Qwen3.5-9B-Q4_K_M.gguf` | 5.3 GB | Standard Q4 |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | 5.6 GB | Higher quality Q4 variant |

GGUF metadata (read via `vram_calc.py`): 32 attention layers, 4 KV heads per layer,
256 head dimension. Note: an earlier version of this README incorrectly stated "8 of
32 layers use a KV cache" — that was wrong. All 32 layers use attention with GQA.

## Starting the server

Build the image once (or after `Dockerfile.rocm` changes, use `--force` to rebuild):

```bash
bash build.docker-rocm.sh
```

> **Important — rebuild required after Dockerfile changes:** The image bakes in
> `HSA_OVERRIDE_GFX_VERSION=10.3.0` and compiles GPU kernels for a specific target
> (`gfx1030`). If you pull a newer version of this repo or change `Dockerfile.rocm`,
> always rebuild with `--force`:
>
> - **Why we recompile at all:** using a pre-built llama.cpp binary and simply setting
>   `HSA_OVERRIDE_GFX_VERSION` does not work — the binary uses auto-detection at
>   runtime and still fails to find working GPU kernels for the device. The GPU target
>   must be specified explicitly at compile time with `-DAMDGPU_TARGETS`. There is no
>   shortcut.
> - **Why we compile for `gfx1030` only** (changed from the earlier `gfx1030;gfx1031`):
>   testing showed the GGML HIP backend's own gfx1031 kernels work for inference, but
>   routing everything through the well-supported gfx1030 path via `HSA_OVERRIDE` is
>   cleaner — smaller binary, faster compile, and avoids the broken rocBLAS gfx1031
>   binaries entirely.
> - **Why `HSA_OVERRIDE` is in the Dockerfile instead of the launch script:** it
>   can no longer be accidentally omitted. An old image without this baked in,
>   combined with a launch script that no longer passes `-e HSA_OVERRIDE_GFX_VERSION`,
>   will run without the override — HIP will use gfx1031 kernels directly, which may
>   work for inference but bypasses the intended gfx1030 code path.

Then start the server:

```bash
bash llama.server.Q4_K_M.docker-rocm.sh 2>&1 | tee llama.server.log
```

```bash
bash llama.server.UD-Q4_K_XL.docker-rocm.sh 2>&1 | tee llama.server.log
```

## Configuration notes

- `-cram 2048` enables a 2GB prompt cache in RAM, speeding up repeated context
  (e.g. system prompts).
- Model files are cached at `~/.cache/huggingface/` on the host and mounted into the
  container, so they persist across restarts and are only downloaded once.
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` is passed into the container at runtime. See the
  gfx1031 section below for why this is required.
- `--swa-full` does not apply to this model (`n_swa = 0`) and should be omitted.
- Qwen3.5 is a thinking model. Use `max_tokens` of at least 400 to give it room to
  reason before producing a final answer. The response arrives in `reasoning_content`
  (thinking) and `content` (answer) fields.
- `--jinja` is required to use the model's own chat template, which handles Qwen3.5's
  thinking tokens correctly.
- `--flash-attn on` reduces VRAM pressure at large context sizes and improves throughput.
- `--defrag-thold 0.1` enables KV cache defragmentation during long sessions, recovering
  fragmented memory as context shifts.
- `--no-warmup` skips the initial forward pass, reducing startup time and deferring
  compute buffer allocation until first inference.

## Measured VRAM limits (RX 6700 XT 12 GB)

Tested with `stress_test.py` using both models under identical settings
(`--ctx-size 131072 --cache-type-k q4_0 --cache-type-v q4_0 --batch-size 1024`):

| prompt tokens | Q4_K_M VRAM | UD-Q4_K_XL VRAM |
|--------------|-------------|-----------------|
| baseline (idle) | 7.39 GB | 7.46 GB |
| ~32K | 10.19 GB | 10.53 GB |
| ~48K | 11.90 GB | 11.93 GB |
| ~64K | **OOM crash** | **OOM crash** |

Both models crash identically at ~64K tokens. The crash occurs in flash attention's
tile allocator (`launch_fattn`) at ~78K cumulative tokens in the attention window
(the 64K prompt builds on cached context from earlier in the conversation). This is
a hard hardware limit — model variant makes no practical difference.

**Safe operating limit: ~40–48K tokens of prompt input.**

The VRAM grows during inference because flash attention needs working memory that
scales with sequence length, not just the static KV cache. The KV cache itself
allocates dynamically as context fills (not pre-allocated at startup).

## Tuning guide

Use `vram_calc.py` to estimate the effect of changes before applying them.
Run `stress_test.py` after changes to measure the real impact.

### Parameters and trade-offs

| Parameter | Increase | Decrease |
|-----------|----------|----------|
| `--ctx-size` | More conversation history fits; KV cache grows | Less history; smaller KV cache |
| `--cache-type-k/v` | `q8_0` = better recall quality, more VRAM; `q4_0` = half the VRAM, slightly fuzzier recall of old tokens | — |
| `--batch-size` | Faster prompt processing (prefill); larger static compute buffers at startup | Slower prefill; lower startup VRAM (does NOT change the OOM crash point) |
| `--flash-attn` | `on` = lower VRAM for long contexts, faster; `off` = more VRAM (full attention matrix), slower | — |

### Current settled config (both models)

```
--ctx-size 131072       # maximum conversation window
--cache-type-k q4_0     # minimum viable cache type at this ctx size
--cache-type-v q4_0     # q8_0 would OOM at 131072 ctx
--batch-size 1024       # faster prefill; startup VRAM same as 512
--flash-attn on         # required at this context size
--defrag-thold 0.1      # helps long agentic sessions
```

**Why q4_0 and not q8_0:** At 131072 ctx, q8_0 KV cache needs ~8 GB, which combined
with model weights (~5.5 GB) exceeds 12 GB before any inference begins. q4_0 halves
the KV cache to ~4 GB. The quality tradeoff is real but minor — q4_0 introduces
slightly imprecise recall of tokens far back in context. Mitigate by having the agent
write important details to disk and re-read them rather than relying on long-range
attention recall.

**Why not reduce ctx-size to get q8_0 back:** The KV cache allocates dynamically
(not pre-allocated at startup), so reducing ctx-size does not free enough VRAM to
offset switching to q8_0. The OOM crash point does not shift meaningfully.

**Why batch-size does not help with OOM:** Batch-size affects static compute buffer
allocation at startup. The OOM crash is caused by flash attention's dynamic working
memory during inference, which scales with sequence length regardless of batch-size.
Confirmed by testing: batch-size 512 vs 1024 gives identical crash points.

## OpenCode configuration

`opencode.json` defines two model entries — one per quantization — because the
`contextLength` hint tells OpenCode how much context to use:

| Entry | contextLength |
|-------|--------------|
| `local-llama/qwen-q4km` | 40960 |
| `local-llama/qwen-udq4kxl` | 40960 |

Only one server runs at a time. Switch the `"model"` field in `opencode.json` to match
whichever launcher you started.

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

1. **Compile llama.cpp with `-DAMDGPU_TARGETS="gfx1030"`** — gfx1030 kernels are
   ISA-identical to gfx1031 and work correctly.

2. **Set `HSA_OVERRIDE_GFX_VERSION=10.3.0`** so the HIP runtime presents the GPU as
   gfx1030, making it use the working gfx1030 rocBLAS kernels. This is set as `ENV`
   in the Dockerfile so it is always active regardless of how the container is run.

### What was discovered along the way

**The GGML HIP backend with gfx1031 kernels actually works for inference** — tested
by running the old dual-target binary without `HSA_OVERRIDE_GFX_VERSION` set. The
GGML backend uses its own compiled kernels and does not route through the broken
rocBLAS gfx1031 binaries. The rocBLAS issue only affects hipBLAS GEMM calls, which
the GGML HIP backend avoids.

As a result, compiling for gfx1030 only (with the override) is the cleanest approach:
smaller binary, faster compile, uses well-tested gfx1030 code paths throughout.

**Earlier approaches that did not work (before settling on the current fix):**

- `GGML_CUDA_FORCE_MMQ=1` — only redirects quantized-weight GEMMs; F16-weight
  tensors still route to hipBLAS and crash.
- `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` — switches compute type but rocBLAS still
  fails to load the gfx1031 kernel binaries regardless.
- Symlinking `TensileLibrary_lazy_gfx1031.dat` → `TensileLibrary_lazy_gfx1030.dat`
  inside the container — HIP won't load gfx1030 code objects on a gfx1031 device
  without `HSA_OVERRIDE_GFX_VERSION`.

### Inside Docker

The base `rocm/dev-ubuntu-24.04` image contains only the standard rocBLAS with
working gfx1030 kernels and no broken gfx1031 files. `HSA_OVERRIDE_GFX_VERSION=10.3.0`
is set as `ENV` in the Dockerfile — no need to pass it at `docker run` time. No
host-side ROCm packages (e.g. `rocblas-gfx1031-backend`) are needed or wanted.
