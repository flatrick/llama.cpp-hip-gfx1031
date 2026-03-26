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

Two weight quantizations are available:

| File | Weight size | Quality |
|------|------------|---------|
| `Qwen3.5-9B-Q4_K_M.gguf` | 5.3 GB | Standard Q4 |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | 5.6 GB | Higher quality Q4 variant (unsloth dynamic) |

Each is paired with two KV cache precision options, giving four launch scripts:

| Script | Weights | KV cache | ctx-size | Est. peak VRAM |
|--------|---------|----------|----------|----------------|
| `Q4_K_M.4_0q.docker-rocm.sh` | Q4_K_M | q4_0 | 57,344 | ~10.85 GB |
| `Q4_K_M.8_0q.docker-rocm.sh` | Q4_K_M | q8_0 | 49,152 | ~10.67 GB |
| `UD-Q4_K_XL.4_0q.docker-rocm.sh` | UD-Q4_K_XL | q4_0 | 49,152 | ~10.52 GB |
| `UD-Q4_K_XL.8_0q.docker-rocm.sh` | UD-Q4_K_XL | q8_0 | 45,056 | ~10.66 GB |

All four are kept under the 11 GB safety threshold (12 GB physical, ~1 GB headroom for
driver overhead and other processes).

GGUF metadata (read via `vram_calc.py`): 32 attention layers, 4 KV heads per layer,
256 head dimension. All 32 layers use attention with GQA.

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

Then start the server with one of the four scripts:

```bash
bash Q4_K_M.4_0q.docker-rocm.sh 2>&1 | tee llama.server.log
bash Q4_K_M.8_0q.docker-rocm.sh 2>&1 | tee llama.server.log
bash UD-Q4_K_XL.4_0q.docker-rocm.sh 2>&1 | tee llama.server.log
bash UD-Q4_K_XL.8_0q.docker-rocm.sh 2>&1 | tee llama.server.log
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

Tested with `stress_test.py` (progressive context fill up to ~62K tokens, then 5 rounds
of sustained load at peak size). All four configs ran at `--ctx-size 65536` during
measurement; ctx-sizes were then reduced to keep peak VRAM under 11 GB.

### Baseline VRAM (model weights + pre-allocated KV cache)

| Config | Baseline |
|--------|----------|
| Q4_K_M + q4_0 | 6.79 GB |
| Q4_K_M + q8_0 | 7.24 GB |
| UD-Q4_K_XL + q4_0 | 7.06 GB |
| UD-Q4_K_XL + q8_0 | 7.53 GB |

### VRAM at ~62K prompt tokens (ctx-size 65536)

| Config | Peak VRAM | Gen speed | Sustained 5× |
|--------|-----------|-----------|--------------|
| Q4_K_M + q4_0 | 11.12 GB | 23.8 tok/s | ✓ |
| Q4_K_M + q8_0 | 11.57 GB | 26.2 tok/s | ✓ |
| UD-Q4_K_XL + q4_0 | 11.43 GB | 23.6 tok/s | ✓ |
| UD-Q4_K_XL + q8_0 | 11.84 GB | 25.0 tok/s | ✓ |

All four configs passed the full ramp and sustained load. The earlier failure of
Q4_K_M + q8_0 was due to ~3 GB of VRAM already in use by another process at the time
of testing — not a model or configuration issue.

### ctx-size limits derived from measurements (target: peak < 11 GB)

| Config | Safe ctx-size | Est. peak | Headroom |
|--------|--------------|-----------|----------|
| Q4_K_M + q4_0 | 57,344 | ~10.85 GB | ~0.15 GB |
| Q4_K_M + q8_0 | 49,152 | ~10.67 GB | ~0.33 GB |
| UD-Q4_K_XL + q4_0 | 49,152 | ~10.52 GB | ~0.48 GB |
| UD-Q4_K_XL + q8_0 | 45,056 | ~10.66 GB | ~0.34 GB |

## Tuning guide

Use `vram_calc.py` to estimate the effect of changes before applying them.
Run `stress_test.py` after changes to measure the real impact. Pass `CTX_SIZE=N` to
match the server's actual `--ctx-size` so the test steps bracket the real limit:

```bash
CTX_SIZE=49152 sudo python stress_test.py
```

### Parameters and trade-offs

| Parameter | Increase | Decrease |
|-----------|----------|----------|
| `--ctx-size` | More conversation history; server refuses oversized requests gracefully | Less history; lower baseline VRAM |
| `--ubatch-size` | Fewer GPU dispatches per batch, faster prefill | Smaller FA working memory per dispatch |
| `--cache-type-k/v` | `q8_0` = ~2 tok/s faster decode, more VRAM; `q4_0` = half the KV cache, slightly fuzzier recall of old tokens | — |
| `--batch-size` | Faster prompt processing (prefill) | Slower prefill; does NOT change the OOM boundary |
| `--flash-attn` | `on` = lower VRAM for long contexts, faster; `off` = more VRAM (full attention matrix), slower | — |

### q4_0 vs q8_0 KV cache

**q8_0 is ~2 tok/s faster** than q4_0 at the same model, consistently across both
weight variants. The GPU decodes 8-bit aligned data more efficiently — better memory
bandwidth utilisation during the decode phase outweighs the smaller cache size of q4_0.

The trade-off is context window: q8_0's larger KV cache costs ~8K tokens of ctx budget
compared to q4_0 at the same VRAM limit. Choose based on what matters more for your
workload — longer context or faster generation.

### Current configs (all four scripts)

```
--ubatch-size 256       # smaller FA tile dispatch; reduces peak working memory
--batch-size 1024       # faster prefill; startup VRAM unaffected vs 512
--flash-attn on         # required at these context sizes
--defrag-thold 0.1      # prevents fragmentation during long agentic sessions
--no-warmup             # defers buffer allocation to first inference
--parallel 1            # single slot; no multi-user overhead
```

Per-script ctx-size and cache-type:

| Script | ctx-size | cache-type-k/v |
|--------|----------|----------------|
| Q4_K_M.4_0q | 57,344 | q4_0 |
| Q4_K_M.8_0q | 49,152 | q8_0 |
| UD-Q4_K_XL.4_0q | 49,152 | q4_0 |
| UD-Q4_K_XL.8_0q | 45,056 | q8_0 |

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
is set as `ENV` in the Dockerfile — no need to pass it at `docker run` time.
