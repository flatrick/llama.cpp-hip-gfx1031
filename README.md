# Running OpenCode Against a Local llama.cpp Server

This repository contains containerized `llama-server` setups for running local
models on an AMD RX 6700 XT with ROCm or Vulkan, plus helper scripts for VRAM
planning and stress testing.

The setup is centered on a pinned `llama.cpp` submodule, ROCm and Vulkan Docker
build flows, a unified Python launcher (`run.py`) with per-model config files,
and a multi-phase stress harness.

## Current repository contents

Top-level files that matter for actually using this repo:

| File | What it does |
|------|--------------|
| `Dockerfile.rocm` | Builds a ROCm-based container image with `llama-server` compiled from the pinned `llama.cpp-src` submodule |
| `Dockerfile.vulkan` | Builds a Vulkan-based container image with `llama-server` compiled from the pinned `llama.cpp-src` submodule |
| `build.docker-rocm.sh` | Initializes the submodule if needed and builds a named ROCm image from a selected local llama.cpp checkout |
| `build.docker-vulkan.sh` | Initializes the submodule if needed and builds a named Vulkan image from a selected local llama.cpp checkout |
| `build.llama-ref.docker-rocm.sh` | Clones an upstream llama.cpp ref into `/tmp` and builds it as a separate ROCm test image without touching the pinned submodule |
| `build.llama-ref.docker-vulkan.sh` | Clones an upstream llama.cpp ref into `/tmp` and builds it as a separate Vulkan test image without touching the pinned submodule |
| `run.py` | Unified launcher — select model, backend, and optional overrides; runs `llama-server` via Docker or native binary |
| `models/` | Per-model JSON configs with backend-specific defaults for context size and KV cache type |
| `opencode.json` | OpenCode config pointing at `http://127.0.0.1:8080/v1` |
| `vram_calc.py` | Interactive VRAM estimator that can read GGUF metadata from the local Hugging Face cache |
| `vram_inspect.py` | Reads `/proc/{pid}/fdinfo` drm fields to show per-process GPU memory; supports watch and delta modes |
| `stress_test.py` | Thin CLI entrypoint for the multi-phase llama-server stress harness |
| `stress_harness/` | Reusable Python package with config, server/watchdog, runtime, VRAM, phase, and reporting logic |
| `tests/stress_harness/` | Lightweight `unittest` coverage for the refactored stress harness |
| `docs/tuning-guide.md` | Longer notes on tuning llama.cpp parameters |
| `docs/stress-test-results.md` | Verified stress test results per model/context configuration |

The `llama.cpp-src` directory is a git submodule currently pinned to commit
`7cadbfce10fc16032cfb576ca4607cd2dd183bf1`.

## Host environment

This repo was built around:

- EndeavourOS / Arch Linux
- AMD Ryzen 5 5600X
- 32 GB RAM
- AMD RX 6700 XT 12 GB (`gfx1031`)

You do not need host-side ROCm packages for this setup. The container carries its
own ROCm userspace and compiled `llama-server` binary. You do need:

- `podman` or `docker`
- access to `/dev/kfd` and `/dev/dri`
- membership in the host `video` and `render` groups

## Build the image

Build the ROCm container image once:

```bash
bash build.docker-rocm.sh
```

Build a different tag from a different local llama.cpp checkout:

```bash
bash build.docker-rocm.sh --image llama-cpp-gfx1031:my-branch --src-dir /path/to/llama.cpp
```

Force a rebuild after changing `Dockerfile.rocm`:

```bash
bash build.docker-rocm.sh --force
```

Quickly test a newer upstream llama.cpp revision in its own image:

```bash
bash build.llama-ref.docker-rocm.sh --ref b8586 --image-tag b8586 --force
```

What the build script does today:

- auto-initializes `llama.cpp-src` if the submodule is missing
- chooses `podman` first, then `docker`
- defaults to the image tag `llama-cpp-gfx1031:latest`
- can package a different local llama.cpp checkout via `--src-dir`
- can build a side-by-side upstream test image via `build.llama-ref.docker-rocm.sh`

Build the Vulkan container image once:

```bash
bash build.docker-vulkan.sh
```

Build a different Vulkan tag from a different local llama.cpp checkout:

```bash
bash build.docker-vulkan.sh --image llama-cpp-vulkan:my-branch --src-dir /path/to/llama.cpp
```

Quickly test a newer upstream llama.cpp revision in its own Vulkan image:

```bash
bash build.llama-ref.docker-vulkan.sh --ref b8495 --image-tag b8495 --force
```

## Start the server

All models are launched through the unified `run.py` script. Select a model
and a backend — settings are loaded from `models/<name>.json` and can be
overridden on the command line.

### List available models

```bash
python run.py --list-models
```

### Common invocations

```bash
# 9B model — ROCm Docker (f16 KV cache, 128k context, ~10.3 GB)
python run.py --model qwen3.5-9b --backend rocm-docker

# 9B model — Vulkan Docker (q8_0 KV cache, full 262k context, ~10.3 GB flat)
python run.py --model qwen3.5-9b --backend vulkan-docker

# 9B model — native Vulkan binary (same settings as vulkan-docker)
python run.py --model qwen3.5-9b --backend vulkan

# 4B model — Vulkan Docker at full native context
python run.py --model qwen3.5-4b --backend vulkan-docker

# 2B / 0.8B — ROCm Docker
python run.py --model qwen3.5-2b --backend rocm-docker
python run.py --model qwen3.5-0.8b --backend rocm-docker
```

### Overriding settings at launch time

Any setting from the model config can be overridden on the command line:

```bash
# Push 9B ROCm context toward the 11 GB ceiling (headless only)
python run.py --model qwen3.5-9b --backend rocm-docker --ctx-size 147456

# Run 9B Vulkan with f16 KV cache instead of the default q8_0
python run.py --model qwen3.5-9b --backend vulkan --cache-k f16 --cache-v f16

# Use a side-by-side test image
python run.py --model qwen3.5-9b --backend rocm-docker --image llama-cpp-gfx1031:b8586

# Different port
python run.py --model qwen3.5-9b --backend vulkan-docker --port 8081
```

The `--image` flag is equivalent to the old `ROCM_IMAGE` / `VULKAN_IMAGE`
environment variables, which are also still honoured if you prefer:

```bash
ROCM_IMAGE=llama-cpp-gfx1031:b8586 python run.py --model qwen3.5-9b --backend rocm-docker
```

### Preview without running

```bash
python run.py --model qwen3.5-9b --backend rocm-docker --dry-run
```

Both host HuggingFace and llama.cpp caches are mounted into the container
automatically. The server is exposed on `http://127.0.0.1:8080`.

### Model configs

Per-model defaults live in `models/<name>.json`. Each file has a `defaults`
section (shared across all backends) and a `backends` section (per-backend
overrides for ctx_size, cache_k, cache_v, etc.). CLI flags take final
precedence over both.

The currently configured models are:

| Model ID | HF reference | ROCm ctx | Vulkan ctx |
|----------|-------------|----------|------------|
| `qwen3.5-9b` | `unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL` | 131,072 (f16) | 262,144 (q8_0) |
| `qwen3.5-4b` | `unsloth/Qwen3.5-4B-GGUF:UD-Q5_K_XL` | 81,920 (f16) | 262,144 (q8_0) |
| `qwen3.5-2b` | `unsloth/Qwen3.5-2B-GGUF:UD-Q5_K_XL` | 209,715 (f16) | 262,144 (q8_0) |
| `qwen3.5-0.8b` | `unsloth/Qwen3.5-0.8B-GGUF:UD-Q5_K_XL` | 230,686 (f16) | 262,144 (q8_0) |

ROCm uses f16 KV cache because the ROCm HIP backend dequantizes quantized KV
types to float32 internally, causing VRAM to grow with context fill rather than
staying flat. f16 eliminates that overhead. See `docs/tuning-guide.md` for the
full explanation.

## Verify the server

Check health:

```bash
curl -s http://127.0.0.1:8080/health
```

Check the model list:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
```

Simple chat completion:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "What is 17 * 23? Answer with just the number."}
    ],
    "max_tokens": 400
  }' | jq '{answer: .choices[0].message.content, tps: .timings.predicted_per_second}'
```

## OpenCode configuration

The checked-in [`opencode.json`](./opencode.json) points OpenCode at the local
OpenAI-compatible endpoint:

- base URL: `http://127.0.0.1:8080/v1`
- API key: dummy placeholder

The model ID in `opencode.json` just needs to match whatever the running
`llama-server` advertises — it does not have to correspond to a real HuggingFace
path. Update it to match your preferred model before starting a session.

## Helper scripts

### `build.llama-ref.docker-rocm.sh`

This is the fastest way to compare a newer upstream llama.cpp revision against
the pinned local submodule. It:

- clones `https://github.com/ggml-org/llama.cpp.git` into a temporary directory under `/tmp`
- optionally checks out a requested ref such as `b8586`
- builds that checkout into its own image tag, for example `llama-cpp-gfx1031:b8586`
- leaves your checked-in `llama.cpp-src` submodule untouched

Example:

```bash
bash build.llama-ref.docker-rocm.sh --ref b8586 --image-tag b8586 --force
python run.py --model qwen3.5-4b --backend rocm-docker --image llama-cpp-gfx1031:b8586
```

### `build.llama-ref.docker-vulkan.sh`

This is the Vulkan equivalent of the ROCm helper above. It:

- clones `https://github.com/ggml-org/llama.cpp.git` into a temporary directory under `/tmp`
- optionally checks out a requested ref such as `b8495`
- builds that checkout into its own image tag, for example `llama-cpp-vulkan:b8495`
- leaves your checked-in `llama.cpp-src` submodule untouched

Example:

```bash
bash build.llama-ref.docker-vulkan.sh --ref b8495 --image-tag b8495 --force
python run.py --model qwen3.5-2b --backend vulkan-docker --image llama-cpp-vulkan:b8495
```

### `vram_inspect.py`

Reads `/proc/{pid}/fdinfo` for a running llama-server and prints all GPU memory
fields (`drm-memory-vram`, `drm-memory-gtt`, `drm-memory-cpu`, `drm-shared-*`)
in a clean table. Deduplicates by `drm-client-id` so shared BOs are not
double-counted.

```bash
# one snapshot (auto-detects llama-server by port 8080)
python vram_inspect.py

# refresh every 2 seconds — watch VRAM live while sending requests
python vram_inspect.py --watch

# take a baseline, press Enter after running a request, see exact delta
python vram_inspect.py --delta

# target a specific pid or port
python vram_inspect.py --pid 12345
python vram_inspect.py --port 8081
```

The `--delta` mode is the most useful for investigating why ROCm VRAM grows with
context: take a baseline snapshot, send a large-context request, press Enter,
and the delta table shows exactly which memory categories (VRAM, GTT, CPU,
shared) changed and by how much.

### `vram_calc.py`

Interactive VRAM calculator for llama.cpp inference. It can:

- read GGUF metadata from a local file or from the Hugging Face cache
- detect hybrid models where KV-cached attention layers are fewer than total blocks
- estimate total VRAM from model size, cache type, ctx-size, and batch-size
- print quick what-if tables for context size and cache precision

Run it with:

```bash
python vram_calc.py
```

### `stress_test.py`

`stress_test.py` is now a thin entrypoint over the internal `stress_harness/`
package. The harness:

- auto-detects the running server's actual context size from `/slots` or `/props`
- resolves the active container from the API port and tracks VRAM per-process
  through container PIDs (Docker/Podman) or by port-matched local PID (native
  llama-server), falling back to system-wide only if neither is found
- uses a request watchdog driven by `/slots`, container liveness, and container logs
- runs five phases: ramp, sustained load, cold-start, defrag stress, and boundary checks
- warns when observed VRAM crosses the 11 GB safety target

Run it against the running server with:

```bash
python stress_test.py
```

Force a specific context size instead of auto-detecting it from the server:

```bash
CTX_SIZE=32768 python stress_test.py
```

Reduce round counts for large-context testing where each round takes many minutes:

```bash
QUICK=1 CTX_SIZE=250000 python stress_test.py
```

`QUICK=1` sets `SUSTAINED_ROUNDS=3`, `COLD_ROUNDS=1`, `DEFRAG_CYCLES=1`.
Individual env vars still override: `QUICK=1 COLD_ROUNDS=2` gives cold_rounds=2.

Useful overrides while investigating slow backends:

```bash
REQUEST_TIMEOUT=3600 STALL_TIMEOUT=300 COLD_ROUNDS=3 python stress_test.py
```

## Why the Docker image targets `gfx1030`

The RX 6700 XT reports as `gfx1031`, but this setup intentionally compiles
`llama.cpp` for `gfx1030` and sets:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

That is baked into `Dockerfile.rocm`, not passed at runtime. The reason is the same
as before: rocBLAS support for `gfx1031` is unreliable here, while the `gfx1030`
path works cleanly for this card in practice.

Today the Dockerfile builds `llama-server` with:

```bash
-DGGML_HIP=ON
-DAMDGPU_TARGETS="gfx1030"
-DLLAMA_CURL=ON
-DLLAMA_BUILD_BORINGSSL=ON
```

## More tuning notes

For longer explanations of the llama.cpp flags used here, see:

- [`docs/tuning-guide.md`](./docs/tuning-guide.md)

That guide is still Qwen/RX 6700 XT oriented, but it is a better place than this
README for VRAM formulas, cache trade-offs, and prompt-processing tuning.
