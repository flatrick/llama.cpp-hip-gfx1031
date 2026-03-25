## Why

The current `*.docker-rocm.sh` launcher scripts embed both image-build logic (git clone + `docker build`) and container-run logic (`docker run`) in the same file, mixing two concerns that have different lifecycles. Rebuilding the image requires editing or re-running a launcher script, and the embedded `if ! docker image inspect` guard is easy to miss when a deliberate rebuild is needed.

## What Changes

- **New**: a dedicated `build.docker-rocm.sh` script that handles source clone + image build, with explicit flags for forced rebuild.
- **Modified**: `llama.server.Q4_K_M.docker-rocm.sh` — strip build logic; assume image already exists and fail fast if it does not.
- **Modified**: `llama.server.UD-Q4_K_XL.docker-rocm.sh` — same stripping as above.
- The `*.docker-llamacpp.sh` scripts are unaffected (they use a pre-built upstream image and already have no build step).
- Native `.sh` scripts are unaffected.

## Capabilities

### New Capabilities

- `container-image-build`: A standalone script for cloning the llama.cpp source and building the custom `llama-cpp-gfx1031:latest` ROCm image from `Dockerfile.rocm`. Supports optional `--force` flag to rebuild even when the image already exists.

### Modified Capabilities

*(none — run behavior is unchanged; only script structure changes)*

## Impact

- Affects: `llama.server.Q4_K_M.docker-rocm.sh`, `llama.server.UD-Q4_K_XL.docker-rocm.sh`
- New file: `build.docker-rocm.sh`
- Host system only — no container or Dockerfile changes
- Users must run `build.docker-rocm.sh` once before using any `*.docker-rocm.sh` launcher
