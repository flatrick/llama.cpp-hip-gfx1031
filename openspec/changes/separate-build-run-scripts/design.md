## Context

The two `*.docker-rocm.sh` launcher scripts currently contain three responsibilities in one file: cloning the llama.cpp source tree, building the `llama-cpp-gfx1031:latest` image, and running the container. The build step is guarded by `if ! docker image inspect ...`, which silently skips a rebuild when the image tag already exists. This makes deliberate rebuilds (e.g. after a `Dockerfile.rocm` change or a new llama.cpp commit) non-obvious.

Validated against: ROCm 7.2.0, llama.cpp commit b8495.

## Goals / Non-Goals

**Goals:**
- Single dedicated `build.docker-rocm.sh` that handles clone + image build, shared by all docker-rocm launchers.
- `*.docker-rocm.sh` launchers become pure run scripts: no clone, no build, fail fast if image is missing.
- Support `--force` (or `-f`) flag on the build script to force a full rebuild without touching the launchers.

**Non-Goals:**
- Changing `*.docker-llamacpp.sh` scripts (they already have no build step).
- Changing `Dockerfile.rocm` content.
- Automating image versioning or tagging beyond the existing `llama-cpp-gfx1031:latest`.
- Adding a CI pipeline or makefile.

## Decisions

### Decision: One shared build script, not per-model build scripts
Both `Q4_K_M` and `UD-Q4_K_XL` docker-rocm launchers use the same image (`llama-cpp-gfx1031:latest`). A single `build.docker-rocm.sh` serves both. Per-model build scripts would duplicate clone and build logic for no benefit.

**Alternative considered**: Makefile with `build` / `run-q4km` / `run-udq4kxl` targets. Rejected — adds a new tool dependency; plain shell scripts are consistent with the existing repo convention.

### Decision: Fail fast in run scripts if image is absent
The old `if ! docker image inspect` guard in the run script silently built the image on first use. The new run scripts will check for the image and exit with a clear error message directing the user to run `build.docker-rocm.sh`. This makes the dependency explicit.

**Alternative considered**: Keep the auto-build guard in run scripts as a convenience. Rejected — it hides the distinction between the two operations, which is the whole point of this change.

### Decision: `--force` flag for rebuild, not a separate `rebuild.sh`
A single `--force` / `-f` flag on `build.docker-rocm.sh` is the minimal surface. A separate `rebuild.sh` would just call `build.docker-rocm.sh --force` and adds a file with no new logic.

## Risks / Trade-offs

- **Users must run `build.docker-rocm.sh` before first use** — previously the run script self-bootstrapped. Mitigation: clear error message in the run scripts pointing to the build script by name.
- **`--force` rebuilds always re-clone** if `llama.cpp-src/.git` is absent, which is unchanged behavior. If the source is already cloned the clone step is skipped regardless of `--force`; only the `docker build` is forced.
