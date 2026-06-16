# llamactl Phase 3 — Builds Tab — Design

**Date:** 2026-06-16
**Status:** Approved (design); ready for implementation planning
**Basis:** `2026-06-10-llamactl-tui-design.md` (overall design), Phase 3 row of
its phasing table; Phase 1/2 as shipped.

## Scope

Phase 3 adds the **Builds tab** and its backing `core/builds.py`: turn an
upstream (or pinned) llama.cpp source ref into a runnable artifact — a container
image or a native binary — recorded in the artifact registry, which this phase
also wires into the Serve tab's launch picker (Phase 2 loads the registry but
does not yet surface it).

Each build = **source ref × target**, run one at a time as a background worker
with a live-streamed log. Phase 3 ships **all four targets**: `rocm-image`,
`vulkan-image`, `rocm-native`, `vulkan-native`.

## Decisions (from brainstorm 2026-06-16)

| Decision | Choice |
|----------|--------|
| Build implementation | **Reimplement natively in Python**, not wrapping `build.*.sh`. The shell scripts stay frozen as legacy. Short git/podman calls use the blocking injected-`Runner`; long build commands use a separate streaming runner (see below). |
| Target scope | **All four** targets this phase (images + native). |
| Source snapshot | **Approach A**: one reused clone under `state/src-cache/`, per-build clean export via `git archive \| tar -x` into a temp build context. No fresh `/tmp` clones; no stored `.tar`. |
| Concurrency | **One build at a time**, enforced by UI button-disable (Phase 2 precedent) plus a core-level guard so a tab switch can't start a second build. |
| Registry | **Existing `core/registry.py` unchanged** — `Artifact` schema already has `image_tag`/`binary_path`. |

## Architecture

New `llamactl/core/builds.py`, obeying the established one-way dependency
`ui → core` (core never imports UI) and the injectable-`Runner` pattern from
`runtime.py` / `lifecycle.py`, so every `git` / `podman` / `cmake` call is
testable with a fake runner.

Composes with **existing** `core/registry.py` (`Artifact`,
`add_artifact` / `remove_artifact` / `load_registry` / `save_registry` shipped
in Phase 1 — no schema changes).

### Two runner abstractions

The existing `Runner = Callable[[list[str]], subprocess.CompletedProcess]`
(`runtime.py:17`) is **blocking, buffered, and hardcodes a 30 s timeout** in its
default impl (`runtime.py:20`). That is fine for short calls (`git rev-parse`,
`git ls-remote`, `git fetch`, `podman rmi`, toolchain probes) but **cannot**
carry a 15–20 min build or stream its output. So Phase 3 adds a second,
injectable abstraction for the long commands:

```
StreamRunner = Callable[[list[str], Path | None], Iterator[str]]   # cmd, cwd → log lines
```

Default impl wraps `subprocess.Popen` with merged stdout/stderr, no timeout,
line-buffered iteration; it yields each line and finishes by raising/encoding a
nonzero exit so the orchestrator can detect failure. `podman build` and
`cmake` go through `StreamRunner`; everything else through the blocking
`Runner`. Both are injected, so tests use a fake that returns canned lines /
exit codes (matching the `CompletedProcess` fakes already in
`tests/llamactl/test_lifecycle.py`).

### Core functions (no Textual imports)

- `resolve_ref(ref, runner) -> ResolvedRef(sha, build_number, display)` — ref
  spec → concrete SHA (uses blocking `Runner`).
- `export_snapshot(sha, src, dest_runner)` — `git archive | tar -x` clean tree
  into a temp dir.
- `build_image(target, context_dir, image_tag, stream_runner) -> Iterator[str]` —
  `podman build` against the unchanged Dockerfile.
- `build_native(target, src_dir, out_dir, stream_runner) -> Iterator[str]` —
  host cmake replicating the Dockerfile flags.
- `detect_native_toolchain(target) -> ToolchainStatus` — host has what the
  target needs.
- `run_build(request, ...)` — orchestrator: resolve → snapshot → build →
  registry, yielding log lines for the UI worker to stream.

State lives under the gitignored `state/`:

```
state/
  src-cache/llama.cpp/        # one reused clone (gitignored)
  builds/<sha>/<target>/llama-server   # native binaries
  registry.toml               # artifact registry (Phase 1)
```

## Source-ref resolution (`resolve_ref`)

Maps a ref spec to a concrete SHA + `build_number`, using the cache clone at
`state/src-cache/llama.cpp` (created on first use via
`git clone https://github.com/ggml-org/llama.cpp.git`):

| Ref spec | Resolution |
|----------|-----------|
| `submodule` | No network. SHA = `git -C llama.cpp-src rev-parse HEAD`; export still routes through `export_snapshot` (so a stale `llama.cpp-src/build/` never leaks into the context). `build_number` = `""`. |
| `latest-tag` | `git ls-remote --tags` → select highest **build** tag (see below) → fetch + resolve. `build_number` = that tag. |
| `tag:<name>` | `git fetch origin tag <name>` → `rev-parse`. `build_number` = `<name>` if it matches `^b\d+$`, else `""`. |
| `branch:<name>` | `git fetch origin <name>` → `rev-parse FETCH_HEAD`. `build_number` = `""`. |
| `commit:<sha>` | `git fetch origin <sha>` (fetch-by-sha fallback) → `rev-parse`. `build_number` = `""`. |

**Highest-build-tag selection (must be numeric, not lexical).** `git ls-remote
--tags` returns build tags (`b9665`) alongside decoys that *contain* a `bNNNN`
substring (`ci_cublas-…`, `fix-release-…-b7083-…`). Selection MUST: keep only
tags matching `^b\d+$`, parse the trailing integer, and pick the numeric max —
a lexical sort wrongly orders `b9` after `b1000`. (The legacy scripts sidestep
this with `git describe` and are no reference here.)

`build_number` is `""` for every non-build-tag ref, matching
`Artifact.build_number`'s documented "`""` if unknown" (`registry.py:26`); no
`git describe` derivation is invented.

Fetches are shallow (`--depth 1`) and incremental against the reused clone.
Tag listing for the UI picker uses `git ls-remote --tags` with no clone needed.

### Snapshot export (Approach A)

`git archive <sha>` streams a tar through a pipe straight into `tar -x`, landing
a **clean** tree (no `.git`, no stale `build/`) in a temp build context. No
`.tar` file is ever written to disk. The context dir is created under
`state/builds/tmp/` (gitignored) via `tempfile.mkdtemp` and removed in a
`try/finally` — even on build failure — mirroring the legacy scripts'
`trap cleanup EXIT`. This is the hygiene `build.docker-rocm.sh` achieves with
`tar --exclude=.git --exclude=build`, sourced from a pinned SHA instead of the
working tree.

Shallow-fetch caveat: `git archive <sha>` only succeeds if `<sha>` is an object
the cache clone actually has. Each ref is fetched (`--depth 1`) into the cache
*before* archive, so the resolved SHA is always present; `submodule` archives
from `llama.cpp-src` directly.

The only thing that *persists* between builds is the gitignored cache clone.
The `submodule` ref skips the cache and exports from `llama.cpp-src` directly.

## The four build targets

After `export_snapshot(sha)` produces a clean tree:

Image tags follow the **legacy/lifecycle convention** as `builds.py` constants
— rocm → `llama-cpp-gfx1031:<sanitized-ref>`, vulkan →
`llama-cpp-vulkan:<sanitized-ref>` (matching `lifecycle.DEFAULT_ROCM_IMAGE` /
`DEFAULT_VULKAN_IMAGE`, `lifecycle.py:36-37`). There is **no** image-prefix
field in `GlobalConfig` and Phase 3 does not add one.

- **`rocm-image`** → `podman build -f Dockerfile.rocm -t llama-cpp-gfx1031:<sanitized-ref> <context>` (Dockerfile unchanged; its `COPY llama.cpp-src` is satisfied by the snapshot).
- **`vulkan-image`** → same with `Dockerfile.vulkan -t llama-cpp-vulkan:<sanitized-ref>`.
- **`rocm-native`** → host cmake replicating Dockerfile.rocm exactly:
  `-DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1030" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=ON -DLLAMA_BUILD_BORINGSSL=ON`,
  then `cmake --build build --target llama-server -j$(nproc)`. The built binary
  is at `build/bin/llama-server`; copy it to
  `state/builds/<sha>/rocm-native/llama-server`. (Runtime
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` is already handled by `lifecycle.py:217` for
  native ROCm — not a build flag.)
- **`vulkan-native`** → cmake
  `-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DLLAMA_BUILD_BORINGSSL=ON`,
  binary `build/bin/llama-server` → `state/builds/<sha>/vulkan-native/llama-server`.

`detect_native_toolchain` runs before a native build and must check the
build-time deps the Dockerfile base images provide implicitly, not just the
compilers:
- **rocm:** `cmake`, `ninja`, `hipcc`, `/opt/rocm` present, **and** hipBLAS dev
  libs (`hipblas-dev` — without them `-DGGML_HIP=ON` fails to configure) +
  `libcurl` dev (for `-DLLAMA_CURL=ON`).
- **vulkan:** `cmake`, `ninja`, `glslc`, `libvulkan-dev`, **and** `libcurl` dev.

Missing → the UI refuses that target and points at the container path (no
half-run cmake). A configure-time failure despite detection still surfaces via
the normal error path. The image cmake flags live as constants in `builds.py`
with a comment pointing back to the Dockerfile they mirror, so any drift is
visible in review.

## Registry & the Builds screen

On success: one `registry.add_artifact(Artifact(target, requested_ref, sha,
build_number, built_at, image_tag=… | binary_path=…))`, atomic-saved.
`add_artifact` already de-dupes on `(target, sha)`, so rebuilding the same SHA
replaces the row.

**Builds tab** (per the overall design's screen spec):

- **Request form**: source-ref picker (`latest-tag` auto-resolves and shows the
  resolved tag; text input for `tag:` / `branch:` / `commit:`) + target
  checkboxes (native targets greyed with a reason when the toolchain is
  absent). "Build" disabled while a build runs.
- **Streaming log pane**: live stdout from the running build (Textual
  thread-worker → posted messages).
- **Artifact registry table**: ref, short sha, build number, date, target,
  in-use marker, with a **delete** action (`podman rmi <tag>` for images /
  `rm -rf state/builds/<sha>/<target>` for native, then remove the registry
  row).

**In-use marker.** A running `ServerInfo` from `lifecycle.find_running` carries
no image tag or binary path (`lifecycle.py:238-302`), so in-use cannot be read
from `lifecycle` alone. Resolve it explicitly: for a container server,
`podman inspect` the running container's image and match against
`Artifact.image_tag`; for a native server, match `/proc/<pid>/cmdline` argv[0]
against `Artifact.binary_path`. If that lookup fails, the marker is omitted
(best-effort, never blocks).

**Serve-picker wiring is Phase 3 work, not a given.** `app.py:38` already
*loads* the registry into `self._artifacts`, but Phase 2's `serve.py` never
surfaces it — the launch form still chooses an image via `resolve_image()`
(model pins + defaults). Wiring the Serve artifact picker to read `_artifacts`
(per the overall design's Serve screen) is part of this phase, so "build at
commit X, then run model Y on it" becomes two actions in one app.

## Error handling

Mirrors the rest of llamactl: every `git` / `podman` / `cmake` nonzero exit
surfaces in-UI with the exit code and the tail of its log — nothing swallowed.
A failed build keeps its streamed log visible and writes **no** registry row,
so the registry only ever lists artifacts that exist. Network failure during
fetch, an unresolvable ref, a missing native toolchain, and a `podman build`
failure each get a distinct, named message. Deleting an artifact whose
image/binary is already gone still removes the stale registry row
(self-healing).

## Testing

Core holds the logic, so core gets the coverage (pytest, fake runner — no real
git / podman / cmake):

- `resolve_ref`: parse `git ls-remote --tags` output; **numeric** highest-build
  selection with a `b9 / b99 / b100 / b1000` ordering case plus `ci_*` /
  `fix-*-bNNNN-*` decoys that must be rejected; each ref-spec branch;
  `build_number` is `""` for non-build-tag refs.
- target builders: exact argv construction for all four (golden-style, against
  both the blocking `Runner` and the fake `StreamRunner`), image-tag
  sanitization, native output-path layout (`build/bin/llama-server` → dest).
- `detect_native_toolchain`: present/absent permutations incl. the implicit
  deps (hipBLAS, libcurl, glslc).
- registry integration: success writes the right `Artifact`; a mid-pipeline
  build failure writes **no** row.
- UI: one or two Textual `Pilot` smoke tests (tab loads, form resolves a ref,
  registry table renders).

## Out of scope (Phase 3)

- Changing `run.py`, `build.*.sh`, the Dockerfiles, or `models/*.json`.
- Multi-build concurrency.
- Cross-compiling or non-gfx1031 targets.
- The Models settings editor (Phase 4) and Test/OOM tab (Phase 5).

**Known limitation (not fixed here):** `Dockerfile.rocm` pins
`rocm/dev-ubuntu-24.04:latest`, so the same SHA built at different times can
yield different binaries. Since `add_artifact` de-dupes on `(target, sha)`, a
rebuild silently replaces the prior row and its provenance. Acceptable for a
single-user setup; pinning the base image is out of scope.

## Relationship to OpenSpec

Per the 2026-06-06 spec-driven retrofit design, implementation should land
through an OpenSpec change extending the `llamactl` capability spec with
scenario-based WHEN/THEN requirements for build resolution, the four targets,
registry updates, and failure modes — with this document as the design basis.
