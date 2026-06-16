# llamactl Phase 3 — Builds Tab — Design

**Date:** 2026-06-16
**Status:** Approved (design); ready for implementation planning
**Basis:** `2026-06-10-llamactl-tui-design.md` (overall design), Phase 3 row of
its phasing table; Phase 1/2 as shipped.

## Scope

Phase 3 adds the **Builds tab** and its backing `core/builds.py`: turn an
upstream (or pinned) llama.cpp source ref into a runnable artifact — a container
image or a native binary — recorded in the artifact registry that the Serve
tab's picker already reads.

Each build = **source ref × target**, run one at a time as a background worker
with a live-streamed log. Phase 3 ships **all four targets**: `rocm-image`,
`vulkan-image`, `rocm-native`, `vulkan-native`.

## Decisions (from brainstorm 2026-06-16)

| Decision | Choice |
|----------|--------|
| Build implementation | **Reimplement natively in Python** (injected-`Runner` pattern), not wrapping `build.*.sh`. The shell scripts stay frozen as legacy. |
| Target scope | **All four** targets this phase (images + native). |
| Source snapshot | **Approach A**: one reused clone under `state/src-cache/`, per-build clean export via `git archive \| tar -x` into a temp build context. No fresh `/tmp` clones; no stored `.tar`. |
| Concurrency | **One build at a time**, guarded by a worker lock. |
| Registry | **Existing `core/registry.py` unchanged** — `Artifact` schema already has `image_tag`/`binary_path`. |

## Architecture

New `llamactl/core/builds.py`, obeying the established one-way dependency
`ui → core` (core never imports UI) and the injectable-`Runner` pattern from
`runtime.py` / `lifecycle.py`, so every `git` / `podman` / `cmake` call is
testable with a fake runner.

Composes with **existing** `core/registry.py` (`Artifact`,
`add_artifact` / `remove_artifact` / `load_registry` / `save_registry` shipped
in Phase 1 — no schema changes).

Core functions (no Textual imports):

- `resolve_ref(ref, runner) -> ResolvedRef(sha, build_number, display)` — ref
  spec → concrete SHA.
- `export_snapshot(sha, dest, runner)` — `git archive | tar -x` clean tree into
  a temp dir.
- `build_image(target, context_dir, image_tag, runner)` — `podman build` against
  the unchanged Dockerfile.
- `build_native(target, src_dir, out_dir, runner)` — host cmake replicating the
  Dockerfile flags.
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
| `submodule` | No network. SHA = `git -C llama.cpp-src rev-parse HEAD`; build context comes from the pinned submodule, not the cache. |
| `latest-tag` | `git ls-remote --tags` → pick highest `bNNNN` tag → fetch + resolve. `build_number` = that tag. |
| `tag:<name>` | `git fetch origin tag <name>` → `rev-parse`. `build_number` = `<name>` if it matches `bNNNN`, else `""`. |
| `branch:<name>` | `git fetch origin <name>` → `rev-parse FETCH_HEAD`. |
| `commit:<sha>` | `git fetch origin <sha>` (fetch-by-sha fallback) → `rev-parse`. |

Fetches are shallow (`--depth 1`) and incremental against the reused clone.
Tag listing for the UI picker uses `git ls-remote --tags` with no clone needed.

### Snapshot export (Approach A)

`git archive <sha>` streams a tar through a pipe straight into `tar -x`, landing
a **clean** tree (no `.git`, no stale `build/`) in a temp build context. No
`.tar` file is ever written to disk; the extracted dir is deleted after the
build. This is the hygiene `build.docker-rocm.sh` achieves with
`tar --exclude=.git --exclude=build`, sourced from a pinned SHA instead of the
working tree.

The only thing that *persists* between builds is the gitignored cache clone.
The `submodule` ref skips the cache and exports from `llama.cpp-src` directly.

## The four build targets

After `export_snapshot(sha)` produces a clean tree:

- **`rocm-image`** → `podman build -f Dockerfile.rocm -t llama-cpp-gfx1031:<sanitized-ref> <context>` (Dockerfile unchanged; its `COPY llama.cpp-src` is satisfied by the snapshot).
- **`vulkan-image`** → same with `Dockerfile.vulkan`, tag from the vulkan image prefix in global config.
- **`rocm-native`** → host cmake replicating Dockerfile.rocm exactly:
  `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=ON -DLLAMA_BUILD_BORINGSSL=ON`,
  then `cmake --build build --target llama-server -j$(nproc)`.
  Output → `state/builds/<sha>/rocm-native/llama-server`. (Runtime
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` is already handled by `lifecycle.py` for
  native ROCm.)
- **`vulkan-native`** → cmake
  `-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DLLAMA_BUILD_BORINGSSL=ON`
  → `state/builds/<sha>/vulkan-native/llama-server`.

`detect_native_toolchain` runs before a native build: ROCm needs
`cmake` / `ninja` / `hipcc` + `/opt/rocm`; Vulkan needs `cmake` / `ninja` /
`glslc` + `libvulkan-dev`. Missing → the UI refuses that target and points at
the container path (no half-run cmake). The image cmake flags live as constants
in `builds.py` with a comment pointing back to the Dockerfile they mirror, so
any drift is visible in review.

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
  row). "In-use" is computed by cross-checking the currently-running server
  (from `lifecycle`) against each artifact.

This is the same registry the Serve tab's artifact picker reads, so "build at
commit X, then run model Y on it" is two actions in one app.

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

- `resolve_ref`: parse `git ls-remote --tags` output, highest-`bNNNN`
  selection, each ref-spec branch, `build_number` derivation.
- target builders: exact argv construction for all four (golden-style),
  image-tag sanitization, native output-path layout.
- `detect_native_toolchain`: present/absent permutations.
- registry integration: success writes the right `Artifact`; simulated failure
  writes nothing.
- UI: one or two Textual `Pilot` smoke tests (tab loads, form resolves a ref,
  registry table renders).

## Out of scope (Phase 3)

- Changing `run.py`, `build.*.sh`, the Dockerfiles, or `models/*.json`.
- Multi-build concurrency.
- Cross-compiling or non-gfx1031 targets.
- The Models settings editor (Phase 4) and Test/OOM tab (Phase 5).

## Relationship to OpenSpec

Per the 2026-06-06 spec-driven retrofit design, implementation should land
through an OpenSpec change extending the `llamactl` capability spec with
scenario-based WHEN/THEN requirements for build resolution, the four targets,
registry updates, and failure modes — with this document as the design basis.
