# llamactl — Persistent TUI Dashboard for llama.cpp — Design

**Date:** 2026-06-10
**Status:** Approved (design); ready for implementation planning
**Scope:** A new persistent TUI dashboard (`llamactl`) that replaces the manual
CLI workflow for launching, building, monitoring, and tuning llama-server on
this repo's ROCm/Vulkan + gfx1031 setup.

## Problem

The repo can already do everything it needs to (run.py launcher, four build
scripts, VRAM tools, stress harness), but the experience is tedious:

- Every launch is a hand-typed CLI invocation (model, backend, container flag,
  preset, overrides, image tag).
- Every new llama.cpp setting requires editing `build_server_args()` in
  `run.py` — the launcher hardcodes each flag it knows about.
- Building a fresh llama.cpp (latest tag, branch HEAD, or specific commit)
  means remembering which of four shell scripts to run with which arguments.
- Switching model/backend/run-mode combinations means re-typing most of it.
- There is no single place to see what is running, how much VRAM it uses, and
  whether a configuration will survive an OOM boundary test.

## Decisions (from brainstorm 2026-06-10)

| Decision | Choice |
|----------|--------|
| TUI mode | **Persistent dashboard** (not a one-shot wizard) |
| v1 scope | Launch flow **plus all four**: build management, live VRAM monitoring, OOM smoke test, settings editor |
| Settings model | **Convention + escape hatch**: mechanical key→flag mapping, tiny exceptions table, no allowlist |
| Server lifecycle | **Detached**; TUI re-attaches on restart; closing the TUI never stops the server |
| run.py fate | **Frozen as-is** (legacy path); TUI is fully self-contained |
| Model configs | **TUI gets its own config dir** (`configs/`, TOML); one-shot migration from `models/*.json` |
| Tech stack | **Python + Textual** |
| Build scope | **Containers AND native binaries** |
| Architecture | **A: Layered core + Textual front-end** (rejected: thin wrapper over existing scripts — conflicts with frozen run.py and new config dir; rejected: daemon + client — overkill for single user/GPU) |

## Architecture

New top-level package with a strict one-way dependency `ui → core`
(core never imports UI):

```
llamactl/
  core/
    config.py      # TOML config store: per-model configs, global config, JSON migration
    mapper.py      # settings dict → llama-server argv (convention + exceptions table)
    runtime.py     # podman/docker detection, container queries (ps/logs/stop by label)
    lifecycle.py   # launch / stop / status / re-attach for container AND native servers
    builds.py      # source ref resolution, image builds, native cmake builds, artifact registry
    monitor.py     # health + /slots polling, per-process VRAM via /proc fdinfo (ported from vram_inspect)
    estimate.py    # pre-launch VRAM estimate (reuses vram_calc's GGUF-metadata logic)
  ui/
    app.py         # Textual App with tabs
    screens/       # serve, models (settings editor), builds, test (OOM check)
configs/
  llamactl.toml    # global: default backend/run-mode, port, VRAM budget, cache paths, name prefix
  models/*.toml    # one file per model (migrated once from models/*.json)
state/             # gitignored: artifact registry, pidfiles, native logs, src cache, last-run info
```

- Entry point: `python -m llamactl` opens the dashboard. A non-interactive
  CLI (`python -m llamactl launch …`) is nearly free later but **not in v1**.
- Legacy untouched: `run.py`, `models/*.json`, `build.*.sh` remain as-is.
- `stress_harness/` is imported by the test tab; `vram_inspect.py` /
  `vram_calc.py` stay as standalone scripts, with their core logic
  extracted/ported into `core/monitor.py` / `core/estimate.py`.

## Config schema

Per-model TOML (`configs/models/<id>.toml`):

```toml
name = "Qwen3.6-35B-A3B UD-Q5_K_XL (unsloth)"
hf = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q5_K_XL"

[settings]            # flat dict — every key maps to a llama-server flag
ctx_size = 262144
batch_size = 1024
flash_attn = "on"
n_cpu_moe = 33        # comments survive (tomlkit)
cram = 2048

[backends.rocm]       # overrides when backend == rocm
cache_type_k = "q8_0"
cache_type_v = "q8_0"
no_mmap = true

[backends.vulkan]
cache_type_k = "q8_0"
cache_type_v = "q8_0"

[presets.thinking-budgeted]
reasoning = "on"
reasoning_budget = 8192
temp = 0.6

[images]              # optional per-model image pins
rocm = "llama-cpp-gfx1031:b8586"
```

- Resolution order unchanged from run.py:
  `settings → preset → backend → UI overrides`.
- Key names mirror the actual llama-server flag (`cache_type_k`, not the
  legacy `cache_k`) so mapping stays mechanical.
- `python -m llamactl migrate` converts `models/*.json` once (renaming legacy
  keys); never runs again unless explicitly asked.
- Global config (`configs/llamactl.toml`): default backend, default run-mode,
  port, VRAM warn budget (11 GB), HF/llama.cpp cache paths, container name
  prefix.

## Settings mapper (`core/mapper.py`)

1. `some_key = value` → `--some-key value` (snake_case → kebab-case, value
   stringified).
2. `flag = true` → `--flag` present; `false`/absent → omitted.
3. Lists → flag repeated per element.
4. Exceptions table for oddballs: `cram → "-cram"`, `hf → "-hf"` (single-dash
   flags). Rarely touched.
5. Reserved keys owned by the launcher (`host`, `port`) are injected by
   lifecycle, never read from model files.
6. **No allowlist.** Unknown keys flow straight through, so new upstream
   flags need only a TOML edit — zero code changes.
7. No magic string values; an absent key is the only "off".

Safety net instead of validation: every launch shows the exact resolved argv
before starting, and an opt-in "validate" action runs `llama-server --help`
from the selected artifact and warns (does not block) on unrecognized flags.

## Server lifecycle (`core/lifecycle.py`)

One managed server at a time (single GPU); discovery is label-based so
hand-started extras don't break anything.

- **Container mode:** `podman run -d --name llamactl-<model>` with labels
  `llamactl.managed=1`, `llamactl.model=…`, `llamactl.backend=…`,
  `llamactl.preset=…`, plus the device/group/volume flags run.py uses today
  (ROCm: `/dev/kfd` + `/dev/dri` + video/render groups; Vulkan: per-node DRI
  passthrough; HF and llama.cpp caches mounted). Re-attach =
  `podman ps --filter label=llamactl.managed`; logs = `podman logs -f`;
  stop = `podman stop`. Labels let the TUI reconstruct *what* is running.
- **Native mode:** spawn with `start_new_session=True`, stdout/stderr →
  `state/logs/<timestamp>.log`, metadata to `state/native-server.json`
  (pid, model, backend, settings hash, log path). Re-attach = pidfile +
  `/proc/<pid>/cmdline` sanity check; logs = tail the file.
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` set for native ROCm exactly as run.py
  does.
- **Model switch** = stop current → launch new, exposed as one
  "restart with…" action.
- Health state machine, polled by `monitor.py`:
  `starting → loading → ready (/health 200) → unhealthy/exited`.

## Build subsystem (`core/builds.py`)

A build request = **source ref × target**.

- **Source refs:** `submodule` (pinned `llama.cpp-src`), `latest-tag`,
  `tag:<name>`, `branch:<name>` (HEAD), `commit:<sha>`. Upstream refs are
  listed via `git ls-remote` (no clone needed), then fetched shallow into a
  cached clone under `state/src-cache/` reused across builds (no fresh /tmp
  clones).
- **Targets:** `rocm-image`, `vulkan-image` (reuse `Dockerfile.rocm` /
  `Dockerfile.vulkan` unchanged), `rocm-native`, `vulkan-native` (host CMake
  builds with the same flags the Dockerfiles use). If the host toolchain for
  a native target is missing, the UI says so and points at the container
  path.
- **Artifact registry** (`state/registry.toml`): per successful build —
  target, requested ref, resolved SHA, llama.cpp build number, date, image
  tag or binary path. Image tags keep today's convention
  (`llama-cpp-gfx1031:<ref>`); native binaries land in
  `state/builds/<sha>/<target>/llama-server`.
- The launch screen's artifact picker is fed from this registry plus the
  `:latest` defaults — "build at commit X, run model Y on it" is two actions
  in one UI.
- Builds run as background workers, log streamed into the builds tab, one
  build at a time. A failed build keeps its log and registers nothing.

## Dashboard screens

**Serve (home).** Status header (state, model, backend, run-mode, artifact,
port, uptime) + live VRAM gauge against the 11 GB budget + scrolling server
log pane. Launch flow: model → backend → container/native → artifact →
preset → optional per-launch key/value overrides. Pre-start: resolved argv
preview + VRAM estimate with red warning if estimate > budget. Actions:
start, stop, restart-with-changes, copy argv.

**Models (settings editor).** Left: model list. Right: editable tree of the
selected TOML (`[settings]`, `[backends.*]`, `[presets.*]`) with
add/edit/delete of arbitrary keys (free-form — the mapper passes anything
through). "Resolved view" toggle shows final merged settings for a chosen
backend+preset. Saves via `tomlkit` preserve comments. Create-new-model from
blank or duplicate; the one-shot JSON import lives here too.

**Builds.** Build request form (source ref picker with `latest-tag`
auto-resolution, branch/commit/tag input; target checkboxes), streaming build
log, artifact registry table (ref, sha, build number, date, target, in-use
marker) with delete action (removes image/binary + registry row).

**Test (OOM check).** Runs against the currently running server: a quick
boundary check derived from the stress harness (ramp to near-full context +
cold-start round, `QUICK=1`-equivalent), streaming phase progress and peak
VRAM, ending in `OK / WARN (>11 GB) / FAIL (OOM or crash)`. Full multi-phase
stress runs stay in `stress_test.py` (linked, not embedded).

Keybindings follow Textual conventions (`q` quit, tab switching, `?` help).
Quitting never touches a running server; the quit dialog reminds you it's
still up.

## Error handling

- Every subprocess failure (podman, git, cmake) surfaces in-UI with exit code
  and last log lines; nothing is silently swallowed.
- Launch failures diagnosed from the signals the stress harness already uses:
  container exited → `podman logs` tail shown; health endpoint never ready →
  "stuck loading" state with log pane focused; OOM patterns highlighted.
- Config errors fail fast naming file and key; an unparseable model file
  disables that model in the list rather than crashing the app.
- The mapper never errors on unknown keys (by design); `--help` validation is
  the opt-in check.

## Testing

Core holds the logic, so core gets the coverage (pytest):

- `mapper.py`: golden tests — settings dict in, exact argv out, covering the
  exceptions table, bools, lists.
- `config.py`: TOML round-trip, layer resolution, JSON migration tested
  against the real `models/*.json` files.
- `lifecycle.py` / `builds.py`: injected command-runner fake (no real
  podman/git in tests) covering label construction, `podman ps` re-attach
  parsing, pidfile staleness, registry updates.
- UI: a handful of Textual `Pilot` smoke tests (app boots, tabs switch,
  launch form resolves argv). The UI stays thin by construction.

## Phasing (each phase ends usable)

1. **Core foundation** — config store, mapper, migration, registry schema;
   pytest suite. Verified via dry-run parity: resolved argv for a migrated
   model ≡ `run.py --dry-run` output for the same model (modulo renamed
   keys).
2. **Serve tab** — lifecycle + monitor + home screen: launch / stop /
   re-attach / logs / live VRAM. From here the TUI replaces daily run.py use.
3. **Builds tab** — source resolution, image + native builds, registry UI.
4. **Models tab** — settings editor with resolved view.
5. **Test tab** — OOM boundary check + pre-launch estimates wired into the
   serve screen.

## Out of scope (v1)

- Non-interactive `llamactl` CLI subcommands (sole exception: the one-shot
  `migrate` helper, which is also reachable from the Models tab).
- Multi-server / multi-GPU management.
- Changing run.py, models/*.json, build.*.sh, or stress harness behavior.
- Embedding the full multi-phase stress suite in the TUI.

## Relationship to OpenSpec

Per the 2026-06-06 spec-driven retrofit design, OpenSpec is the canonical
spec system and this superpowers brainstorm is its ideation front-end.
Implementation should land through an OpenSpec change introducing a new
`llamactl` capability spec (scenario-based WHEN/THEN verification per
`openspec/config.yaml` rules), with this document referenced as the design
basis.
