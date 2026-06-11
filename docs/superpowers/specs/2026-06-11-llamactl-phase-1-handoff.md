# llamactl Phase 1 → Phase 2 Handoff Notes

**Date:** 2026-06-11
**Status:** Phase 1 merged to main at `78c7e2b` (117 tests passing, argv parity proven).
**Purpose:** Capture implementation outcomes, decisions, and deferred review
findings that are not in the design doc
(`2026-06-10-llamactl-tui-design.md`) or the Phase 1 plan, so Phase 2
planning starts from reality rather than memory.

## What Phase 1 actually delivered

- `llamactl/core/mapper.py` — `to_argv(settings)` + `build_server_argv(hf,
  settings, host, port)`. The latter is the seam the Phase 2 lifecycle layer
  consumes: lifecycle owns `host`/`port` injection and the `-hf` model ref;
  the mapper never reads reserved keys (`host`, `port`).
- `llamactl/core/config.py` — `ConfigError` is the single failure type for
  everything config-shaped (parse errors, missing/ill-typed keys, wrong-shaped
  sections, unknown presets). `load_all` returns `(configs, errors)`; the
  Phase 2 model list must render `errors` as disabled models, not crash.
  `load_global` validates types (`_require_int/_require_float/_require_str`)
  but deliberately does NOT validate enum domains (`default_backend`,
  `default_mode`) — that was deferred to the lifecycle layer, which knows the
  legal values. Phase 2 must implement that check.
- `llamactl/core/registry.py` — deliberately does NOT use `ConfigError`:
  parse/IO errors propagate raw because the registry is self-written state.
  Consequence for Phase 2: the launch screen's artifact picker must wrap
  `load_registry` in a try/except (both `tomllib.TOMLDecodeError` and
  `TypeError` from unknown keys) and degrade to "registry unreadable", not
  crash the TUI. Writes are atomic (temp file + `replace`).
- `llamactl/core/migrate.py` + `python -m llamactl migrate` — done, ran once,
  output committed. `RUN_PY_IMPLICIT_DEFAULTS` materialized run.py's hardcoded
  fallbacks; `cache_type_k/v` were deliberately EXCLUDED (run.py's f16
  fallback equals llama-server's own default, and all real models set cache
  types per-backend — a model-level value would mislead the dashboard).
  `min_p = 0.0` and `jinja = true` MUST stay materialized: llama-server's own
  defaults differ (0.05 / off).

## Decisions made during implementation (not in the design doc)

1. **OpenSpec archiving deferred.** The change `add-llamactl-core` is
   implemented and its `tasks.md` fully checked, but NOT archived. Decide at
   Phase 2 planning: archive per-phase changes (`add-llamactl-serve`, …) or
   one archive when the dashboard is complete. The repo's openspec rules
   apply (`openspec archive` without `--yes`, never hand-edit
   `openspec/specs/`).
2. **Each phase gets its own implementation plan**, written only when the
   phase starts, from the design doc plus the then-current real APIs. Plans
   live in `docs/superpowers/plans/`.
3. **Test layout**: `tests/__init__.py`, `tests/llamactl/__init__.py`,
   `tests/stress_harness/__init__.py` exist to avoid pytest basename
   collisions (two `test_config.py` files). Root `conftest.py` guarantees
   repo-root imports. Keep this pattern for Phase 2 test files.
4. **Parity baseline**: `tests/llamactl/test_parity.py` runs the real
   `run.py --dry-run` (44 tokens for qwen3.6-35b-a3b/rocm) and skips on
   machines without the `llama-cpp-gfx1031:latest` image. It pins migration
   fidelity — do not weaken it when Phase 2 refactors.

## Deferred review findings (carry into Phase 2 work)

- `resolve_settings` returns a SHALLOW copy: a list-valued setting in the
  resolved dict aliases the `ModelConfig`'s list. Nothing mutates resolved
  settings today; if Phase 2 ever does (e.g. settings editor round-trip),
  switch to a deep copy first.
- `flash_attn` is the one on/off setting stored as a string (`"on"`), while
  `jinja`/`no_mmap`/`no_mmproj` are booleans. The Phase 4 settings editor
  cannot assume on/off settings are uniformly boolean.
- `migrate` does not warn when `ctx_size` is absent from a source JSON
  (run.py hard-required it; the new mapper just omits the flag). All 20 real
  models have it; only matters if migration is ever re-run on new inputs.
- `__main__.py` derives `REPO_ROOT` as `parent.parent` of the package — valid
  only for the repo-local layout; revisit if llamactl is ever packaged.
- `images` table values are not type-validated in `load_model` (annotated
  `dict[str, str]` but an int value would load). Harmless until something
  consumes `images` — the Phase 2 image-resolution code should validate or
  tolerate.
- Tracked `__pycache__/*.pyc` files churn in `git status` on every pytest
  run (pre-existing repo state, untouched by Phase 1). Candidate cleanup:
  `git rm -r --cached` them and gitignore (already partially ignored).

## Phase 2 (next): lifecycle + Serve tab

Design-doc scope, restated with Phase 1 reality:
- `core/runtime.py` (podman/docker detection, label-filtered ps/logs/stop) and
  `core/lifecycle.py` (detached launch via `podman run -d --name
  llamactl-<model>` + `llamactl.*` labels; native via `start_new_session` +
  pidfile under `state/`; re-attach; health state machine polling
  `/health` & `/slots` in `core/monitor.py` with fdinfo VRAM, ported from
  `vram_inspect.py`).
- Lifecycle composes: `load_global` + `load_model`/`load_all` +
  `resolve_settings` + `build_server_argv`, container device/volume flags
  replicated from `run.py` (`/dev/kfd` + `/dev/dri` + video/render groups for
  ROCm; per-node DRI passthrough for Vulkan; HF + llama.cpp cache mounts;
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` for native ROCm only).
- First Textual dependency arrives here (`textual` is NOT yet a dependency of
  anything committed; only `tomlkit` is used beyond stdlib).
- UI: Serve tab per the design (status header, VRAM gauge vs
  `vram_budget_gb`, log pane, launch flow with argv preview).
