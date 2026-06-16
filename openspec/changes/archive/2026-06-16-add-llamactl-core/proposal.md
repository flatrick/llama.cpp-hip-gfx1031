# Add llamactl core foundation

## Why

Launching llama-server today means hand-typing long `run.py` invocations, and
every new llama.cpp flag requires editing `build_server_args()` in `run.py`.
The approved design (`docs/superpowers/specs/2026-06-10-llamactl-tui-design.md`)
introduces `llamactl`, a self-contained TUI dashboard. This change delivers its
Phase 1: the UI-agnostic core.

## What Changes

- New `llamactl/core/` package: settings→argv mapper, TOML model/global config
  store, JSON→TOML migration, artifact-registry persistence.
- New `configs/` directory: `llamactl.toml` (global) and `models/*.toml`
  (migrated once from `models/*.json`).
- New CLI entry: `python -m llamactl migrate` (the only CLI surface in Phase 1).
- No existing script changes: `run.py`, `models/*.json`, `build.*.sh`, and the
  stress harness are untouched (frozen legacy path).

## Impact

- Affected scripts: none modified; new package + configs added alongside.
- Host/container: host-only Python tooling; no container or Dockerfile changes.
- New capability spec: `llamactl`.
