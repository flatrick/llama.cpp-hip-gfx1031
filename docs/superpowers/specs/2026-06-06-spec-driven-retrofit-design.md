# Spec-Driven Retrofit — Design

**Date:** 2026-06-06
**Status:** Approved (design); ready for implementation planning
**Scope:** Retrofit Specification-Driven Design (SDD) onto this previously vibe-coded repo.

## Problem

The repo already has OpenSpec installed (`openspec/`, `.claude/` skills, `opsx:*`
commands) but it was only used for two early changes, both archived. Since then,
development was vibe-coded (e.g. commit "current state (I am way too lazy to write
all changes with this commit)"). Concretely:

- The only canonical spec is self-referential: `openspec/specs/openspec-config/spec.md`
  describes the config file itself. **No capability has a spec.**
- The large `refactor-launcher-presets` change was archived **without capability
  specs** — its proposal/design/tasks exist, but no `specs/launcher/` or
  `specs/model-configs/` delta ever landed.
- `openspec/config.yaml` is **stale and self-contradictory**: its launcher-status,
  VRAM-budget, conventions, and decision-logic sections still describe the
  *deleted* per-profile shell scripts (`Q4_K_M.4_0q.docker-rocm.sh`, etc.) from
  before the `run.py` + `models/*.json` rewrite, and it disagrees with the
  `openspec-config` spec on whether `HSA_OVERRIDE_GFX_VERSION` is set.

Two spec systems are now available: OpenSpec (in-repo) and superpowers specs
(`docs/superpowers/specs/`).

## Decisions

- **Goals (all four):** retrofit specs onto existing code, clean up / reconcile the
  stale OpenSpec config, establish going-forward discipline, and consolidate on one
  spec system.
- **Canonical spec system: OpenSpec.** It is already wired in, fits the user's
  global `openspec-workflow.md` rules, and suits this repo's scenario-based ops
  nature. Superpowers brainstorming remains an optional ideation front-end that
  feeds OpenSpec proposals.
- **Capabilities in scope (all):** `launcher`, `model-configs`, `stress-harness`,
  `vram-tools`, `docker-builds`.
- **Approach: B — phased, config-first then per-capability.** Rejected A
  (single one-shot change: review surface too large to validate specs against code)
  and C (config + discipline only: doesn't deliver the requested full retrofit).

## Target end-state

1. An accurate `openspec/config.yaml` — no references to deleted shell-script
   launchers, one consistent `HSA_OVERRIDE` story; describes the `run.py` +
   `models/*.json` world that actually exists.
2. Canonical specs for all five capabilities under `openspec/specs/`, each derived
   from current code, using WHEN/THEN scenarios per the config's spec rules.
3. A thin going-forward workflow (a section in `AGENTS.md` + spec-triage rule) so
   the next change starts from a spec.

**Method:** every spec lands the OpenSpec-correct way — via a change that is
`openspec archive`d; `openspec/specs/` is never hand-edited.

**Out of scope:** changing any runtime behavior of `run.py`, the harness, or the
Docker builds. This is documentation/spec retrofit. Discrepancies found while
speccing are *flagged as notes*, not fixed here.

## Phase 1 — Reconcile OpenSpec config

One change: `reconcile-openspec-config` (mirrors the archived `update-openspec-config`).
Runs first so later specs are written against truth.

Stale-config fix list:

| Section | Problem | Fix |
|---|---|---|
| Current Launcher Status | Table lists 6 deleted scripts (`Q4_K_M.4_0q.docker-rocm.sh`, `chatml.*`, …) | Replace with `run.py --model X --backend {rocm,vulkan} [--container] [--preset Z]` + model/preset status |
| VRAM Budget | Per-script profiles (`ctx=57344 q4_0 ~10.85 GB`) that no longer exist | Re-anchor to model-JSON reality (e.g. 9B: rocm 131072/f16, vulkan 262144/q8_0) |
| Conventions | `<weights>.<kv>.docker-rocm.sh` / `chatml.*` naming | Replace with `models/<name>.json` + `defaults`/`backends`/`presets`/`images` shape |
| Decision Logic | Routes to `*.4_0q` / `*.8_0q` profiles | Re-route to `--backend` + `--preset` + `--container` choices |
| HSA_OVERRIDE | Context says baked into the image; `openspec-config` spec says not set in native launchers — but `run.py` now sets it automatically for native ROCm | Reconcile to current truth: baked into `Dockerfile.rocm` (gfx1030 container build) **and** set by `run.py` for native ROCm (verify exact line before writing) |

Two artifacts move in this one change:
- `config.yaml` — edited directly (config file, not a canonical spec).
- `openspec/specs/openspec-config/spec.md` — carries a **MODIFIED** delta (its
  requirements still assume a launcher-script status table and old HSA guidance);
  `openspec archive` rewrites the canonical spec.

Verification: `openspec validate --all` passes; re-read of `config.yaml` shows zero
references to `*.docker-rocm.sh` profile scripts and a single consistent
`HSA_OVERRIDE` story.

## Phase 2 — Capability specs (core retrofit)

Three changes, each derived by reading current code, each archived once validated:

| Change | Capabilities | Why grouped |
|---|---|---|
| `spec-launcher` | `launcher`, `model-configs` | Resolution logic and JSON schema are two halves of one contract |
| `spec-stress-harness` | `stress-harness` | Self-contained; existing unit tests anchor scenarios |
| `spec-vram-and-docker` | `vram-tools`, `docker-builds` | Lower-churn, stable; split later only if large |

Requirement outlines (all from current code):

- **launcher** (`run.py`) — settings resolution order (`defaults→preset→backend→CLI`,
  `resolve_settings`); image resolution order (`default→images.{backend}→CLI`,
  `resolve_image`); `--backend {rocm,vulkan}`; `--container` wraps podman/docker and
  native ROCm auto-sets `HSA_OVERRIDE=10.3.0`; `-l` filtering (4 modes); `--dry-run`;
  error handling (unknown model lists options, unknown preset hard-errors with
  available presets); defaults `8080` / `0.0.0.0`.
- **model-configs** (`models/*.json`) — JSON shape (`name`, `hf`, `defaults`,
  `backends{rocm,vulkan}`, `presets`, optional `images`); one file per underlying
  model; field precedence.
- **stress-harness** (`stress_harness/`, `stress_test.py`) — ctx auto-detect from
  `/slots`|`/props`; per-process VRAM via container PIDs → port-matched PID →
  system-wide fallback; watchdog (slots/liveness/logs); five phases in order
  (Ramp→Sustained→ColdStart→Defrag→Boundary); 11 GB warning; env overrides
  (`CTX_SIZE`, `QUICK`, `*_ROUNDS`, `REQUEST_TIMEOUT`, `STALL_TIMEOUT`).
- **vram-tools** — `vram_calc` (GGUF from file/HF cache, hybrid detection, VRAM
  estimate, what-if tables); `vram_inspect` (fdinfo drm fields, dedupe by
  `drm-client-id`, watch/delta modes, port/pid targeting).
- **docker-builds** — `Dockerfile.rocm` flags (`GGML_HIP=ON`,
  `AMDGPU_TARGETS=gfx1030`, `LLAMA_CURL=ON`, `LLAMA_BUILD_BORINGSSL=ON`, baked
  `HSA_OVERRIDE`); `Dockerfile.vulkan`; `build.docker-*.sh` (submodule init,
  podman-first, default tags, `--force/--src-dir/--image`); `build.llama-ref.*`
  (clone upstream ref into `/tmp`, side-by-side image, submodule untouched).

Scenario style & validation — scenarios are WHEN/THEN and reproducible. Anchor on
commands that need **no GPU** so specs are verifiable on any machine:
- launcher / model-configs → `run.py --dry-run` and `run.py -l` output.
  e.g. *WHEN `run.py --model qwen3.5-9b --backend rocm --container --dry-run` THEN
  the printed command contains `--ctx-size 131072` and `--cache-k f16`.*
- stress-harness → the existing `tests/stress_harness/` unit tests.
- docker-builds → `build.*.sh --help` / dry inspection + Dockerfile assertions.

Scenarios that genuinely require the gfx1031 box are **tagged hardware-verified**,
not silently assumed.

## Phase 3 — Going-forward discipline

One small change: `establish-spec-workflow` (mostly doc edits, no new capability
spec; dogfoods the workflow).

Deliverables:

1. A "Spec-driven workflow" section in `AGENTS.md`: non-trivial changes start as an
   OpenSpec change (`propose → spec deltas → tasks → implement → archive`);
   `openspec/specs/` is write-once-by-archive; run `openspec validate --all` before
   commit.
2. A spec-triage rule:
   | Change type | Path |
   |---|---|
   | Typo, doc tweak, comment, version bump | Direct commit, no change |
   | New flag, behavior change, new model preset, harness/phase change | Full OpenSpec change |
   | New script or Docker pattern | Full change **+** `design.md` (per existing config rule) |
3. A definition-of-done line: spec scenarios written, `openspec validate --all`
   green, tasks checked off, then `openspec archive`.

## Execution order & gates

1. `reconcile-openspec-config` (Phase 1 — hard prerequisite)
2. `spec-launcher` (launcher + model-configs)
3. `spec-stress-harness`
4. `spec-vram-and-docker`
5. `establish-spec-workflow` (Phase 3 — last)

Changes 2–4 are independent of each other; 1 must go first; 5 is last. Each is one
logical commit, archived only after its gate passes.

Validation gate per change (none archives until all pass):
- `openspec validate --all` green.
- Every machine-checkable scenario (`--dry-run` / `-l` / unit tests) runs and matches.
- Hardware-only scenarios are explicitly tagged, not assumed.
- No canonical `openspec/specs/` file was hand-edited (delta specs + `archive` only).

## Risks

- *Specs drifting from code mid-retrofit* → behavior frozen during retrofit (out of scope).
- *Discovering real bugs/discrepancies while speccing* (e.g. config claims
  `images.{backend}` but `qwen3.5-9b.json` has no `images` key; native HSA behavior
  differs from the docstring) → recorded as flagged notes in the proposal, not fixed
  here; each becomes a candidate follow-up change.
- *Over-speccing stable code* → vram/docker grouped, lighter scenario coverage.

## Note on doc location

This design doc (the *plan for the retrofit*) lives in `docs/superpowers/specs/`.
The capability specs it produces live in `openspec/specs/` via the phases above —
OpenSpec remains canonical.
