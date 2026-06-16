# llamactl Phase 5 — Pre-launch VRAM Estimate + OOM Test Tab — Design

**Date:** 2026-06-16
**Status:** Approved (design); ready for implementation planning
**Scope:** The final phase of the llamactl dashboard. Adds (1) a pre-launch VRAM
estimate to the Serve screen and (2) a Test tab that runs a quick OOM boundary
check against the currently running server. Completes the v1 feature set from
the 2026-06-10 TUI design.

## Background

Phases 1–4 (core foundation, Serve tab, Builds tab, Models editor) are merged to
`main`. Phase 5 is the last item in the original phasing plan:

> 5. **Test tab** — OOM boundary check + pre-launch estimates wired into the
>    serve screen.

Two pieces remain unbuilt:

- `core/estimate.py` does not exist yet (the architecture reserved it for the
  pre-launch VRAM estimate).
- The Test tab is a placeholder `TabPane` in `ui/app.py`.

This design references the 2026-06-10 TUI design as its basis and keeps its
constraints: the strict one-way `ui → core` dependency, core-holds-the-logic
testing, and the explicit out-of-scope rule that **stress harness behavior must
not change**.

## Decisions (from brainstorm 2026-06-16)

| Decision | Choice |
|----------|--------|
| OOM test execution | **Slim `core/oomtest.py` runner** that drives the *existing* harness phase objects with llamactl's own VRAM source + a TUI reporter. Not a subprocess; not the harness's internal `StressTestRunner` (which builds its own container monitor). |
| Test phases | **Full QUICK sequence** — `Ramp → Sustained → ColdStart → Defrag → Boundary` at `QUICK=1` counts (3 sustained / 1 cold / 1 defrag). Near-parity with `stress_test.py QUICK=1`. |
| VRAM source during test | **llamactl's `core/monitor.read_vram_kib`** (fdinfo), so both **native and container** servers report real peak VRAM. |
| Estimate when GGUF absent | **Graceful "unavailable"** — local HF-cache lookup only; no network, no guessing; launch never blocked. |
| Verdict thresholds | Reuse the harness success logic + global `vram_warn_gb` (11 GB): FAIL on any phase failure/OOM/crash, WARN when peak ≥ budget, OK otherwise. |
| Harness source | **Read-only.** Phase classes are *constructed* with injected collaborators via their existing constructor params; no edits to `stress_harness/`. |

## Architecture

New modules, preserving `ui → core` (core never imports UI):

```
llamactl/core/
  estimate.py      # NEW — pre-launch VRAM estimate from GGUF metadata
  oomtest.py       # NEW — slim QUICK boundary runner over harness phases
llamactl/ui/screens/
  test.py          # NEW — Test tab (TestScreen)
  serve.py         # EDIT — estimate line beneath the argv preview
ui/app.py          # EDIT — mount TestScreen in the existing Test TabPane
```

- `stress_harness/` is imported, never modified.
- `vram_calc.py` stays a standalone script; its GGUF-metadata math is **ported**
  (extracted) into `core/estimate.py`, matching the original design's intent to
  move logic into core while leaving the legacy script in place.
- `core/monitor.py` (`read_vram_kib`, `check_health`) and `core/lifecycle.py`
  (running-server discovery for container and native) are reused as-is.

## Pre-launch VRAM estimate (`core/estimate.py`)

Ports the calibrated, self-contained VRAM math from `vram_calc.py`:

- Reused functions (ported, not imported): `read_gguf_metadata`,
  `model_params_from_gguf`, `kv_cache_gb`, `compute_buffer_gb`, and
  `hf_url_to_cache_path` for cache resolution.
- Reused constants (calibrated for this repo's gfx1031 setup):
  `VRAM_OVERHEAD_GB = 0.6`, `COMPUTE_BUFFER_PER_512_GB = 0.9`,
  `CACHE_TYPE_BYTES`.

**Public API:**

```python
@dataclass(frozen=True, slots=True)
class Estimate:
    total_gb: float
    model_gb: float
    kv_gb: float
    compute_gb: float
    overhead_gb: float

def estimate_vram(model: ModelConfig, resolved_settings: dict) -> Estimate | None: ...
```

- Resolves the model's `hf` field to a local HF-cache GGUF path. **If the GGUF
  is not in the cache, returns `None`** — the "unavailable" case. No network
  access.
- Pulls `ctx_size`, `cache_type_k`, `cache_type_v`, `batch_size` from the
  already-resolved settings dict (the Serve screen resolves these before
  building argv, so the estimate sees the same layered values the launch will
  use). Missing keys fall back to llama-server defaults already encoded in
  `vram_calc.py`.
- Pure function over (GGUF metadata + settings); no I/O beyond reading the GGUF
  header. Immutable return value.

**Serve wiring (`ui/screens/serve.py`):** `_LaunchForm._refresh_argv_preview()`
already runs after settings resolve and writes the `#argv-preview` Static. Add a
sibling `#estimate-line` Static directly beneath it:

- GGUF present:
  `Est: 10.4 GB  (model 7.1 + KV 2.7 + buf 0.6)  — budget 11 GB ✓`
- Over budget: same line, red styling, `⚠` instead of `✓`.
- GGUF absent: `Est: unavailable — model not downloaded`.

The launch button is **never disabled** by the estimate (warn-never-block, per
the original design). The existing `_VramGauge` 11 GiB budget reactive supplies
the budget value so the estimate and the live gauge agree.

## OOM boundary runner (`core/oomtest.py`)

The runner reproduces `StressTestRunner.run()`'s phase chain (~40 lines of
orchestration) so it can inject llamactl's own collaborators, which the harness
`StressTestRunner` does not allow (it constructs a container `VramMonitor`
internally). No phase logic is reimplemented — the real harness phase classes
are constructed and run.

**Phase chain (QUICK counts):**

```
RampPhase → SustainedPhase → ColdStartPhase → DefragPhase → BoundaryPhase
```

built from `StressConfig.from_env({"QUICK": "1", "API_URL": <serve url>,
"CTX_SIZE": <resolved ctx_size>})`, short-circuiting on the first phase failure
exactly as the harness runner does.

**Three injected adapters** (all satisfy the duck-typed surface the phase classes
already call):

1. **`_MonitorVramAdapter`** — exposes `.read() -> float | None` (GB), backed by
   `core/monitor.read_vram_kib(pid)`. The pid (native) or container target comes
   from `core/lifecycle`'s running-server record. Phases and `PeakVramSampler`
   only ever call `.read()`, so this is a drop-in for the harness `VramMonitor`.
   Works for native and container servers. Returns `None` if fdinfo is
   unreadable; the verdict then degrades to OK/FAIL without a VRAM number.
2. **`TextualReporter`** — matches `ConsoleReporter`'s method surface
   (`error`, `start_run`, `start_phase`, `record_sample`, `finish_phase`, and
   the run-summary call). Instead of printing, it posts Textual messages to the
   Test screen for live table/banner updates. No stdout parsing anywhere.
3. **Log reader** — for failure excerpts only. Container servers reuse the
   harness `ContainerRuntimeInspector` pointed at llamactl's managed container.
   Native servers use a small reader that tails `state/logs/<ts>.log` (path from
   `lifecycle`).

**Public API:**

```python
@dataclass(frozen=True, slots=True)
class OomTestResult:
    verdict: str          # "OK" | "WARN" | "FAIL"
    peak_vram_gb: float | None
    last_ok_tokens: int | None
    failed_phase: str | None
    detail: str           # human-readable summary / last log excerpt

def run_oom_check(server, reporter, *, cancel: Callable[[], bool]) -> OomTestResult: ...
```

- `server` is the running-server descriptor from `lifecycle` (model, backend,
  pid/container, log path, port).
- `cancel` is checked between phases and between steps; when it returns `True`
  the runner stops and returns a `FAIL`/cancelled result with whatever was
  gathered.
- Executed from the UI in a Textual `@work(thread=True)` worker so the event
  loop stays responsive.

## Verdict semantics

Reuses the harness's own per-phase success logic plus the global budget:

- **FAIL** — any phase reports failure: HTTP error from the server, OOM log
  pattern, health endpoint dropping, container/process crash, or stall timeout.
- **WARN** — all phases succeeded but peak VRAM ≥ `vram_warn_gb` (11 GB from
  global config).
- **OK** — all phases succeeded and peak VRAM < budget.

When the VRAM source returned `None` throughout, WARN cannot be determined, so a
fully-successful run reports **OK** with peak shown as `n/a`.

## Test tab UI (`ui/screens/test.py`)

- **Precondition:** reads the running-server state from `lifecycle`. With no
  managed server running, the "Run check" button is disabled and the pane shows
  "Start a server on the Serve tab first."
- **Header:** target model / backend / `ctx_size` / baseline VRAM.
- **Controls:** a single **Run / Stop** button. Run launches the worker; Stop
  sets the cancel flag.
- **Live phase table:** columns mirror the console reporter — phase, step/round,
  prefill, gen tok/s, peak VRAM, post VRAM, status — appended row-by-row from
  `TextualReporter` messages.
- **Verdict banner:** `OK` / `WARN (>11 GB)` / `FAIL (OOM/crash)` with the
  `detail` string (and last log excerpt on failure).
- **Full-suite pointer:** a static note with the exact command to run the
  complete multi-phase stress suite (`python stress_test.py`), linked — not
  embedded — per the original out-of-scope rule.
- **Mount:** replace the placeholder body of the existing `Test` `TabPane` in
  `ui/app.py` with `TestScreen()`.

## Error handling

- Server vanishes mid-test → test stops with `FAIL`, last log excerpt shown
  (the harness phases already surface this signal).
- Estimate: unreadable or corrupt GGUF header → treated as "unavailable"
  (logged once; never crashes the Serve screen).
- VRAM source unavailable (fdinfo unreadable, e.g. permissions) → `.read()`
  returns `None`; phases still run; verdict degrades to OK/FAIL without the VRAM
  number rather than erroring.
- Worker exceptions are caught at the `@work` boundary and rendered as a `FAIL`
  banner with the traceback summary; the UI never hangs in "running".

## Testing

Core holds the logic, so core gets the coverage (pytest):

- `estimate.py`: golden test — known GGUF metadata (a real `models/*` file or a
  crafted header fixture) plus a settings dict in, expected `Estimate`
  breakdown out; returns `None` when the cache path does not resolve; honours
  `cache_type_k/v` byte sizes and `batch_size` scaling.
- `oomtest.py`: inject a fake VRAM adapter (scripted GB readings) and a fake
  server client; assert (a) the phase sequence and short-circuit-on-failure
  behaviour, (b) verdict threshold boundaries (peak just under vs. at vs. over
  budget → OK/WARN; phase failure → FAIL), (c) cancellation between phases. No
  real podman/git/server in tests.
- UI: one Textual `Pilot` smoke test — Test tab disabled with no running server;
  the verdict banner renders correctly when fed a synthetic `OomTestResult`.

## Out of scope (unchanged from v1)

- Changing `stress_harness/` behavior, `run.py`, `models/*.json`, or the build
  scripts.
- Embedding the full multi-phase stress suite in the TUI (it stays in
  `stress_test.py`, linked).
- Multi-server / multi-GPU management.
- Network access for the estimate (local HF-cache only).

## Relationship to OpenSpec

Per the 2026-06-06 spec-driven retrofit design, OpenSpec is the canonical spec
system and this superpowers brainstorm is its ideation front-end. Implementation
should land through an OpenSpec change that adds scenario-based requirements to
the `llamactl` capability spec — at minimum:

- `core.estimate.estimate_vram` SHALL compute VRAM from local GGUF metadata and
  resolved settings, and SHALL return "unavailable" (no estimate) when the GGUF
  is not in the local cache, never blocking launch.
- `core.oomtest.run_oom_check` SHALL run the QUICK phase sequence against the
  running server using llamactl's own VRAM source, and SHALL classify the result
  as OK / WARN / FAIL per the budget and phase-failure rules above.

with this document referenced as the design basis.
