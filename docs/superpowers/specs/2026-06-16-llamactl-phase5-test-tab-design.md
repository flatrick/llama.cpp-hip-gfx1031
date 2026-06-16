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
| Verdict thresholds | Reuse the harness success logic + global `vram_budget_gb` (11 GiB, passed to the harness as `VRAM_WARN_GB`): FAIL on any phase failure/OOM/crash, WARN when peak ≥ budget, OK otherwise; "OK (degraded)" when VRAM is unavailable. |
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
  `model_params_from_gguf`, `kv_cache_gb`, `compute_buffer_gb`.
- Reused constants (calibrated for this repo's gfx1031 setup):
  `VRAM_OVERHEAD_GB = 0.6`, `COMPUTE_BUFFER_PER_512_GB = 0.9`,
  `CACHE_TYPE_BYTES`.
- **`hf_url_to_cache_path` is NOT ported** — it expects a
  `https://huggingface.co/{org}/{repo}/resolve/{ref}/{file}` URL, but
  `ModelConfig.hf` is always the `-hf` repo-spec form `org/repo:QUANT` (verified
  across all `configs/models/*.toml`). Porting it as-is would resolve nothing,
  so every estimate would silently read "unavailable." A new resolver replaces
  it (below).

**GGUF-path resolution (the corrected design).** llama.cpp's `-hf` downloader
does **not** populate the HF-hub `models--org--repo/snapshots/` layout; it caches
into `GlobalConfig.llama_cache` (`~/.cache/llama.cpp`) with flattened filenames
plus `manifest=org=repo=QUANT.json` sidecars. The resolver therefore:

1. Parses the `org/repo:QUANT` spec from `model.hf`.
2. Looks in `global_cfg.llama_cache` — prefer reading the
   `manifest=…={QUANT}.json` to get the exact `.gguf` filename; otherwise glob a
   flattened `*{repo}*{QUANT}*.gguf` match.
3. Falls back to the HF-hub snapshot layout under `global_cfg.hf_cache` (for
   models pulled via `huggingface-cli`).
4. Returns `None` if nothing matches → the graceful "unavailable" case.

**Public API:**

```python
@dataclass(frozen=True, slots=True)
class Estimate:
    total_gb: float
    model_gb: float
    kv_gb: float
    compute_gb: float
    overhead_gb: float

def estimate_vram(
    model: ModelConfig,
    resolved_settings: dict,
    global_cfg: GlobalConfig,   # needed for llama_cache / hf_cache paths
) -> Estimate | None: ...
```

- Resolves the model's `hf` field to a local GGUF path via the resolver above.
  **If no GGUF is found, returns `None`** — the "unavailable" case. No network
  access.
- Pulls `ctx_size`, `cache_type_k`, `cache_type_v`, `batch_size` from the
  already-resolved settings dict (the Serve screen resolves these before
  building argv, so the estimate sees the same layered values the launch will
  use). Missing keys fall back to llama-server defaults already encoded in
  `vram_calc.py`.
- Pure function over (GGUF metadata + settings + cache paths); no I/O beyond
  reading the GGUF header (and a manifest sidecar). Immutable return value.
- **Units note:** `vram_calc.py` divides by `1024**3` throughout — values are
  GiB labelled "GB". The `_VramGauge` and the global budget are likewise GiB.
  Everything is consistently GiB-mislabelled-as-GB, so the estimate and the live
  gauge agree; do not "fix" one side in isolation.

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

built from `StressConfig.from_env(...)`, short-circuiting on the first phase
failure exactly as the harness runner does.

**ctx_size — let the harness auto-detect (do not force `CTX_SIZE`).** The running
server's real context may differ from the launch-time resolved setting (server
clamping, or a server started outside llamactl). `runner.run()` queries
`client.server_ctx_size()` via `/slots` precisely for this; `BoundaryPhase`
sends a prompt at `ctx_size + max_tokens*2` and asserts HTTP 400, so a stale ctx
produces a false FAIL/pass. We therefore **omit** `CTX_SIZE` and let the harness
detect it. We **do** override `VRAM_WARN_GB` from `global_cfg.vram_budget_gb` so
the verdict threshold matches the gauge budget.

**Replicating `run.run()`:** the orchestration is mechanical but must reproduce
the `server_healthy()` precheck, the `server_ctx_size()` detect, the baseline
VRAM read, the per-phase short-circuit, and the final `final_vram_gb` read —
not just the phase loop.

**Injected adapters** (all satisfy the duck-typed surface the phase classes +
the request watchdog actually call):

1. **`_MonitorVramAdapter`** — exposes `.read() -> float | None` (GiB), dividing
   `core/monitor.read_vram_kib(pid)` by `1024**2`. **PID resolution is the
   critical detail** (`ServerInfo.pid` is `None` for containers):
   - Native → `server.pid`.
   - Container → `get_container_pid(server.container_name, find_runtime())`,
     exactly as `serve.py:_poll_vram_and_health` does. **Resolve once, up front**
     and close over it — `PeakVramSampler` calls `.read()` on a 200 ms daemon
     thread, so resolving the pid via `podman inspect` per tick would spawn
     dozens of subprocesses per request. The adapter must be cheap and reentrant.
   - Returns `None` only if fdinfo is genuinely unreadable; the verdict then
     degrades (see "degraded verdict" below).
2. **`TextualReporter`** — matches `ConsoleReporter`'s **full** method surface:
   `error`, `start_run`, `start_phase`, `record_sample(phase_key, sample,
   warn_at)` (three positional args), `finish_phase`, and `finish_run(result)`.
   Two subtleties: `error` is **dual-purpose** — used both for the fatal
   "server not reachable" message and for informational mid-phase notes
   (`ColdStartPhase` prints the leak threshold via `error`), so it must NOT
   render every call as a failure banner. And `finish_run` is called **only on
   the all-phases-passed path**, so the FAIL banner must not depend on it.
3. **Runtime-inspector adapter** — the phase constructors require **both**
   `runtime_inspector` and `runtime_info`, and the watchdog calls
   `runtime_inspector.container_running(runtime_info)` on every poll plus
   `start_log_reader(...)` for failure excerpts. Container servers reuse the real
   `ContainerRuntimeInspector` pointed at llamactl's managed container. Native
   servers need a **native inspector adapter** whose `start_log_reader` returns a
   tail-reader over `server.log_path` (a re-attached native server still carries
   `log_path`; handle `log_path=None` by yielding an empty reader) and whose
   `container_running` / `container_pids` / `api_host_port` return safe `None`s.
   The injected log reader must expose the `ContainerLogReader` surface the
   phases use: `line_count`, `dump_lines`, `stop`.

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
- `cancel` is checked **between phases** — the harness phase `.run()` methods
  loop internally with blocking `send_request` (up to `request_timeout=1800s`)
  and have no cancel hook, and we cannot edit them. So Stop sets the flag, the UI
  shows "stopping…", and cancellation takes effect when the current phase
  returns; an in-flight request is not interrupted. (Finer granularity would
  require harness edits, which are out of scope.)
- Executed from the UI in a Textual `@work(thread=True)` worker so the event
  loop stays responsive.

**Thread-safety contract.** The worker thread (and the harness code it calls)
must never mutate widgets directly. The `TextualReporter` updates the UI only via
`self.app.post_message(...)` (or `call_from_thread`); the Test screen handles
those messages on the event loop to append table rows and set the banner. The
VRAM adapter's `.read()` is invoked from the `PeakVramSampler` daemon thread, so
it must be self-contained (pre-resolved pid, no shared mutable state).

## Verdict semantics

Reuses the harness's own per-phase success logic plus the global budget
(`global_cfg.vram_budget_gb`, passed in as `VRAM_WARN_GB`):

- **FAIL** — any phase reports failure: HTTP error from the server, OOM log
  pattern, health endpoint dropping, container/process crash, or stall timeout.
- **WARN** — all phases succeeded but peak VRAM ≥ budget (11 GiB).
- **OK** — all phases succeeded and peak VRAM < budget.

**Degraded verdict when VRAM is unavailable.** If the VRAM source returns `None`
throughout (e.g. fdinfo unreadable), the run is explicitly a *degraded* check:
WARN cannot be determined, and the harness's leak/drift detection in
Sustained/ColdStart/Defrag is gated on `post_vram_gb is not None`, so those
checks are **skipped**. A fully-successful degraded run reports **OK (degraded:
boundary + liveness only, peak n/a)** — the banner says so, rather than implying
a clean leak-free verdict.

**Expected runtime.** `QUICK=1` only shrinks the sustained/cold/defrag counts; it
does **not** shrink the Ramp `build_steps` (~12 ascending sizes up to
`ctx*0.95`). At large `ctx_size` (e.g. 131072) Ramp dominates and a "quick" check
can still run several minutes. The Test tab should set expectations (e.g. a
"this can take a few minutes at large context" note), not imply instant feedback.

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

- `estimate.py` math: golden test — known GGUF metadata (a real `models/*` file
  or a crafted header fixture) plus a settings dict in, expected `Estimate`
  breakdown out; honours `cache_type_k/v` byte sizes and `batch_size` scaling.
- `estimate.py` resolver: parses `org/repo:QUANT`; finds a flattened GGUF /
  reads a `manifest=…` sidecar under a fake `llama_cache`; falls back to the
  hub layout under a fake `hf_cache`; returns `None` when nothing matches.
- `oomtest.py` pid resolution: native → `server.pid`; container →
  `get_container_pid` (injected fake runtime), resolved once.
- `oomtest.py` flow: inject a fake VRAM adapter (scripted GiB readings) and a
  fake server client; assert (a) the phase sequence and short-circuit-on-failure
  behaviour, (b) verdict threshold boundaries (peak just under vs. at vs. over
  budget → OK/WARN; phase failure → FAIL), (c) the degraded path (all-`None`
  VRAM → "OK (degraded)"), (d) cancellation between phases. No real
  podman/git/server in tests.
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
  resolved settings, resolving the `org/repo:QUANT` `hf` spec against
  `GlobalConfig.llama_cache` (then `hf_cache`), and SHALL return "unavailable"
  (no estimate) when no local GGUF is found, never blocking launch.
- `core.oomtest.run_oom_check` SHALL run the QUICK phase sequence against the
  running server using llamactl's own VRAM source (PID resolved once — native
  pid, or container pid via `get_container_pid`), and SHALL classify the result
  as OK / WARN / FAIL (or "OK (degraded)" when VRAM is unavailable) per the
  budget and phase-failure rules above.

## Revision history

- **2026-06-16 (rev 2):** Incorporated a fresh-eyes design review (opus
  sub-agent). Fixes: GGUF path resolver rewritten for the `-hf`
  `org/repo:QUANT` cache layout under `llama_cache` (the ported
  `hf_url_to_cache_path` would have resolved nothing); container VRAM PID
  resolved once via `get_container_pid`; full injected method surface for the
  reporter and runtime inspector enumerated; thread-safety contract
  (`post_message`); ctx auto-detect instead of forced `CTX_SIZE`; cancellation
  scoped to between-phases; degraded-verdict semantics when VRAM is `None`;
  expected-runtime note (Ramp dominates).

with this document referenced as the design basis.
