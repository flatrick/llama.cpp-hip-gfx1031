# Test-pane improvements — design

Date: 2026-06-17
Status: Approved (pre-implementation)
Branch: `fix/dashboard-model-ui`

## Summary

Three cohesive improvements to the llamactl **Test** pane (`llamactl/ui/screens/test.py`),
plus one small shared helper in `llamactl/core/monitor.py`:

- **A. Quick-mode checkbox** — expose the currently-hardcoded `QUICK=1` as a UI toggle.
- **B. Responsive Stop** — make the Stop button take effect within ~one in-flight
  request instead of waiting for the whole current phase.
- **C. More live data** — show a live VRAM gauge, a phase/progress indicator, and a
  peak/headroom summary during a run, mirroring the Serve tab.

All three touch the same screen and share the VRAM-reading helper, so they ship as one
spec.

## Background

- The in-tab OOM check (`oomtest.run_oom_check`) **hardcodes `"QUICK": "1"`**
  (`oomtest.py:295`), so the Test tab always runs the reduced suite.
- `QUICK` (interpreted in `stress_harness/config.py:50`) reduces three parameters:
  - `sustained_rounds`: 3 (quick) vs 20 (full)
  - `cold_rounds`: 1 (quick) vs 8 (full)
  - `defrag_cycles`: 1 (quick) vs 10 (full)
- The footer text's `QUICK=1` mention actually documents the **external**
  `python stress_test.py` CLI, not the tab — a source of confusion.
- `cancel()` is only checked **between phases** in `run_phases` (`oomtest.py:253,265`),
  never inside the phase loops (`stress_harness/phases.py`). Each round also blocks on
  an HTTP request that can run a long time at large context. So Stop is ignored until
  the entire current phase (e.g. 20 sustained rounds) completes.
- `stress_harness/` phases are shared with the external CLI (`stress_test.py` →
  `StressTestRunner`), so changes there must stay backward-compatible.

## A. Quick-mode checkbox

**UI (`test.py`)**
- Add `Checkbox("Quick mode", value=True, id="quick-mode")` in `compose()`, placed
  just above the "Run check" button.
- Default **checked** = quick (preserves today's behavior). Unchecking runs the full
  suite.
- Disable the checkbox while a run is in progress (kept in sync with the Run→Stop
  toggle in `_refresh_precondition` / `_start_run` / `_on_verdict`); re-enable on
  verdict.
- `_start_run` reads `self.query_one("#quick-mode", Checkbox).value` **on the UI
  thread** (never from the worker thread — existing thread-safety rule) and passes it
  to `_run_worker`.

**Core (`oomtest.py`)**
- `run_oom_check(..., quick: bool = True)` builds the env with
  `"QUICK": "1" if quick else "0"`. Default `True` preserves existing call sites/tests.

**Footer**
- Reword to drop the confusing `QUICK=1` hint; state that the checkbox controls
  quick vs. full thoroughness of this tab's check, and that `python stress_test.py`
  remains the comprehensive multi-phase suite.

## B. Responsive Stop (per-round cancel)

- `BasePhase.__init__` accepts `cancel: Callable[[], bool] = lambda: False`.
- `run_phases` adds `cancel` to the `kw` dict passed to every phase. The external
  `runner.py` constructs phases without `cancel`, so the default no-op keeps CLI
  behavior unchanged (**backward-compatible**).
- Each phase loop (`ramp`, `sustained`, `cold-start`, `defrag`, `boundary`) checks
  `if self.cancel(): break` at the top of each iteration and returns its partial
  result. `run_phases` already short-circuits between phases when `cancel()` is true.
- `run_oom_check`: after the run, if `cancel()` is true, return
  `OomTestResult("STOPPED", …)` so the verdict reads as stopped-by-user.
- Verdict rendering (`_on_verdict`): render `STOPPED` neutrally (not as a red
  failure).

Result: Stop latency drops from "rest of the current phase (minutes)" to roughly one
in-flight request. Mid-request interruption is explicitly **out of scope**.

## C. More live data

**Shared helper (`monitor.py`)**
- Extract `read_server_vram_kib(server: ServerInfo, runtime: str | None) -> int | None`
  that dispatches: native → `read_vram_kib(pid)`; container → `read_container_vram_kib(name, runtime)`;
  returns `None` when unreadable. Serve's `_poll_vram_and_health` is refactored to use
  it, removing the duplicated native-vs-container branch.

**Live VRAM gauge**
- Reuse the Serve tab's `_VramGauge`. The Test pane polls VRAM every 2s **while a run
  is active** via a UI-thread `set_interval` (independent of the worker thread). Reset
  the gauge when the run ends.

**Phase / progress indicator**
- A `Static` showing the current phase and round (e.g. "Sustained — round 7"),
  updated from the reporter's `start_phase` / `record_sample` via new thread-safe
  `Message` types posted from the worker (consistent with the existing
  `_PhaseRow` / `_Verdict` pattern).

**Peak / headroom summary**
- A `Static` the screen updates as samples arrive: tracks the running max peak VRAM and
  shows "Peak X.X / budget Y.Y GiB (headroom Z.Z)". Budget comes from `vram_warn_gb`,
  already passed to `record_sample`.

## Testing (TDD)

- **`oomtest`**: `run_oom_check` with `quick=False` builds a `StressConfig` with full
  rounds (`sustained_rounds=20`, `cold_rounds=8`, `defrag_cycles=10`); `quick=True`
  keeps reduced rounds. Cancellation: a phase given a `cancel` that returns True after
  N rounds stops at N (does not run the full loop); `run_oom_check` returns a `STOPPED`
  verdict when cancelled.
- **`monitor`**: `read_server_vram_kib` dispatches native → `read_vram_kib`,
  container → `read_container_vram_kib`, and surfaces `None` on unreadable.
- **`test_test_ui`**: checkbox present, default-checked, disabled during a run, and its
  value is passed as `quick` (monkeypatch `run_oom_check` to capture the kwarg); gauge /
  progress / peak widgets present and update on posted messages; `STOPPED` verdict
  renders neutrally.

## Out of scope (YAGNI)

- No persistence of the checkbox state across launches.
- No mid-request interruption of Stop.
- No per-parameter (individual round-count) controls — just the single quick/full
  toggle.
