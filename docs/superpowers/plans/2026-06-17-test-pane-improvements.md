# Test-pane Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a quick/full mode checkbox, make Stop responsive within ~one request, and show live VRAM + phase progress + peak/headroom in the llamactl Test pane.

**Architecture:** Expose the hardcoded `QUICK=1` as a `run_oom_check(quick=...)` parameter driven by a UI checkbox. Thread the existing `cancel` callable into each stress phase so loops break per-round (backward-compatible default for the external CLI). Reuse the Serve tab's `_VramGauge` via a new shared `monitor.read_server_vram_kib` helper; the Test screen polls it on a UI-thread interval during a run and surfaces phase/peak via thread-safe messages posted from the worker.

**Tech Stack:** Python 3.14, Textual 8.2.7, pytest (asyncio strict).

**Spec:** `docs/superpowers/specs/2026-06-17-test-pane-improvements-design.md`

**Branch:** `fix/dashboard-model-ui` (already pushed; continue here).

---

## File Structure

- `llamactl/core/oomtest.py` — add `quick` param to `run_oom_check`; thread `cancel` into phases; return `STOPPED` verdict on cancel.
- `stress_harness/phases.py` — `BasePhase` accepts `cancel`; each phase loop checks it per iteration.
- `llamactl/core/monitor.py` — add `read_server_vram_kib(server, runtime)` shared helper.
- `llamactl/ui/screens/serve.py` — refactor `_poll_vram_and_health` to use the new helper.
- `llamactl/ui/screens/test.py` — checkbox, live VRAM gauge, progress + peak widgets, message handlers, interval poll, pass `quick`/`cancel` through.
- Tests: `tests/llamactl/test_oomtest.py`, `tests/llamactl/test_monitor.py`, `tests/llamactl/test_test_ui.py`, plus a phase-cancel test in `tests/llamactl/` (new `test_phases_cancel.py`).

Run the whole suite with: `python -m pytest tests/llamactl -q`

---

## Task 1: `run_oom_check` honours a `quick` parameter

**Files:**
- Modify: `llamactl/core/oomtest.py` (the `run_oom_check` signature + env build, ~lines 284-299)
- Test: `tests/llamactl/test_oomtest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_oomtest.py`:

```python
def test_run_oom_check_quick_flag_controls_rounds(monkeypatch):
    """quick=True keeps reduced rounds; quick=False uses full rounds.

    Capture the StressConfig the run builds by stubbing the network/inspection so
    run_oom_check stops right after config creation.
    """
    import llamactl.core.oomtest as oom
    from llamactl.core.lifecycle import ServerInfo

    captured = {}

    class _StubClient:
        def __init__(self, config):
            captured["config"] = config
        def server_healthy(self):
            return False  # short-circuits run_oom_check right after config build

    monkeypatch.setattr(oom, "LlamaServerClient", _StubClient)

    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="", pid=1234,
    )

    class _Cfg:
        vram_budget_gb = 11.0

    oom.run_oom_check(server, _NullReporter(), global_cfg=_Cfg(), quick=True)
    assert captured["config"].sustained_rounds == 3
    assert captured["config"].cold_rounds == 1

    oom.run_oom_check(server, _NullReporter(), global_cfg=_Cfg(), quick=False)
    assert captured["config"].sustained_rounds == 20
    assert captured["config"].cold_rounds == 8
    assert captured["config"].defrag_cycles == 10
```

If `_NullReporter` does not already exist in the test module, add it near the top:

```python
class _NullReporter:
    def start_run(self, result): pass
    def start_phase(self, phase): pass
    def record_sample(self, phase_key, sample, warn_at): pass
    def finish_phase(self, phase): pass
    def finish_run(self, result): pass
    def error(self, message): pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llamactl/test_oomtest.py::test_run_oom_check_quick_flag_controls_rounds -q`
Expected: FAIL — `run_oom_check()` got an unexpected keyword argument `quick`.

- [ ] **Step 3: Add the `quick` parameter**

In `llamactl/core/oomtest.py`, change the signature and env build:

```python
def run_oom_check(
    server: ServerInfo,
    reporter: Reporter,
    *,
    global_cfg: GlobalConfig,
    cancel: Callable[[], bool] = lambda: False,
    phase_set: PhaseSet = DEFAULT_PHASES,
    quick: bool = True,
) -> OomTestResult:
    """Run the OOM boundary check against the running server.

    quick=True (default) runs the reduced suite; quick=False runs full rounds.
    """
    api_url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
    config = StressConfig.from_env({
        "QUICK": "1" if quick else "0",
        "API_URL": api_url,
        "VRAM_WARN_GB": str(global_cfg.vram_budget_gb),
        # NB: do NOT set CTX_SIZE — let the harness auto-detect from /slots.
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llamactl/test_oomtest.py::test_run_oom_check_quick_flag_controls_rounds -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "feat: add quick parameter to run_oom_check"
```

---

## Task 2: Phases honour a `cancel` callable per round

**Files:**
- Modify: `stress_harness/phases.py` (`BasePhase.__init__`; loops in `RampPhase`, `SustainedPhase`, `ColdStartPhase`, `DefragPhase`, `BoundaryPhase`)
- Test: `tests/llamactl/test_phases_cancel.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/llamactl/test_phases_cancel.py`:

```python
from __future__ import annotations

from stress_harness.config import StressConfig
from stress_harness.models import PhaseSample, RuntimeInfo
from stress_harness.phases import SustainedPhase


class _Reporter:
    def start_phase(self, phase): pass
    def record_sample(self, phase_key, sample, warn_at): pass


class _Client:
    def __init__(self):
        self.calls = 0

    def send_request(self, **kwargs):
        self.calls += 1
        # Minimal request stand-in; phases only read it back via PhaseSample.
        return object()


class _Inspector:
    def start_log_reader(self, runtime_info):
        return None


class _PromptBuilder:
    def build(self, target, prefix=""):
        return "x"


class _Monitor:
    def read(self):
        return 1.0


def test_sustained_phase_stops_when_cancelled():
    """With cancel() True after 2 rounds, the phase must not run all 20 rounds."""
    config = StressConfig(sustained_rounds=20)
    client = _Client()
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] > 2   # allow 2 rounds, then cancel

    phase = SustainedPhase(
        config=config,
        client=client,
        prompt_builder=_PromptBuilder(),
        vram_monitor=_Monitor(),
        runtime_inspector=_Inspector(),
        runtime_info=RuntimeInfo(runtime=None, container_id=None, status_message="x"),
        reporter=_Reporter(),
        cancel=cancel,
    )
    phase.run(last_ok_tokens=1000)
    assert client.calls <= 2  # stopped early, not 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llamactl/test_phases_cancel.py -q`
Expected: FAIL — `SustainedPhase.__init__()` got an unexpected keyword argument `cancel`.

- [ ] **Step 3: Add `cancel` to `BasePhase` and check it in loops**

In `stress_harness/phases.py`, add the import and the constructor param:

```python
from typing import Callable
```

Update `BasePhase.__init__` to accept and store `cancel` (append it as the last param with a default so existing keyword construction in `runner.py` is unaffected):

```python
    def __init__(
        self,
        config: StressConfig,
        client: LlamaServerClient,
        prompt_builder: PromptBuilder,
        vram_monitor: VramMonitor,
        runtime_inspector: ContainerRuntimeInspector,
        runtime_info: RuntimeInfo,
        reporter: ConsoleReporter,
        cancel: Callable[[], bool] = lambda: False,
    ) -> None:
        self.config = config
        self.client = client
        self.prompt_builder = prompt_builder
        self.vram_monitor = vram_monitor
        self.runtime_inspector = runtime_inspector
        self.runtime_info = runtime_info
        self.reporter = reporter
        self.cancel = cancel
```

Add `if self.cancel(): break` (or `return`) at the **top** of each loop body:

`RampPhase.run` — change `for target in steps:` body start to:

```python
        for target in steps:
            if self.cancel():
                break
            prompt = self.prompt_builder.build(target)
```

`SustainedPhase.run` — change `for index in range(1, self.config.sustained_rounds + 1):` body start to:

```python
        for index in range(1, self.config.sustained_rounds + 1):
            if self.cancel():
                break
            sample = self._sample_request(log_reader, str(index), prompt)
```

`ColdStartPhase.run` — change `for index in range(1, self.config.cold_rounds + 1):` body start to:

```python
        for index in range(1, self.config.cold_rounds + 1):
            if self.cancel():
                break
            prompt = self.prompt_builder.build(last_ok_tokens, prefix=f"[cold-start round {index}]")
```

`DefragPhase.run` — change `for cycle in range(1, self.config.defrag_cycles + 1):` body start to:

```python
        for cycle in range(1, self.config.defrag_cycles + 1):
            if self.cancel():
                break
            fill_prompt = self.prompt_builder.build(last_ok_tokens, prefix=f"[defrag cycle {cycle} fill]")
```

`BoundaryPhase.run` — add right after `self.reporter.start_phase(result)`:

```python
        if self.cancel():
            return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llamactl/test_phases_cancel.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm no regression in the external harness path**

Run: `python -m pytest tests/llamactl -q`
Expected: all pass (the external CLI constructs phases without `cancel`, so the default no-op preserves behaviour).

- [ ] **Step 6: Commit**

```bash
git add stress_harness/phases.py tests/llamactl/test_phases_cancel.py
git commit -m "feat: stress phases honour a per-round cancel callable"
```

---

## Task 3: `run_phases`/`run_oom_check` wire `cancel` in and report STOPPED

**Files:**
- Modify: `llamactl/core/oomtest.py` (`run_phases` `kw` dict; `run_oom_check` post-run cancel check)
- Test: `tests/llamactl/test_oomtest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_oomtest.py`:

```python
def test_run_oom_check_returns_stopped_when_cancelled(monkeypatch):
    """If cancel() is true after phases run, the verdict is STOPPED."""
    import llamactl.core.oomtest as oom
    from llamactl.core.lifecycle import ServerInfo

    # Healthy server so we get past the early return.
    class _Client:
        def __init__(self, config): pass
        def server_healthy(self): return True
        def server_ctx_size(self): return 4096

    monkeypatch.setattr(oom, "LlamaServerClient", _Client)
    # Stub run_phases so we don't hit the network; report no phases, no peak.
    monkeypatch.setattr(oom, "run_phases", lambda **kw: ([], None, False))
    # Native path avoids container inspection.
    monkeypatch.setattr(oom, "build_vram_monitor", lambda server, runtime: _StubMonitor())

    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="", pid=1234, log_path=None,
    )

    class _Cfg:
        vram_budget_gb = 11.0

    result = oom.run_oom_check(
        server, _NullReporter(), global_cfg=_Cfg(), cancel=lambda: True
    )
    assert result.verdict == "STOPPED"
```

Add the monitor stub near the other test helpers:

```python
class _StubMonitor:
    def read(self):
        return None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llamactl/test_oomtest.py::test_run_oom_check_returns_stopped_when_cancelled -q`
Expected: FAIL — verdict is not `"STOPPED"` (classify_verdict returns OK-degraded for empty phases).

- [ ] **Step 3: Wire cancel into run_phases kw and add the STOPPED check**

In `llamactl/core/oomtest.py`, add `cancel` to the `kw` dict inside `run_phases`:

```python
    kw = dict(
        config=config,
        client=client,
        prompt_builder=prompt_builder,
        vram_monitor=vram_monitor,
        runtime_inspector=runtime_inspector,
        runtime_info=runtime_info,
        reporter=reporter,
        cancel=cancel,
    )
```

In `run_oom_check`, replace the final `return classify_verdict(...)` (currently the last line, ~line 340) with:

```python
    if cancel():
        return OomTestResult(
            "STOPPED", peak, None, None, "Stopped by user."
        )
    return classify_verdict(phases, peak, global_cfg.vram_budget_gb, vram_available)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llamactl/test_oomtest.py::test_run_oom_check_returns_stopped_when_cancelled -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "feat: run_oom_check threads cancel into phases and reports STOPPED"
```

---

## Task 4: `read_server_vram_kib` shared helper + Serve refactor

**Files:**
- Modify: `llamactl/core/monitor.py` (add helper; import `ServerInfo`)
- Modify: `llamactl/ui/screens/serve.py` (`_poll_vram_and_health` uses the helper)
- Test: `tests/llamactl/test_monitor.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_monitor.py`:

```python
def test_read_server_vram_kib_dispatches_native(monkeypatch):
    from llamactl.core.lifecycle import ServerInfo
    import llamactl.core.monitor as mon

    monkeypatch.setattr(mon, "read_vram_kib", lambda pid, **kw: 12345 if pid == "1234" else 0)
    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="", pid=1234,
    )
    assert mon.read_server_vram_kib(server, None) == 12345


def test_read_server_vram_kib_dispatches_container(monkeypatch):
    from llamactl.core.lifecycle import ServerInfo
    import llamactl.core.monitor as mon

    monkeypatch.setattr(
        mon, "read_container_vram_kib",
        lambda name, runtime, **kw: 999 if name == "llamactl-m" else None,
    )
    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="container",
        host="0.0.0.0", port=8080, started_at="", container_name="llamactl-m",
    )
    assert mon.read_server_vram_kib(server, "/usr/bin/docker") == 999


def test_read_server_vram_kib_none_when_unresolvable():
    from llamactl.core.lifecycle import ServerInfo
    import llamactl.core.monitor as mon

    # container mode but no runtime → None
    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="container",
        host="0.0.0.0", port=8080, started_at="", container_name="llamactl-m",
    )
    assert mon.read_server_vram_kib(server, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llamactl/test_monitor.py -k read_server_vram_kib -q`
Expected: FAIL — `module 'llamactl.core.monitor' has no attribute 'read_server_vram_kib'`.

- [ ] **Step 3: Add the helper**

In `llamactl/core/monitor.py`, extend the lifecycle import:

```python
from llamactl.core.lifecycle import ServerInfo, ServerState
```

Add the function (place after `read_container_vram_kib`):

```python
def read_server_vram_kib(server: ServerInfo, runtime: str | None) -> int | None:
    """VRAM (KiB) for a managed server, or None if it cannot be read.

    Native: read the server's own /proc/<pid>/fdinfo.
    Container: read fdinfo inside the container via `exec` (needs runtime).
    """
    if server.mode == "native":
        if server.pid is None:
            return None
        return read_vram_kib(str(server.pid))
    if server.container_name and runtime is not None:
        return read_container_vram_kib(server.container_name, runtime)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llamactl/test_monitor.py -k read_server_vram_kib -q`
Expected: PASS

- [ ] **Step 5: Refactor Serve's poll to use the helper**

In `llamactl/ui/screens/serve.py`, inside `_poll_vram_and_health`, change the import block:

```python
            from llamactl.core.monitor import check_health, read_server_vram_kib
```

Replace the whole VRAM-resolution block (the `vram_kib: int | None = None` branch that calls `read_vram_kib` / `read_container_vram_kib`) with:

```python
            from llamactl.core.runtime import find_runtime
            rt = find_runtime() if mode == "container" else None
            vram_kib = await asyncio.to_thread(
                read_server_vram_kib, self._server_info, rt
            )
            gauge = self.query_one(_VramGauge)
            gauge.budget_kib = int(app._global_cfg.vram_budget_gb * 1024 * 1024)
            gauge.vram_kib = vram_kib
```

- [ ] **Step 6: Run the Serve UI + monitor suites**

Run: `python -m pytest tests/llamactl/test_serve_ui.py tests/llamactl/test_monitor.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add llamactl/core/monitor.py llamactl/ui/screens/serve.py tests/llamactl/test_monitor.py
git commit -m "refactor: extract read_server_vram_kib and use it in the Serve poll"
```

---

## Task 5: Quick-mode checkbox in the Test pane

**Files:**
- Modify: `llamactl/ui/screens/test.py` (import `Checkbox`; `compose`; `_refresh_precondition`; `_start_run`; `_run_worker`; `_on_verdict`)
- Test: `tests/llamactl/test_test_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_test_ui.py`:

```python
@pytest.mark.asyncio
async def test_quick_checkbox_default_checked_and_passed(tmp_path, monkeypatch):
    """The Quick-mode checkbox defaults to checked and its value is passed as quick."""
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.ui.screens.test as test_mod
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import Checkbox, TabbedContent

    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="", pid=1234,
    )
    monkeypatch.setattr(test_mod, "find_running", lambda *_a, **_kw: server)

    captured = {}
    def fake_run_oom_check(srv, reporter, *, global_cfg, cancel, quick=True):
        captured["quick"] = quick
        from llamactl.core.oomtest import OomTestResult
        return OomTestResult("OK", 1.0, 100, None, "done")
    monkeypatch.setattr(test_mod, "run_oom_check", fake_run_oom_check)

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        cb = screen.query_one("#quick-mode", Checkbox)
        assert cb.value is True  # default checked = quick

        # Uncheck → full run.
        cb.value = False
        await pilot.pause()
        screen._start_run(screen.query_one("#btn-run-test"))
        # Worker runs on a thread; give it a moment to call run_oom_check.
        for _ in range(50):
            if "quick" in captured:
                break
            await pilot.pause(0.02)
        assert captured["quick"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llamactl/test_test_ui.py::test_quick_checkbox_default_checked_and_passed -q`
Expected: FAIL — no widget with id `quick-mode`.

- [ ] **Step 3: Add the checkbox and thread `quick` through**

In `llamactl/ui/screens/test.py`, extend the widgets import:

```python
from textual.widgets import Button, Checkbox, DataTable, Label, Static
```

In `compose`, add the checkbox just above the Run button:

```python
        yield Static("", id="test-precondition")
        yield Checkbox("Quick mode (faster, fewer rounds)", value=True, id="quick-mode")
        yield Button("Run check", id="btn-run-test", disabled=True)
```

In `_refresh_precondition`, keep the checkbox disabled state in sync — after the `btn.disabled = self._test_active` line in the `else` branch, add:

```python
            try:
                self.query_one("#quick-mode", Checkbox).disabled = self._test_active
            except NoMatches:
                pass
```

In `_start_run`, read the checkbox on the UI thread and pass it to the worker. Change the worker dispatch:

```python
        button.label = "Stop"
        quick = True
        try:
            quick = self.query_one("#quick-mode", Checkbox).value
        except NoMatches:
            pass
        self.query_one("#quick-mode", Checkbox).disabled = True
        self.run_worker(
            lambda: self._run_worker(server, global_cfg, quick),
            thread=True,
            exclusive=True,
            group="oom-test",
        )
```

Update `_run_worker` to accept and forward `quick`:

```python
    def _run_worker(self, server: ServerInfo, global_cfg: GlobalConfig, quick: bool) -> None:
        reporter = TextualReporter(self)
        try:
            result = run_oom_check(
                server,
                reporter,
                global_cfg=global_cfg,
                cancel=self._cancel_evt.is_set,
                quick=quick,
            )
        except Exception as exc:
            result = OomTestResult(
                "FAIL", None, None, None, f"Test crashed: {exc!r}"
            )
        self.post_message(_Verdict(result))
```

In `_on_verdict`, re-enable the checkbox after the run (after `self._test_active = False`):

```python
        self._test_active = False
        try:
            self.query_one("#quick-mode", Checkbox).disabled = False
        except NoMatches:
            pass
        self._refresh_precondition()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llamactl/test_test_ui.py::test_quick_checkbox_default_checked_and_passed -q`
Expected: PASS

- [ ] **Step 5: Update the footer text**

In `compose`, replace the `id="test-footer"` Static content with:

```python
        yield Static(
            "[dim]Quick mode runs a reduced subset; uncheck it for a fuller pass "
            "(more sustained/cold/defrag rounds — takes longer). For the complete "
            "multi-phase suite run `python stress_test.py`.[/dim]",
            id="test-footer",
        )
```

- [ ] **Step 6: Commit**

```bash
git add llamactl/ui/screens/test.py tests/llamactl/test_test_ui.py
git commit -m "feat: add Quick-mode checkbox to the Test pane"
```

---

## Task 6: Live VRAM gauge in the Test pane

**Files:**
- Modify: `llamactl/ui/screens/test.py` (import `_VramGauge`; `compose`; `_start_run`; `_on_verdict`; add `_poll_test_vram`)
- Test: `tests/llamactl/test_test_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_test_ui.py`:

```python
@pytest.mark.asyncio
async def test_test_pane_live_vram_gauge_updates(tmp_path, monkeypatch):
    """The Test pane's VRAM gauge updates from read_server_vram_kib during a run."""
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.ui.screens.test as test_mod
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.screens.test import TestScreen
    from llamactl.ui.screens.serve import _VramGauge
    from textual.widgets import TabbedContent

    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="", pid=1234,
    )
    monkeypatch.setattr(test_mod, "find_running", lambda *_a, **_kw: server)
    monkeypatch.setattr(test_mod, "read_server_vram_kib", lambda srv, rt: 3_000_000)

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        screen._test_active = True
        screen._poll_test_vram()
        await pilot.pause()
        gauge = screen.query_one("#test-vram", _VramGauge)
        assert gauge.vram_kib == 3_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llamactl/test_test_ui.py::test_test_pane_live_vram_gauge_updates -q`
Expected: FAIL — no widget `#test-vram` / no `_poll_test_vram`.

- [ ] **Step 3: Add the gauge, poll method, and interval lifecycle**

In `llamactl/ui/screens/test.py`, add imports:

```python
from llamactl.core.lifecycle import ServerInfo, find_running
from llamactl.core.monitor import read_server_vram_kib
from llamactl.core.runtime import find_runtime
from llamactl.ui.screens.serve import _VramGauge
```

In `compose`, add the gauge (just below the precondition Static):

```python
        yield _VramGauge(id="test-vram")
```

In `__init__`, add a handle for the poll timer:

```python
        self._vram_timer = None
```

Add the poll method (UI thread):

```python
    def _poll_test_vram(self) -> None:
        """Update the live VRAM gauge while a run is active. UI-thread only."""
        if not self._test_active:
            return
        server = self._server()
        if server is None:
            return
        rt = find_runtime() if server.mode == "container" else None
        kib = read_server_vram_kib(server, rt)
        try:
            gauge = self.query_one("#test-vram", _VramGauge)
            gauge.budget_kib = int(self.app._global_cfg.vram_budget_gb * 1024 * 1024)
            gauge.vram_kib = kib
        except NoMatches:
            pass
```

In `_start_run`, after setting `self._test_active = True`, start the interval:

```python
        self._test_active = True
        if self._vram_timer is None:
            self._vram_timer = self.set_interval(2.0, self._poll_test_vram)
        self._poll_test_vram()  # immediate first reading
```

In `_on_verdict`, after `self._test_active = False`, stop polling and reset the gauge:

```python
        self._test_active = False
        if self._vram_timer is not None:
            self._vram_timer.stop()
            self._vram_timer = None
        try:
            self.query_one("#test-vram", _VramGauge).vram_kib = 0
        except NoMatches:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llamactl/test_test_ui.py::test_test_pane_live_vram_gauge_updates -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/test.py tests/llamactl/test_test_ui.py
git commit -m "feat: live VRAM gauge in the Test pane"
```

---

## Task 7: Phase/progress indicator and peak/headroom summary

**Files:**
- Modify: `llamactl/ui/screens/test.py` (new messages `_Progress`, `_PeakStat`; `TextualReporter.start_phase`/`record_sample`; `compose`; handlers; reset state in `_start_run`; verdict colour for STOPPED)
- Test: `tests/llamactl/test_test_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_test_ui.py`:

```python
@pytest.mark.asyncio
async def test_progress_and_peak_update_from_messages(tmp_path, monkeypatch):
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.ui.screens.test as test_mod
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.screens.test import TestScreen, _Progress, _PeakStat
    from textual.widgets import Static, TabbedContent

    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="", pid=1234,
    )
    monkeypatch.setattr(test_mod, "find_running", lambda *_a, **_kw: server)

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)

        screen.post_message(_Progress("sustained — round 7"))
        screen.post_message(_PeakStat(9.5, 11.0))
        await pilot.pause()

        assert "round 7" in str(screen.query_one("#test-progress", Static).render())
        peak_text = str(screen.query_one("#test-peak", Static).render())
        assert "9.5" in peak_text and "11.0" in peak_text


@pytest.mark.asyncio
async def test_stopped_verdict_renders_neutrally(tmp_path):
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen, _Verdict
    from llamactl.core.oomtest import OomTestResult
    from textual.widgets import Static, TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        screen.post_message(_Verdict(OomTestResult("STOPPED", None, None, None, "Stopped by user.")))
        await pilot.pause()
        text = str(screen.query_one("#verdict", Static).render())
        assert "STOPPED" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llamactl/test_test_ui.py -k "progress_and_peak or stopped_verdict" -q`
Expected: FAIL — `_Progress`/`_PeakStat` not defined; no `#test-progress`/`#test-peak`.

- [ ] **Step 3: Add messages, widgets, reporter posts, and handlers**

In `llamactl/ui/screens/test.py`, add two message classes near `_PhaseRow`/`_Verdict`:

```python
class _Progress(Message):
    """Current phase/round line for the progress indicator."""
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class _PeakStat(Message):
    """A peak-VRAM observation (gb) with the budget (gb) for headroom display."""
    def __init__(self, peak_gb: float | None, budget_gb: float | None) -> None:
        self.peak_gb = peak_gb
        self.budget_gb = budget_gb
        super().__init__()
```

In `TextualReporter.start_phase`, post a progress update (keep the existing divider row):

```python
    def start_phase(self, phase) -> None:
        self._screen.post_message(_PhaseRow(("", f"── {phase.title} ──", "", "", "", "", "")))
        self._screen.post_message(_Progress(phase.title))
```

In `TextualReporter.record_sample`, after building/posting the row, also post progress + peak:

```python
        self._screen.post_message(_Progress(f"{phase_key} — {sample.label}"))
        self._screen.post_message(_PeakStat(sample.peak_vram_gb, warn_at))
```

In `compose`, add two Statics (below the verdict Static):

```python
        yield Static("", id="test-progress")
        yield Static("", id="test-peak")
```

In `__init__`, track the running peak:

```python
        self._peak_gb: float | None = None
```

Add handlers:

```python
    @on(_Progress)
    def _on_progress(self, message: _Progress) -> None:
        try:
            self.query_one("#test-progress", Static).update(message.text)
        except NoMatches:
            pass

    @on(_PeakStat)
    def _on_peak(self, message: _PeakStat) -> None:
        if message.peak_gb is not None:
            if self._peak_gb is None or message.peak_gb > self._peak_gb:
                self._peak_gb = message.peak_gb
        if self._peak_gb is None:
            return
        budget = message.budget_gb
        try:
            label = self.query_one("#test-peak", Static)
        except NoMatches:
            return
        if budget is not None:
            headroom = budget - self._peak_gb
            label.update(f"Peak {self._peak_gb:.1f} / budget {budget:.1f} GiB (headroom {headroom:.1f})")
        else:
            label.update(f"Peak {self._peak_gb:.1f} GiB")
```

In `_start_run`, reset the running peak and clear the statics (alongside the existing table clear):

```python
        self._peak_gb = None
        try:
            self.query_one("#test-progress", Static).update("")
            self.query_one("#test-peak", Static).update("")
        except NoMatches:
            pass
```

In `_on_verdict`, render `STOPPED` neutrally — update the colour selection:

```python
        if result.verdict.startswith("OK"):
            colour = "green"
        elif result.verdict in ("WARN", "STOPPED"):
            colour = "yellow"
        else:
            colour = "red"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llamactl/test_test_ui.py -k "progress_and_peak or stopped_verdict" -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/llamactl -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add llamactl/ui/screens/test.py tests/llamactl/test_test_ui.py
git commit -m "feat: phase progress and peak/headroom summary in the Test pane"
```

---

## Final verification

- [ ] Run the whole suite: `python -m pytest tests/llamactl -q` — expect all green.
- [ ] Manual smoke (optional, needs a running model): launch llamactl, start a container, open Test tab → checkbox visible & checked, VRAM gauge populates, run shows phase progress + peak; press Stop mid-run → stops within ~one request and shows a neutral "STOPPED" verdict; uncheck Quick → fuller run.
- [ ] Push: `git push` (upstream already set to `origin/fix/dashboard-model-ui`).
