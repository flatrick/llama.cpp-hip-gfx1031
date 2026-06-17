# Sudo-free Container Peak/Post VRAM in the OOM Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Test pane's OOM check report container peak/post VRAM without sudo, by reading fdinfo inside the container (via `docker exec`) instead of host `/proc/<pid>/fdinfo`, at a slower sample cadence to bound the per-tick `exec` overhead.

**Architecture:** `llamactl/core/oomtest.py::build_vram_monitor` currently resolves the container's *host* PID and reads host fdinfo — which is root-owned, so a non-root user gets permission-denied → peak/post show `n/a`. Switch the container reader to `monitor.read_container_vram_kib(container_name, runtime)` (already used by the dashboard gauge, reads fdinfo *inside* the container — no sudo). Because that reader runs a `docker exec` per call and `PeakVramSampler` polls every 200 ms, add a per-monitor `sample_interval_ms` and use a slower interval (1 s) for container mode. Native mode is unchanged (cheap direct read at 200 ms).

**Tech Stack:** Python 3.14, pytest (asyncio strict). Run tests with `python -m pytest`.

**Approach chosen by user:** exec-per-tick + slower cadence (over system-wide sysfs or a long-lived streaming reader).

**Context — current behaviour (do not break):**
- `build_vram_monitor(server, runtime, get_pid=get_container_pid)` (oomtest.py:49). Native: reader = `read_vram_kib(str(server.pid))`. Container: resolves host pid via `get_pid` then `read_vram_kib(host_pid)`.
- `read_container_vram_kib(container_name, runtime)` (monitor.py) returns summed KiB inside the container, or None. Already imported? No — only `read_vram_kib` is imported in oomtest.py; add the import.
- `PeakVramSampler(monitor, interval_ms=200)` (monitoring.py). Constructed in `stress_harness/phases.py` at two sites (`_sample_request` ~line 57 and `BoundaryPhase.run` ~line 287) as `PeakVramSampler(self.vram_monitor).start()`.
- `VramMonitor.__init__(self, reader, mode)` (monitoring.py). Also built by the external CLI via `VramMonitor.create(...)` — changes must be backward-compatible (new arg defaults).
- The verdict/`run_phases`/`build_vram_monitor(server, runtime)` call at oomtest.py:325 already uses the 2-arg form.

**DO NOT run this plan while a Test-pane run is in progress** (the suite and any manual smoke would contend with the live run). Execute once the user's current run is finished.

---

## Task 1: Per-monitor sample interval

**Files:**
- Modify: `stress_harness/monitoring.py` (`VramMonitor.__init__`)
- Modify: `stress_harness/phases.py` (two `PeakVramSampler(...)` constructions)
- Test: `tests/llamactl/test_oomtest.py` (VramMonitor attribute), `tests/llamactl/test_phases_cancel.py` (phases honour the interval)

- [ ] **Step 1: Write the failing tests**

Add to `tests/llamactl/test_oomtest.py`:

```python
def test_vram_monitor_default_sample_interval():
    mon = VramMonitor(lambda: 1.0, "test")
    assert mon.sample_interval_ms == 200


def test_vram_monitor_custom_sample_interval():
    mon = VramMonitor(lambda: 1.0, "test", sample_interval_ms=1000)
    assert mon.sample_interval_ms == 1000
```

Add to `tests/llamactl/test_phases_cancel.py` (reuse the existing stub classes `_Reporter`, `_Client`, `_Inspector`, `_PromptBuilder`, `_Monitor` already in that file; add this test and a small spy):

```python
def test_sample_request_uses_monitor_interval(monkeypatch):
    """The phase's PeakVramSampler must use the monitor's configured cadence."""
    import stress_harness.phases as phases_mod
    from stress_harness.monitoring import VramMonitor

    captured = {}

    class _SpySampler:
        def __init__(self, monitor, interval_ms=200):
            captured["interval_ms"] = interval_ms
        def start(self):
            return self
        def stop(self):
            return None

    monkeypatch.setattr(phases_mod, "PeakVramSampler", _SpySampler)

    monitor = VramMonitor(lambda: 1.0, "test", sample_interval_ms=1000)
    config = StressConfig(sustained_rounds=1)
    phase = SustainedPhase(
        config=config,
        client=_Client(),
        prompt_builder=_PromptBuilder(),
        vram_monitor=monitor,
        runtime_inspector=_Inspector(),
        runtime_info=RuntimeInfo(runtime=None, container_id=None, status_message="x"),
        reporter=_Reporter(),
    )
    phase.run(last_ok_tokens=1000)
    assert captured["interval_ms"] == 1000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_oomtest.py -k sample_interval tests/llamactl/test_phases_cancel.py::test_sample_request_uses_monitor_interval -q`
Expected: FAIL — `VramMonitor.__init__()` got an unexpected keyword argument `sample_interval_ms`; and the spy interval is 200, not 1000.

- [ ] **Step 3: Add `sample_interval_ms` to `VramMonitor`**

In `stress_harness/monitoring.py`, change `VramMonitor.__init__`:

```python
class VramMonitor:
    def __init__(self, reader, mode: str, sample_interval_ms: int = 200) -> None:
        self._reader = reader
        self.mode = mode
        self.sample_interval_ms = sample_interval_ms
```

- [ ] **Step 4: Wire phases to use it**

In `stress_harness/phases.py`, change BOTH occurrences of:

```python
        sampler = PeakVramSampler(self.vram_monitor).start()
```

to:

```python
        sampler = PeakVramSampler(
            self.vram_monitor, interval_ms=self.vram_monitor.sample_interval_ms
        ).start()
```

(There are two: in `_sample_request` and in `BoundaryPhase.run`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_oomtest.py -k sample_interval tests/llamactl/test_phases_cancel.py -q`
Expected: PASS. Then full suite: `python -m pytest tests/llamactl -q` — expect all pass (default interval 200 keeps the CLI path unchanged).

- [ ] **Step 6: Commit**

```bash
git add stress_harness/monitoring.py stress_harness/phases.py tests/llamactl/test_oomtest.py tests/llamactl/test_phases_cancel.py
git commit -m "feat: per-monitor VRAM sample interval (default 200ms)"
```

---

## Task 2: Container VRAM reader uses `docker exec` (sudo-free) at slower cadence

**Files:**
- Modify: `llamactl/core/oomtest.py` (`build_vram_monitor`; imports)
- Test: `tests/llamactl/test_oomtest.py` (replace the host-pid container tests)

- [ ] **Step 1: Write the failing tests**

In `tests/llamactl/test_oomtest.py`:

(a) Update the native test to drop the now-removed `get_pid` argument. Change `test_build_vram_monitor_native_uses_pid` so its call reads:

```python
    mon = build_vram_monitor(server, runtime=None)
```

(b) Replace `test_build_vram_monitor_container_resolves_pid_once` and `test_build_vram_monitor_returns_none_when_no_pid` entirely with these three tests:

```python
def test_build_vram_monitor_container_uses_exec_reader(monkeypatch):
    """Container mode reads fdinfo inside the container (no host pid, no sudo)."""
    monkeypatch.setattr(
        "llamactl.core.oomtest.read_container_vram_kib",
        lambda name, runtime, **kw: 1024 * 1024 if name == "llamactl-m" else None,  # 1 GiB
    )
    server = ServerInfo(model_id="m", backend="rocm", preset="", mode="container",
                        host="0.0.0.0", port=8080, started_at="",
                        container_name="llamactl-m")
    mon = build_vram_monitor(server, runtime="podman")
    assert abs(mon.read() - 1.0) < 1e-9


def test_build_vram_monitor_container_uses_slower_interval(monkeypatch):
    """Container mode samples less often (exec per tick is heavier than a file read)."""
    monkeypatch.setattr(
        "llamactl.core.oomtest.read_container_vram_kib",
        lambda name, runtime, **kw: 0,
    )
    server = ServerInfo(model_id="m", backend="rocm", preset="", mode="container",
                        host="0.0.0.0", port=8080, started_at="",
                        container_name="llamactl-m")
    mon = build_vram_monitor(server, runtime="podman")
    assert mon.sample_interval_ms == 1000


def test_build_vram_monitor_container_none_without_runtime():
    server = ServerInfo(model_id="m", backend="rocm", preset="", mode="container",
                        host="0.0.0.0", port=8080, started_at="",
                        container_name="llamactl-m")
    # find_runtime() will be attempted; force None by leaving runtime unset and
    # patching find_runtime to None.
    import llamactl.core.oomtest as oom
    orig = oom.find_runtime
    oom.find_runtime = lambda: None
    try:
        mon = build_vram_monitor(server, runtime=None)
        assert mon.read() is None
    finally:
        oom.find_runtime = orig
```

(Leave `test_build_vram_monitor_native_no_pid_returns_none` as-is — it already calls `build_vram_monitor(server, runtime=None)`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_oomtest.py -k build_vram_monitor -q`
Expected: FAIL — container tests still hit the host-pid path / `read_container_vram_kib` not imported / `get_pid` removed errors.

- [ ] **Step 3: Implement**

In `llamactl/core/oomtest.py`:

(a) Extend the monitor import (line 30):

```python
from llamactl.core.monitor import read_container_vram_kib, read_vram_kib
```

(b) Remove the now-unused `get_container_pid` from the runtime import (line 31) if it is no longer referenced anywhere else in the file (it is only used by `build_vram_monitor`'s default). Change:

```python
from llamactl.core.runtime import find_runtime, get_container_pid
```
to:
```python
from llamactl.core.runtime import find_runtime
```

(c) Add a module constant near the top of the module (after imports):

```python
# Container VRAM is read via `docker exec` (per tick). That is far heavier than a
# native /proc read, so sample it less often than the 200ms native default.
CONTAINER_VRAM_SAMPLE_INTERVAL_MS = 1000
```

(d) Replace the whole `build_vram_monitor` function with:

```python
def build_vram_monitor(
    server: ServerInfo,
    runtime: str | None,
) -> VramMonitor:
    """A harness VramMonitor whose reader reports the managed server's VRAM in GiB.

    Native: read the server's own /proc/<pid>/fdinfo (cheap, 200ms cadence).
    Container: read fdinfo *inside* the container via `exec` (no host sudo). That
    exec is heavy, so use a slower sample cadence.
    """
    if server.mode == "native":
        pid = str(server.pid) if server.pid else None
        mode = f"per-process native (PID: {pid or 'unknown'})"

        def _reader() -> float | None:
            if pid is None:
                return None
            kib = read_vram_kib(pid)
            return kib / 1024 ** 2 if kib else None

        return VramMonitor(_reader, mode)

    rt = runtime or find_runtime()
    cname = server.container_name
    mode = f"per-container exec (container: {cname or 'unknown'})"

    def _reader() -> float | None:
        if not (rt and cname):
            return None
        kib = read_container_vram_kib(cname, rt)
        # None = unreadable, 0 = no DRM memory; both mean "no usable reading".
        return kib / 1024 ** 2 if kib else None

    return VramMonitor(
        _reader, mode, sample_interval_ms=CONTAINER_VRAM_SAMPLE_INTERVAL_MS
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_oomtest.py -k build_vram_monitor -q`
Expected: PASS. Then the whole oomtest file `python -m pytest tests/llamactl/test_oomtest.py -q`, then the full suite `python -m pytest tests/llamactl -q` — expect all pass.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "feat: OOM test reads container VRAM via exec (sudo-free) at slower cadence"
```

---

## Final verification

- [ ] Full suite: `python -m pytest tests/llamactl -q` — expect all green.
- [ ] Manual smoke (after the user's current run, needs a running container): start a model container, run the Test pane check → the peak and post columns now show real GiB values (not `n/a`) without sudo; native mode still works.
- [ ] Note any follow-up: if 1 s cadence proves too coarse for short requests, consider the long-lived streaming-exec reader (deferred) instead of lowering the interval (which would raise exec overhead).

## Out of scope (YAGNI)

- System-wide sysfs fallback (different signal — total GPU, not the container).
- Long-lived streaming `exec` reader (more complex; revisit only if exec-per-tick overhead is a problem).
- Making the cadence user-configurable (a constant is enough).
