# llamactl Phase 5 — Code-Review Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the fixes from the multi-agent code review of the Phase 5 "Test tab" merge (`beb01cd..3ee4bbf`): two blockers (falsely-low VRAM estimate; worker thread leak on exit) plus the convergent cleanups (dead code, bare-except, input validation, type annotations, markup escaping).

**Architecture:** All changes are localized to three already-merged modules — `llamactl/core/estimate.py`, `llamactl/core/oomtest.py`, `llamactl/ui/screens/test.py` — plus their existing test files under `tests/llamactl/`. No new modules, no public-API renames. Cancellation moves from a plain `bool` to a `threading.Event`; the estimate gains a "missing KV metadata → unavailable" guard; `resolve_gguf_path` validates its spec; reporter params get a `Reporter` Protocol.

**Tech Stack:** Python 3.14, Textual, pytest (`asyncio_mode = "strict"`), `stress_harness` (read-only), `rich.markup` (already a Textual dependency).

**Review source:** Four-aspect review (correctness, concurrency, security, code-quality). Findings referenced inline per task.

**Test command (used throughout):**
```bash
python -m pytest tests/llamactl/ -q
```

---

## File Structure

| File | Change | Responsibility after change |
|------|--------|------------------------------|
| `llamactl/core/estimate.py` | modify | `estimate_vram` returns `None` when KV metadata is missing; `resolve_gguf_path` validates the `org/repo:QUANT` spec and logs which pattern matched |
| `llamactl/core/oomtest.py` | modify | Add `Reporter` Protocol + type annotations; collapse the four duplicated return-tuples in `run_phases` into one closure; drop the dead post-loop `cancel()` check |
| `llamactl/ui/screens/test.py` | modify | `threading.Event` cancellation; attributes initialized in `__init__`; `on_unmount` cancels the worker group; `except NoMatches` instead of bare `except`; verdict detail escaped before markup interpolation |
| `tests/llamactl/test_estimate.py` | modify | Cover missing-KV → `None`; spec-validation rejections |
| `tests/llamactl/test_oomtest.py` | modify | Assert `run_phases` return-tuple parity after refactor (existing tests already cover orchestration) |
| `tests/llamactl/test_test_ui.py` | modify | Cover cancel-on-unmount; verdict-markup escaping |

**Reference facts (verified against the merged code — use these exact signatures):**

- `estimate.py`: `model_params_from_gguf(path) -> dict` with keys `arch, block_count, kv_layers, kv_heads, head_dim, weight_gb`; `compute_estimate(*, params, ctx_size, cache_type_k, cache_type_v, batch_size) -> Estimate`; `resolve_gguf_path(hf_spec: str, global_cfg: GlobalConfig) -> Path | None`; `estimate_vram(model, resolved_settings, global_cfg) -> Estimate | None`. Module logger is `_log = logging.getLogger(__name__)`.
- `oomtest.py`: `run_phases(*, config_steps, phase_set, cancel, reporter, client, prompt_builder, vram_monitor, runtime_inspector, runtime_info, config=None) -> tuple[list[PhaseResult], float | None, bool]`; `run_oom_check(server, reporter, *, global_cfg, cancel=..., phase_set=DEFAULT_PHASES) -> OomTestResult`. Reporter duck-typed surface: `start_run(result)`, `start_phase(phase)`, `record_sample(phase_key, sample, warn_at)`, `finish_phase(phase)`, `finish_run(result)`, `error(message)`.
- `test.py`: `TestScreen(Widget)`; cancellation today is `self._cancelled: bool` set in `on_mount`, read via `cancel=lambda: self._cancelled`; worker launched with `run_worker(..., thread=True, exclusive=True, group="oom-test")`; verdict rendered as `self.query_one("#verdict", Static).update(f"[{colour} bold]{result.verdict}[/{colour} bold]  {result.detail}")`.
- Test fixtures: `tests/llamactl/test_test_ui.py` has `_repo_with_model(tmp_path) -> Path` and uses `LlamaCtlApp(repo_root=...)` with `app.run_test(headless=True)`; widget content is asserted via `str(widget.render())` (not `.renderable`) in this Textual version.

---

## Task 1: Estimate is "unavailable" when GGUF lacks KV metadata (Blocker A)

> **Review finding (Correctness — Important #1):** when a GGUF omits `head_count_kv`/`key_length`, `model_params_from_gguf` defaults them to `0`, so `kv_gb == 0` and the Serve tab shows a confident green "✓ under budget" for a config that may OOM. Fix: treat missing KV metadata as "estimate unavailable" (`None`), matching the existing missing-file behavior — never display a falsely-low number.

**Files:**
- Modify: `llamactl/core/estimate.py`
- Test: `tests/llamactl/test_estimate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_estimate.py`:

```python
def test_estimate_vram_none_when_kv_heads_missing(monkeypatch):
    """A GGUF whose header lacks KV-head metadata must yield None, not a
    falsely-low estimate that omits the KV-cache term."""
    from llamactl.core import estimate as est_mod
    from llamactl.core.config import GlobalConfig, ModelConfig

    monkeypatch.setattr(
        est_mod, "resolve_gguf_path", lambda hf, cfg: __import__("pathlib").Path("/x/m.gguf")
    )
    monkeypatch.setattr(
        est_mod, "model_params_from_gguf",
        lambda path: {"arch": "llm", "block_count": 32, "kv_layers": 32,
                      "kv_heads": 0, "head_dim": 128, "weight_gb": 7.0},
    )
    model = ModelConfig(id="m", name="M", hf="org/repo:Q5_K_M", settings={},
                        backends={}, presets={}, images={}, path=None)
    result = est_mod.estimate_vram(model, {"ctx_size": 4096}, _DummyCfg())
    assert result is None


def test_estimate_vram_none_when_head_dim_missing(monkeypatch):
    from llamactl.core import estimate as est_mod
    from llamactl.core.config import ModelConfig

    monkeypatch.setattr(
        est_mod, "resolve_gguf_path", lambda hf, cfg: __import__("pathlib").Path("/x/m.gguf")
    )
    monkeypatch.setattr(
        est_mod, "model_params_from_gguf",
        lambda path: {"arch": "llm", "block_count": 32, "kv_layers": 32,
                      "kv_heads": 8, "head_dim": 0, "weight_gb": 7.0},
    )
    model = ModelConfig(id="m", name="M", hf="org/repo:Q5_K_M", settings={},
                        backends={}, presets={}, images={}, path=None)
    assert est_mod.estimate_vram(model, {"ctx_size": 4096}, _DummyCfg()) is None
```

Add this minimal config stub near the top of the test file if one is not already present (check first — reuse the existing fixture if the file already builds a `GlobalConfig`):

```python
class _DummyCfg:
    """Stand-in GlobalConfig: estimate_vram only forwards it to resolve_gguf_path,
    which is monkeypatched in these tests, so no real fields are read."""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/llamactl/test_estimate.py::test_estimate_vram_none_when_kv_heads_missing -v`
Expected: FAIL — `estimate_vram` currently returns an `Estimate` (with `kv_gb == 0.0`), not `None`.

- [ ] **Step 3: Add the guard in `estimate_vram`**

In `llamactl/core/estimate.py`, inside `estimate_vram`, after the `model_params_from_gguf` call succeeds and before `compute_estimate`, insert:

```python
    if not params.get("kv_heads") or not params.get("head_dim"):
        _log.warning(
            "estimate: GGUF %s lacks KV-cache metadata (kv_heads=%s, head_dim=%s); "
            "estimate unavailable rather than under-counting KV", path,
            params.get("kv_heads"), params.get("head_dim"),
        )
        return None
```

So the tail of the function reads:

```python
    try:
        params = model_params_from_gguf(str(path))
    except Exception as exc:  # corrupt/unreadable header
        _log.warning("estimate: cannot read GGUF %s: %s", path, exc)
        return None
    if not params.get("kv_heads") or not params.get("head_dim"):
        _log.warning(
            "estimate: GGUF %s lacks KV-cache metadata (kv_heads=%s, head_dim=%s); "
            "estimate unavailable rather than under-counting KV", path,
            params.get("kv_heads"), params.get("head_dim"),
        )
        return None
    return compute_estimate(
        params=params,
        ctx_size=int(resolved_settings.get("ctx_size", _DEFAULT_CTX)),
        cache_type_k=resolved_settings.get("cache_type_k"),
        cache_type_v=resolved_settings.get("cache_type_v"),
        batch_size=int(resolved_settings.get("batch_size", _DEFAULT_BATCH)),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_estimate.py -q`
Expected: PASS (new tests green; existing estimate tests still green — `compute_estimate` is unchanged and its golden tests use non-zero KV params).

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/estimate.py tests/llamactl/test_estimate.py
git commit -m "fix: report estimate unavailable when GGUF lacks KV metadata"
```

---

## Task 2: Validate the `org/repo:QUANT` spec in `resolve_gguf_path` (Security #1 + Correctness Minor)

> **Review findings:** Security flagged that `hf_spec` (from user-authored TOML) is interpolated raw into `glob` patterns and a `models--{org}--{repo}` path segment — a `../` in the spec injects into the glob base (low severity, operator-owned config, read-only `getsize`+header, but fails the "validate inputs at boundaries" checklist). Correctness flagged that the repo-less `*{quant}*.gguf` fallback can silently match a different model. Fix: reject specs whose components contain anything outside `[A-Za-z0-9._-]`, and log at debug which pattern matched so a wrong match is auditable.

**Files:**
- Modify: `llamactl/core/estimate.py`
- Test: `tests/llamactl/test_estimate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_estimate.py`:

```python
import re

import pytest


@pytest.mark.parametrize("spec", [
    "../../etc/passwd:Q5_K_M",       # traversal in repo part
    "org/repo:../../../q",           # traversal in quant
    "org/re*po:Q5_K_M",              # glob metachar in repo
])
def test_resolve_gguf_path_rejects_unsafe_spec(spec, tmp_path):
    from llamactl.core.estimate import resolve_gguf_path
    assert resolve_gguf_path(spec, _cfg_with_caches(tmp_path)) is None
```

Add this helper if the file does not already build a `GlobalConfig` pointing at tmp caches (reuse the existing one if present):

```python
def _cfg_with_caches(tmp_path):
    from pathlib import Path
    from llamactl.core.config import GlobalConfig
    (tmp_path / "hf").mkdir()
    (tmp_path / "llama").mkdir()
    return GlobalConfig(
        default_backend="rocm", default_mode="container", port=8080,
        vram_budget_gb=11.0, hf_cache=Path(tmp_path / "hf"),
        llama_cache=Path(tmp_path / "llama"), name_prefix="llamactl",
    )
```

> **Before coding Step 3, verify the `GlobalConfig` field list** by reading `llamactl/core/config.py` — the plan header lists `(default_backend, default_mode, port, vram_budget_gb, hf_cache, llama_cache, name_prefix)`. If the dataclass requires additional fields, add them to `_cfg_with_caches` to match.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/llamactl/test_estimate.py::test_resolve_gguf_path_rejects_unsafe_spec -v`
Expected: FAIL — current code only checks for a `:` and a `/`, so `../../etc/passwd:Q5_K_M` is interpolated into the glob and returns `None` only incidentally (and `org/re*po` would glob). At least one parametrized case fails.

- [ ] **Step 3: Add validation + debug logging**

In `llamactl/core/estimate.py`, add a module-level constant near the other constants:

```python
_SAFE_SPEC_PART = re.compile(r"^[A-Za-z0-9._-]+$")
```

Add `import re` to the imports block.

Replace the head of `resolve_gguf_path` (the parsing + early-return section) with:

```python
    repo_part, _, quant = hf_spec.partition(":")
    if not quant or "/" not in repo_part:
        return None
    org, repo = repo_part.split("/", 1)
    # Validate components before interpolating into glob/path patterns: reject
    # path-traversal and glob metacharacters in operator-supplied `hf` specs.
    if not all(_SAFE_SPEC_PART.match(part) for part in (org, repo, quant)):
        _log.warning("estimate: rejecting unsafe hf spec %r", hf_spec)
        return None
```

Then, at each of the two `return Path(matches[-1])` sites, add a debug log immediately before the return so a wrong match is auditable:

```python
            _log.debug("estimate: resolved %s via pattern %r -> %s",
                       hf_spec, pattern, matches[-1])
            return Path(matches[-1])
```

(Apply to both the llama.cpp-cache loop and the HF-hub snapshot block; `pattern` is in scope at both sites.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_estimate.py -q`
Expected: PASS — all parametrized unsafe specs now return `None`; existing resolver tests (valid `org/repo:QUANT`, empty-quant guard, fallback, hub-subdir) remain green because their specs are alphanumeric.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/estimate.py tests/llamactl/test_estimate.py
git commit -m "fix: validate hf spec components before glob in resolve_gguf_path"
```

---

## Task 3: Thread-safe cancellation via `threading.Event`; init attrs in `__init__` (Blocker B, part 1)

> **Review findings (Concurrency — Important #2; Quality — Minor #5):** `_cancelled` is a plain `bool` written on the UI thread and read on the worker thread — correct only by CPython GIL accident, not by design, and undocumented as a thread boundary. It is also first set in `on_mount`, so any handler firing before mount would `AttributeError`. Fix: use a `threading.Event` as the cross-thread cancel signal and initialize all instance state in `__init__`.

**Files:**
- Modify: `llamactl/ui/screens/test.py`
- Test: `tests/llamactl/test_test_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_test_ui.py`:

```python
@pytest.mark.asyncio
async def test_cancel_signal_is_threading_event(tmp_path):
    """Cancellation must be a threading.Event (explicit cross-thread signal),
    available immediately after construction — not a plain bool set in on_mount."""
    import threading
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        assert isinstance(screen._cancel_evt, threading.Event)
        assert not screen._cancel_evt.is_set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/llamactl/test_test_ui.py::test_cancel_signal_is_threading_event -v`
Expected: FAIL — `AttributeError: 'TestScreen' object has no attribute '_cancel_evt'`.

- [ ] **Step 3: Replace the bool with an Event and move init to `__init__`**

In `llamactl/ui/screens/test.py`:

Add `import threading` at the top of the file (after `from __future__ import annotations`).

Add an `__init__` to `TestScreen` (before `compose`), and reduce `on_mount` to UI work only:

```python
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._running = False          # whether a test run is in progress
        self._cancel_evt = threading.Event()  # cross-thread cancel signal

    def on_mount(self) -> None:
        self._refresh_precondition()
```

In `on_button_pressed`, replace the Stop branch:

```python
        if self._running:
            # User pressed "Stop"
            self._cancel_evt.set()
            event.button.label = "Stopping…"
            event.button.disabled = True
```

In `_start_run`, replace `self._cancelled = False` with:

```python
        self._cancel_evt.clear()
```

In `_run_worker`, replace the `cancel=` argument:

```python
            result = run_oom_check(
                server,
                reporter,
                global_cfg=global_cfg,
                cancel=self._cancel_evt.is_set,
            )
```

(`self._cancel_evt.is_set` is a bound method matching the `Callable[[], bool]` that `run_oom_check`/`run_phases` expect — no lambda needed.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_test_ui.py -q`
Expected: PASS — the new test is green and the existing UI tests (`disabled_without_server`, `verdict_banner_renders`) still pass (they never referenced `_cancelled`).

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/test.py tests/llamactl/test_test_ui.py
git commit -m "refactor: use threading.Event for OOM-test cancellation; init state in __init__"
```

---

## Task 4: Cancel the worker on unmount to prevent thread leak / shutdown hang (Blocker B, part 2)

> **Review finding (Concurrency — Important #1):** there is no `on_unmount`/quit handler that cancels the run. The worker thread is non-daemon and only polls the cancel signal between phases, so quitting during a multi-minute run can leave the thread running (open HTTP client, sampler thread) and delay interpreter exit. Fix: on unmount, set the cancel event and cancel the worker group so no further phases start.

**Files:**
- Modify: `llamactl/ui/screens/test.py`
- Test: `tests/llamactl/test_test_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_test_ui.py`:

```python
@pytest.mark.asyncio
async def test_unmount_sets_cancel_signal(tmp_path):
    """Tearing down the screen mid-run must signal cancellation so the worker
    stops before the next phase (no thread leak / shutdown hang)."""
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        # Simulate an in-progress run.
        screen._running = True
        screen.on_unmount()
        assert screen._cancel_evt.is_set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/llamactl/test_test_ui.py::test_unmount_sets_cancel_signal -v`
Expected: FAIL — `AttributeError: 'TestScreen' object has no attribute 'on_unmount'`.

- [ ] **Step 3: Add `on_unmount`**

In `llamactl/ui/screens/test.py`, add immediately after `on_mount`:

```python
    def on_unmount(self) -> None:
        # Stop any in-flight run: set the cancel signal so the worker won't start
        # another phase, and cancel the worker group. (Textual cannot interrupt an
        # in-flight HTTP request mid-phase, but this prevents the thread leak /
        # delayed-exit when the app quits during a run.)
        self._cancel_evt.set()
        try:
            self.workers.cancel_group(self, "oom-test")
        except Exception:
            # cancel_group raises if no worker group exists yet; harmless on teardown.
            pass
```

> The bare `except Exception` here is acceptable and intentional: `cancel_group` is best-effort cleanup during teardown and we must not let it break unmount. This is distinct from the `_refresh_precondition` bare-except fixed in Task 6 (that one is on a live query path).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_test_ui.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/test.py tests/llamactl/test_test_ui.py
git commit -m "fix: cancel OOM-test worker on unmount to avoid thread leak on exit"
```

---

## Task 5: Remove dead cancel check and de-duplicate the return tuple in `run_phases` (Cross-cutting #1 + Quality #1)

> **Review findings (Correctness Minor #3; Quality Important #1+#2):** the standalone `if cancel(): return ...` at lines 248–249 is unreachable — the phases-2–4 loop already returns on `cancel()` after the last (defrag) iteration. Separately, the `(phases, max(peaks) if peaks else None, any_reading)` tuple is copy-pasted at four return sites, risking drift on future edits. Fix: one `_result()` closure; drop the dead check.

**Files:**
- Modify: `llamactl/core/oomtest.py`
- Test: `tests/llamactl/test_oomtest.py`

- [ ] **Step 1: Confirm existing coverage, then add a parity assertion**

The existing `test_oomtest.py` already exercises short-circuit-on-failure and cancel at each chain position (per the review). Add one explicit assertion that the cancel-after-defrag path still returns the accumulated peak (guards the refactor):

```python
def test_run_phases_cancel_after_defrag_returns_peak():
    """Cancelling so that defrag is the last phase run must still return the
    peak accumulated so far (regression guard for the run_phases refactor)."""
    from llamactl.core.oomtest import run_phases, PhaseSet

    calls = {"n": 0}

    def make_phase(key, peak, *, success=True, last_ok=10):
        class _P:
            def __init__(self, **kw): pass
            def run(self, *a):
                return PhaseResult(
                    key=key, title=key, samples=[
                        PhaseSample(label="s", prompt_length_chars=1, request=None,
                                    peak_vram_gb=peak, post_vram_gb=None,
                                    ok=success, status="ok")
                    ],
                    success=success, summary="", log_excerpt=[], details={},
                    last_ok_tokens=last_ok,
                )
        return _P

    # cancel() flips True after 4 phases have run (ramp+sustained+coldstart+defrag).
    def cancel():
        calls["n"] += 1
        return calls["n"] > 4

    pset = PhaseSet(
        ramp=make_phase("ramp", 5.0), sustained=make_phase("sustained", 6.0),
        cold_start=make_phase("cold", 7.0), defrag=make_phase("defrag", 9.0),
        boundary=make_phase("boundary", 99.0),  # must NOT run
    )

    class _Rep:
        def finish_phase(self, r): pass

    phases, peak, vram_avail = run_phases(
        config_steps=[1], phase_set=pset, cancel=cancel, reporter=_Rep(),
        client=None, prompt_builder=None, vram_monitor=None,
        runtime_inspector=None, runtime_info=None, config=None,
    )
    assert [p.key for p in phases] == ["ramp", "sustained", "cold", "defrag"]
    assert peak == 9.0
    assert vram_avail is True
```

> Before running, confirm the `PhaseResult`/`PhaseSample` constructor keywords match `stress_harness.models` (the plan header lists `PhaseResult(key, title, samples, success, summary, log_excerpt, details, last_ok_tokens)` and `PhaseSample(label, prompt_length_chars, request, peak_vram_gb, post_vram_gb, ok, status)`). Adjust the fakes if the dataclass differs.

- [ ] **Step 2: Run the test to verify it passes against current code**

Run: `python -m pytest tests/llamactl/test_oomtest.py::test_run_phases_cancel_after_defrag_returns_peak -v`
Expected: PASS — this is a characterization test of current behavior; it must stay green through the refactor.

- [ ] **Step 3: Refactor `run_phases`**

In `llamactl/core/oomtest.py`, inside `run_phases`, add a closure after `any_reading = False` / `_track`:

```python
    def _result() -> tuple[list[PhaseResult], float | None, bool]:
        return phases, (max(peaks) if peaks else None), any_reading
```

Replace the Ramp early return:

```python
    # Phase 1: Ramp
    ramp = phase_set.ramp(**kw).run(config_steps)
    _track(ramp)
    if not ramp.success or not ramp.last_ok_tokens or cancel():
        return _result()
    last_ok = ramp.last_ok_tokens
```

Replace the phases-2–4 loop body and delete the dead post-loop check:

```python
    # Phases 2–4: Sustained, ColdStart, Defrag
    for attr, arg in (
        ("sustained", last_ok),
        ("cold_start", last_ok),
        ("defrag", last_ok),
    ):
        result = getattr(phase_set, attr)(**kw).run(arg)
        _track(result)
        if not result.success or cancel():
            return _result()

    # Phase 5: Boundary
    ctx = (
        config.ctx_size_override
        if (config and config.ctx_size_override)
        else last_ok
    )
    boundary = phase_set.boundary(**kw).run(ctx)
    _track(boundary)
    return _result()
```

(The previous standalone `if cancel(): return ...` between the loop and Phase 5 is removed entirely.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_oomtest.py -q`
Expected: PASS — the new parity test and all existing orchestration tests stay green (behavior is identical; only duplication removed).

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "refactor: de-dup run_phases return tuple; drop dead cancel check"
```

---

## Task 6: Replace bare `except` with `except NoMatches` in `_refresh_precondition` (Cross-cutting #3)

> **Review findings (Concurrency Minor; Quality Important #4):** `_refresh_precondition` uses `except Exception: return`, which is broader than intended and inconsistent with the file's own `except NoMatches` pattern used everywhere else — it would mask a genuine query/`find_running` failure. Fix: catch only the widget-lookup miss, matching the rest of the file.

**Files:**
- Modify: `llamactl/ui/screens/test.py`

- [ ] **Step 1: Make the change**

In `llamactl/ui/screens/test.py`, in `_refresh_precondition`, change:

```python
        try:
            pre = self.query_one("#test-precondition", Static)
            btn = self.query_one("#btn-run-test", Button)
        except Exception:
            return
```

to:

```python
        try:
            pre = self.query_one("#test-precondition", Static)
            btn = self.query_one("#btn-run-test", Button)
        except NoMatches:
            return
```

(`NoMatches` is already imported at the top of the file: `from textual.css.query import NoMatches`.)

- [ ] **Step 2: Run the tests to verify nothing regressed**

Run: `python -m pytest tests/llamactl/test_test_ui.py -q`
Expected: PASS — the precondition path is exercised by `test_test_tab_disabled_without_server`; a genuine widget miss still returns quietly, but unexpected errors now surface.

- [ ] **Step 3: Commit**

```bash
git add llamactl/ui/screens/test.py
git commit -m "refactor: narrow _refresh_precondition except to NoMatches"
```

---

## Task 7: Escape verdict detail before Textual-markup interpolation (Correctness/Security Minor)

> **Review findings (Correctness; Security note):** `result.detail` can carry log-excerpt text (`failed.log_excerpt[-3:]`) which may contain `[...]` sequences. It is interpolated into a Textual markup f-string in `_on_verdict`, so bracketed log content is parsed as markup — a rendering corruption (not code execution). Fix: escape `detail` with `rich.markup.escape` before interpolation.

**Files:**
- Modify: `llamactl/ui/screens/test.py`
- Test: `tests/llamactl/test_test_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/llamactl/test_test_ui.py`:

```python
@pytest.mark.asyncio
async def test_verdict_detail_with_brackets_is_escaped(tmp_path):
    """Log-excerpt detail containing [..] must render literally, not be parsed
    as Textual markup."""
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen, _Verdict
    from llamactl.core.oomtest import OomTestResult
    from textual.widgets import Static, TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        screen.post_message(_Verdict(
            OomTestResult("FAIL", None, None, "ramp",
                          "llama_model_load: error [exit code 1]")))
        await pilot.pause()
        rendered = str(screen.query_one("#verdict", Static).render())
        assert "[exit code 1]" in rendered
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/llamactl/test_test_ui.py::test_verdict_detail_with_brackets_is_escaped -v`
Expected: FAIL — Textual parses `[exit code 1]` as a (invalid/dropped) markup tag, so the literal substring is absent from the rendered output.

- [ ] **Step 3: Escape the detail**

In `llamactl/ui/screens/test.py`, add the import at the top:

```python
from rich.markup import escape
```

In `_on_verdict`, change the verdict update to escape `detail` (the verdict label itself is a fixed safe string and stays inside markup):

```python
            self.query_one("#verdict", Static).update(
                f"[{colour} bold]{result.verdict}[/{colour} bold]  {escape(result.detail)}"
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_test_ui.py -q`
Expected: PASS — including `test_verdict_banner_renders` (a plain-text detail is unaffected by `escape`).

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/test.py tests/llamactl/test_test_ui.py
git commit -m "fix: escape OOM-test verdict detail before Textual markup"
```

---

## Task 8: Add a `Reporter` Protocol and type-annotate `run_phases` / `run_oom_check` / worker (Quality Important #3)

> **Review finding (Quality — Important #3):** several public/worker signatures are untyped despite the project's "type annotations on all signatures" standard and recent type-fix history: `run_phases`' `reporter/client/prompt_builder/runtime_inspector/runtime_info/config`, `run_oom_check`'s `reporter`, and `TestScreen._run_worker`. The reporter shape is already documented by `TextualReporter` and the tests' recording reporters — promote it to a `Protocol`.

**Files:**
- Modify: `llamactl/core/oomtest.py`, `llamactl/ui/screens/test.py`

- [ ] **Step 1: Add the `Reporter` Protocol**

In `llamactl/core/oomtest.py`, extend the typing import and add the Protocol after the imports / before `build_vram_monitor`:

```python
from typing import Any, Protocol


class Reporter(Protocol):
    """Duck-typed reporter surface the phases + runner drive.

    `error` is dual-purpose (informational and fatal); `finish_run` is called
    only when all phases pass.
    """

    def start_run(self, result: Any) -> None: ...
    def start_phase(self, phase: Any) -> None: ...
    def record_sample(self, phase_key: str, sample: Any, warn_at: float | None) -> None: ...
    def finish_phase(self, phase: Any) -> None: ...
    def finish_run(self, result: Any) -> None: ...
    def error(self, message: str) -> None: ...
```

(If `from typing import Any` already exists, replace it with the combined import above rather than duplicating.)

- [ ] **Step 2: Annotate the orchestration signatures**

In `run_phases`, annotate the previously-bare parameters:

```python
def run_phases(
    *,
    config_steps: list[int],
    phase_set: PhaseSet,
    cancel: Callable[[], bool],
    reporter: Reporter,
    client: LlamaServerClient,
    prompt_builder: PromptBuilder,
    vram_monitor: VramMonitor,
    runtime_inspector: Any,
    runtime_info: RuntimeInfo,
    config: StressConfig | None = None,
) -> tuple[list[PhaseResult], float | None, bool]:
```

In `run_oom_check`, annotate `reporter`:

```python
def run_oom_check(
    server: ServerInfo,
    reporter: Reporter,
    *,
    global_cfg: GlobalConfig,
    cancel: Callable[[], bool] = lambda: False,
    phase_set: PhaseSet = DEFAULT_PHASES,
) -> OomTestResult:
```

(`LlamaServerClient`, `PromptBuilder`, `VramMonitor`, `RuntimeInfo`, `StressConfig`, `ServerInfo`, `GlobalConfig` are all already imported at the top of `oomtest.py`. `runtime_inspector` stays `Any` because it is either `ContainerRuntimeInspector` or the local `NativeInspector` — a union with no shared protocol; leaving it `Any` avoids a premature abstraction.)

- [ ] **Step 3: Annotate the worker in `test.py`**

In `llamactl/ui/screens/test.py`, annotate `_run_worker` (imports for `ServerInfo` and `GlobalConfig` — add `GlobalConfig` to the existing `from llamactl.core...` imports if absent):

```python
    def _run_worker(self, server: ServerInfo, global_cfg: GlobalConfig) -> None:
```

Add the import if needed:

```python
from llamactl.core.config import GlobalConfig
```

- [ ] **Step 4: Run the full suite + a smoke import**

```bash
python -m pytest tests/llamactl/ -q
python -c "import llamactl.core.oomtest, llamactl.ui.screens.test"
```
Expected: PASS; import succeeds (no annotation/forward-ref errors). If the project runs `ruff`/`mypy` via hooks, expect them to pass too.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py llamactl/ui/screens/test.py
git commit -m "refactor: add Reporter Protocol; annotate run_phases/run_oom_check/worker"
```

---

## Final Verification

- [ ] **Run the complete llamactl suite:**

```bash
python -m pytest tests/llamactl/ -q
```
Expected: all tests pass (the suite was at 212 passing before these fixes; this plan adds ~7 tests and changes no existing behavior except the two intended fixes).

- [ ] **Confirm the diff is scoped** to the three modules + their tests:

```bash
git diff --stat <first-fix-commit>^..HEAD
```
Expected: only `llamactl/core/estimate.py`, `llamactl/core/oomtest.py`, `llamactl/ui/screens/test.py`, and `tests/llamactl/*` changed.

---

## Self-Review Notes (deferred / out of scope)

These review findings are intentionally **not** in this plan:

- **Cancellation granularity inside `RampPhase.run`** (Concurrency Important #3): each phase cannot be interrupted mid-run because the `stress_harness` phases are deliberately unmodified. After Task 4, "Stop"/quit still wait out the current phase. This is a documented limitation of the no-fork harness port; fixing it would require a cancel hook in `stress_harness` itself — a separate change against that subsystem, not the Phase 5 UI. The `#test-footer` already warns a run "can take a few minutes"; if desired, extend that note to mention Stop latency in a follow-up.
- **`NativeLogReader` path confinement** (Security Minor #3) and **`_peak_from` `getattr` consistency** (Quality Minor #6): low-value defense-in-depth / cosmetic; track separately if a hardening pass is scheduled.

**Spec coverage check:** every Critical/Important review finding maps to a task — Blocker A → Task 1; Security #1 + repo-less fallback → Task 2; Concurrency Important #1 → Task 4; Concurrency Important #2 + Quality Minor #5 → Task 3; Quality Important #1+#2 + Correctness Minor #3 → Task 5; Quality Important #4 → Task 6; Quality Important #3 → Task 8; the markup-escaping correctness/security note → Task 7.
