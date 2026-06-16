# llamactl Phase 5 — Pre-launch VRAM Estimate + OOM Test Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-launch VRAM estimate to the Serve screen and a Test tab that runs a quick OOM boundary check against the running llama-server.

**Architecture:** Two new pure-logic core modules (`core/estimate.py`, `core/oomtest.py`) plus one new UI screen (`ui/screens/test.py`); the Serve screen and app shell get small edits. `core/oomtest.py` drives the *existing, unmodified* `stress_harness` phase classes, injecting llamactl's own VRAM source (`core/monitor.read_vram_kib`) and a Textual-aware reporter through the phases' existing constructor parameters. `core/estimate.py` ports `vram_calc.py`'s GGUF math and resolves the model's `org/repo:QUANT` spec against the llama.cpp `-hf` cache.

**Tech Stack:** Python 3.14, Textual, pytest (`asyncio_mode = "strict"`), `stress_harness` (read-only), `vram_calc.py` (ported, not imported).

**Design basis:** `docs/superpowers/specs/2026-06-16-llamactl-phase5-test-tab-design.md` (rev 2).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `llamactl/core/estimate.py` (create) | GGUF-path resolution for `org/repo:QUANT` + VRAM estimate math; `estimate_vram(model, settings, global_cfg) -> Estimate \| None` |
| `llamactl/core/oomtest.py` (create) | Slim QUICK boundary runner over harness phases; VRAM-monitor + native-inspector adapters; `run_oom_check(...) -> OomTestResult`; pure `classify_verdict(...)` |
| `llamactl/ui/screens/test.py` (create) | `TestScreen` (precondition, controls, live table, verdict banner) + thread-safe `TextualReporter` posting Textual messages |
| `llamactl/ui/screens/serve.py` (modify) | Add `#estimate-line` Static beneath `#argv-preview`; populate it in `_refresh_argv_preview` |
| `llamactl/ui/app.py` (modify) | Mount `TestScreen()` in the existing `Test` `TabPane` |
| `tests/llamactl/test_estimate.py` (create) | Resolver + math golden tests |
| `tests/llamactl/test_oomtest.py` (create) | PID resolution, verdict classification, phase orchestration with fakes |
| `tests/llamactl/test_test_ui.py` (create) | Pilot smoke test: tab disabled without server; banner renders |

**Reference facts (verified against the code — use these exact signatures):**

- `vram_calc.py` provides `read_gguf_metadata(path)`, `model_params_from_gguf(path) -> {"arch","block_count","kv_layers","kv_heads","head_dim","weight_gb"}`, `kv_cache_gb(kv_layers, kv_heads, head_dim, ctx_size, cache_bytes)`, `compute_buffer_gb(batch_size)`, and constants `CACHE_TYPE_BYTES`, `VRAM_OVERHEAD_GB = 0.6`, `COMPUTE_BUFFER_PER_512_GB = 0.9`. **Do NOT port `hf_url_to_cache_path`** (it expects a `huggingface.co/...` URL; `ModelConfig.hf` is `org/repo:QUANT`).
- `llamactl/core/config.py`: `GlobalConfig(default_backend, default_mode, port, vram_budget_gb, hf_cache: Path, llama_cache: Path, name_prefix)`; `ModelConfig(id, name, hf, settings, backends, presets, images, path)`; `resolve_settings(model, preset, backend, overrides) -> dict`; `ConfigError`.
- `llamactl/core/monitor.py`: `read_vram_kib(pid: str, fdinfo_root="/proc") -> int` (sums DRM memory KiB; returns 0 if pid missing).
- `llamactl/core/lifecycle.py`: `ServerInfo(model_id, backend, preset, mode, host, port, started_at, container_name=None, log_path=None, pid=None)` (`pid` is `None` for containers); `find_running(global_cfg, state_dir, runtime=None, runner=...) -> ServerInfo | None`.
- `llamactl/core/runtime.py`: `find_runtime() -> str | None`; `get_container_pid(container_name, runtime, runner=...) -> int | None`.
- `stress_harness`: `StressConfig.from_env(environ: dict) -> StressConfig` (honours `QUICK`, `API_URL`, `VRAM_WARN_GB`, `CTX_SIZE`); `StressConfig.build_steps(ctx_size) -> list[int]`; `StressConfig.filler_chunk`, `.tokens_per_chunk`, `.vram_warn_gb`, `.ctx_size_override`, `.ctx_size_fallback`. `PromptBuilder(filler_chunk, tokens_per_chunk)`. `LlamaServerClient(config)` with `server_healthy() -> bool`, `server_ctx_size() -> int`. `VramMonitor(reader, mode)` — `reader` is a zero-arg callable returning `float | None` (GiB); `.read()` calls it. `RuntimeInfo(runtime, container_id, status_message, vram_mode="n/a")`. `ContainerRuntimeInspector(api_url, log_lines=30)` with `detect() -> RuntimeInfo`, `start_log_reader(info)`, `container_running(info)`. Phase classes `RampPhase, SustainedPhase, ColdStartPhase, DefragPhase, BoundaryPhase`, each constructed as `Phase(config, client, prompt_builder, vram_monitor, runtime_inspector, runtime_info, reporter)`; run signatures: `RampPhase.run(steps)`, `SustainedPhase.run(last_ok_tokens)`, `ColdStartPhase.run(last_ok_tokens)`, `DefragPhase.run(last_ok_tokens)`, `BoundaryPhase.run(ctx_size)`. Each returns a `PhaseResult(key, title, samples, success, summary, log_excerpt, details, last_ok_tokens)`; `PhaseSample(label, prompt_length_chars, request, peak_vram_gb, post_vram_gb, ok, status)`.
- Reporter duck-typed surface the phases + runner call: `start_run(result)`, `start_phase(phase)`, `record_sample(phase_key, sample, warn_at)`, `finish_phase(phase)`, `finish_run(result)`, `error(message)`. **`error` is dual-purpose** (fatal *and* informational); **`finish_run` is called only when all phases pass.**
- `ContainerLogReader` surface phases use: `line_count()`, `dump_lines(limit)`, `stop()`.
- Tests live in `tests/llamactl/`; UI tests use `pytest.mark.asyncio` + `app.run_test(headless=True)`.

---

## Task 1: Estimate math + `Estimate` dataclass

**Files:**
- Create: `llamactl/core/estimate.py`
- Test: `tests/llamactl/test_estimate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_estimate.py
from __future__ import annotations

from llamactl.core.estimate import Estimate, compute_estimate


def test_compute_estimate_breakdown():
    params = {
        "kv_layers": 32, "kv_heads": 8, "head_dim": 128, "weight_gb": 7.0,
    }
    est = compute_estimate(
        params=params,
        ctx_size=4096,
        cache_type_k="q8_0",   # 1.0 byte
        cache_type_v="q8_0",   # 1.0 byte
        batch_size=512,
    )
    # KV = (1.0 + 1.0) * 32 * 8 * 128 * 4096 / 1024**3
    expected_kv = (1.0 + 1.0) * 32 * 8 * 128 * 4096 / 1024**3
    assert est.model_gb == 7.0
    assert abs(est.kv_gb - expected_kv) < 1e-9
    assert abs(est.compute_gb - 0.9) < 1e-9          # 0.9 * (512/512)
    assert abs(est.overhead_gb - 0.6) < 1e-9
    assert abs(est.total_gb - (7.0 + expected_kv + 0.9 + 0.6)) < 1e-9


def test_compute_estimate_defaults_to_f16_cache():
    params = {"kv_layers": 1, "kv_heads": 1, "head_dim": 1, "weight_gb": 1.0}
    est = compute_estimate(params=params, ctx_size=1, cache_type_k=None,
                           cache_type_v=None, batch_size=512)
    # f16 = 2.0 bytes each → (2+2) * 1 * 1 * 1 * 1 / 1024**3
    assert abs(est.kv_gb - (4.0 / 1024**3)) < 1e-15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_estimate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamactl.core.estimate'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/estimate.py
"""Pre-launch VRAM estimate.

Ports the GGUF-metadata VRAM math from vram_calc.py (calibrated for this repo's
gfx1031 setup) and resolves the model's `org/repo:QUANT` -hf spec against the
local llama.cpp cache. Local-only: no network access. Returns None ("estimate
unavailable") when the GGUF is not on disk; never blocks launch.
"""
from __future__ import annotations

from dataclasses import dataclass

# Calibrated constants (ported from vram_calc.py).
CACHE_TYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "f32": 4.0, "q8_0": 1.0, "q5_1": 0.6875,
    "q5_0": 0.625, "q4_1": 0.5625, "q4_0": 0.5,
}
VRAM_OVERHEAD_GB = 0.6
COMPUTE_BUFFER_PER_512_GB = 0.9


@dataclass(frozen=True, slots=True)
class Estimate:
    total_gb: float
    model_gb: float
    kv_gb: float
    compute_gb: float
    overhead_gb: float


def compute_estimate(
    *,
    params: dict,
    ctx_size: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
    batch_size: int,
) -> Estimate:
    """Pure VRAM math from GGUF params + resolved settings. Units are GiB."""
    k_bytes = CACHE_TYPE_BYTES.get(cache_type_k or "f16", 2.0)
    v_bytes = CACHE_TYPE_BYTES.get(cache_type_v or "f16", 2.0)
    model_gb = float(params["weight_gb"])
    kv_gb = (
        (k_bytes + v_bytes)
        * params["kv_layers"]
        * params["kv_heads"]
        * params["head_dim"]
        * ctx_size
        / 1024 ** 3
    )
    compute_gb = COMPUTE_BUFFER_PER_512_GB * (batch_size / 512)
    total = model_gb + kv_gb + compute_gb + VRAM_OVERHEAD_GB
    return Estimate(
        total_gb=total,
        model_gb=model_gb,
        kv_gb=kv_gb,
        compute_gb=compute_gb,
        overhead_gb=VRAM_OVERHEAD_GB,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_estimate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/estimate.py tests/llamactl/test_estimate.py
git commit -m "feat: estimate VRAM math + Estimate dataclass"
```

---

## Task 2: GGUF-path resolver for `org/repo:QUANT`

**Files:**
- Modify: `llamactl/core/estimate.py`
- Test: `tests/llamactl/test_estimate.py`

**Note for implementer:** Before finalizing, inspect a real `~/.cache/llama.cpp` directory to confirm the cached `.gguf` filename pattern (llama.cpp's `-hf` downloader flattens names, e.g. `unsloth_Qwen3.5-9B-GGUF_*Q5_K_M*.gguf`, sometimes with a `manifest=…` sidecar). The glob below matches on the repo tail + quant substrings, which is robust to the exact prefix; widen it if the real names differ.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/llamactl/test_estimate.py
from pathlib import Path

from llamactl.core.estimate import resolve_gguf_path


def _global_cfg(tmp_path: Path):
    from llamactl.core.config import GlobalConfig
    return GlobalConfig(
        llama_cache=tmp_path / "llama",
        hf_cache=tmp_path / "hf",
    )


def test_resolve_finds_flattened_gguf_in_llama_cache(tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    target = cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-UD-Q5_K_XL.gguf"
    target.write_bytes(b"GGUF")
    got = resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg)
    assert got == target


def test_resolve_falls_back_to_hf_hub_layout(tmp_path):
    cfg = _global_cfg(tmp_path)
    snap = cfg.hf_cache / "models--unsloth--Qwen3.5-9B-GGUF" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    target = snap / "Qwen3.5-9B-UD-Q5_K_XL.gguf"
    target.write_bytes(b"GGUF")
    got = resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg)
    assert got == target


def test_resolve_returns_none_when_absent(tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    assert resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg) is None


def test_resolve_returns_none_on_malformed_spec(tmp_path):
    cfg = _global_cfg(tmp_path)
    assert resolve_gguf_path("no-colon-here", cfg) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_estimate.py -k resolve -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_gguf_path'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to llamactl/core/estimate.py
import glob
from pathlib import Path

from llamactl.core.config import GlobalConfig


def resolve_gguf_path(hf_spec: str, global_cfg: GlobalConfig) -> Path | None:
    """Resolve an `org/repo:QUANT` -hf spec to a local .gguf path, or None.

    Order: llama.cpp -hf cache (flattened names) → HF hub snapshot layout.
    Local-only; never raises.
    """
    if ":" not in hf_spec or "/" not in hf_spec.split(":", 1)[0]:
        return None
    repo_part, quant = hf_spec.split(":", 1)
    org, repo = repo_part.split("/", 1)

    # 1. llama.cpp -hf cache: flattened filenames containing repo + quant.
    llama_cache = Path(global_cfg.llama_cache).expanduser()
    for pattern in (f"*{repo}*{quant}*.gguf", f"*{quant}*.gguf"):
        matches = sorted(glob.glob(str(llama_cache / pattern)))
        if matches:
            return Path(matches[-1])

    # 2. HF hub snapshot layout (huggingface-cli downloads).
    hf_cache = Path(global_cfg.hf_cache).expanduser()
    hub = hf_cache / "hub" if (hf_cache / "hub").is_dir() else hf_cache
    pattern = str(hub / f"models--{org}--{repo}" / "snapshots" / "*" / f"*{quant}*.gguf")
    matches = sorted(glob.glob(pattern))
    if matches:
        return Path(matches[-1])

    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_estimate.py -k resolve -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/estimate.py tests/llamactl/test_estimate.py
git commit -m "feat: resolve org/repo:QUANT to local GGUF path"
```

---

## Task 3: `estimate_vram` end-to-end

**Files:**
- Modify: `llamactl/core/estimate.py`
- Test: `tests/llamactl/test_estimate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/llamactl/test_estimate.py
from llamactl.core.estimate import estimate_vram
from llamactl.core.config import ModelConfig


def _model(hf: str) -> ModelConfig:
    return ModelConfig(
        id="m", name="m", hf=hf, settings={}, backends={}, presets={}, images={},
        path=Path("x.toml"),
    )


def test_estimate_vram_returns_none_when_gguf_absent(tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    assert estimate_vram(model, {"ctx_size": 4096}, cfg) is None


def test_estimate_vram_uses_resolved_settings(tmp_path, monkeypatch):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    gguf = cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_x-UD-Q5_K_XL.gguf"
    gguf.write_bytes(b"GGUF")

    fake_params = {"kv_layers": 4, "kv_heads": 2, "head_dim": 64, "weight_gb": 3.0}
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: fake_params,
    )
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    settings = {"ctx_size": 2048, "cache_type_k": "q8_0",
                "cache_type_v": "q8_0", "batch_size": 1024}
    est = estimate_vram(model, settings, cfg)
    assert est is not None
    assert est.model_gb == 3.0
    assert abs(est.compute_gb - 0.9 * (1024 / 512)) < 1e-9


def test_estimate_vram_returns_none_on_unreadable_gguf(tmp_path, monkeypatch):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_x-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: (_ for _ in ()).throw(ValueError("bad header")),
    )
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    assert estimate_vram(model, {"ctx_size": 2048}, cfg) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_estimate.py -k estimate_vram -v`
Expected: FAIL with `ImportError: cannot import name 'estimate_vram'`

- [ ] **Step 3: Write minimal implementation**

Port `model_params_from_gguf` (and its helpers `read_gguf_metadata`, `_read_string`, `_read_value`, GGUF type IDs, `GGUF_MAGIC`) verbatim from `vram_calc.py` into `estimate.py`, then add:

```python
# add to llamactl/core/estimate.py
import logging

_log = logging.getLogger(__name__)

_DEFAULT_CTX = 4096
_DEFAULT_BATCH = 512


def estimate_vram(model, resolved_settings: dict, global_cfg: GlobalConfig):
    """Estimate VRAM (GiB) for a model + resolved settings, or None if the GGUF
    is not locally available / not readable. Never raises."""
    path = resolve_gguf_path(model.hf, global_cfg)
    if path is None:
        return None
    try:
        params = model_params_from_gguf(str(path))
    except Exception as exc:  # corrupt/unreadable header
        _log.warning("estimate: cannot read GGUF %s: %s", path, exc)
        return None
    return compute_estimate(
        params=params,
        ctx_size=int(resolved_settings.get("ctx_size", _DEFAULT_CTX)),
        cache_type_k=resolved_settings.get("cache_type_k"),
        cache_type_v=resolved_settings.get("cache_type_v"),
        batch_size=int(resolved_settings.get("batch_size", _DEFAULT_BATCH)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_estimate.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/estimate.py tests/llamactl/test_estimate.py
git commit -m "feat: estimate_vram ties resolver + GGUF params + math"
```

---

## Task 4: Wire the estimate into the Serve screen

**Files:**
- Modify: `llamactl/ui/screens/serve.py` (`_LaunchForm.compose` adds `#estimate-line`; `_refresh_argv_preview` populates it)
- Test: `tests/llamactl/test_test_ui.py` (created here for the Serve assertion; UI test for the Test tab is added in Task 10)

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_test_ui.py
from __future__ import annotations

from pathlib import Path

import pytest


def _repo_with_model(tmp_path: Path) -> Path:
    models = tmp_path / "configs" / "models"
    models.mkdir(parents=True)
    (tmp_path / "state").mkdir()
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (models / "m.toml").write_text(
        'name = "M"\nhf = "unsloth/Foo-GGUF:Q5_K_M"\n\n[settings]\nctx_size = 4096\n'
    )
    return tmp_path


@pytest.mark.asyncio
async def test_serve_shows_estimate_unavailable_when_no_gguf(tmp_path):
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.serve import _LaunchForm
    from textual.widgets import Static, Select

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        form = app.query_one(_LaunchForm)
        model_select = form.query_one("#model-select", Select)
        # select the only model
        model_select.value = "m"
        await pilot.pause()
        line = form.query_one("#estimate-line", Static)
        assert "unavailable" in str(line.renderable).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_test_ui.py::test_serve_shows_estimate_unavailable_when_no_gguf -v`
Expected: FAIL — no widget with id `estimate-line`.

- [ ] **Step 3: Write minimal implementation**

In `llamactl/ui/screens/serve.py`, in `_LaunchForm.compose`, add the estimate line directly after the existing `#argv-preview` Static (currently the last yielded widget, `serve.py:232`):

```python
        yield Static("(select a model to preview the launch command)", id="argv-preview")
        yield Static("", id="estimate-line")
```

Then extend `_refresh_argv_preview` (`serve.py:325-348`). After the `preview.update(...)` call inside the `try`, and in the model-less / error branches, set the estimate line. Replace the method body's tail with:

```python
    def _refresh_argv_preview(self) -> None:
        try:
            preview = self.query_one("#argv-preview", Static)
            est_line = self.query_one("#estimate-line", Static)
        except NoMatches:
            return

        model = self._get_selected_model()
        if model is None:
            preview.update("(select a model to preview the launch command)")
            est_line.update("")
            return

        backend = self._get_selected_backend()
        preset = self._get_selected_preset()

        try:
            app: LlamaCtlApp = self.app  # type: ignore[assignment]
            port = app._global_cfg.port
            settings = resolve_settings(model, preset, backend, {})
            image = resolve_image(model, backend, self._get_selected_artifact())
            argv = build_server_argv(model.hf, settings, "0.0.0.0", port)
            argv_str = " ".join(argv)
            preview.update(f"[bold]Image:[/bold] {image}\n[bold]argv:[/bold] {argv_str}")
            est_line.update(self._format_estimate(model, settings, app._global_cfg))
        except Exception as exc:
            preview.update(f"[red]Config error: {exc}[/red]")
            est_line.update("")

    @staticmethod
    def _format_estimate(model, settings, global_cfg) -> str:
        from llamactl.core.estimate import estimate_vram
        est = estimate_vram(model, settings, global_cfg)
        if est is None:
            return "[dim]Est: unavailable — model not downloaded[/dim]"
        budget = global_cfg.vram_budget_gb
        mark = "✓" if est.total_gb <= budget else "⚠"
        body = (
            f"Est: {est.total_gb:.1f} GB  "
            f"(model {est.model_gb:.1f} + KV {est.kv_gb:.1f} + buf {est.compute_gb:.1f})  "
            f"— budget {budget:.0f} GB {mark}"
        )
        return body if est.total_gb <= budget else f"[red]{body}[/red]"
```

Add `from textual.widgets import Static` is already imported in serve.py; confirm `Static` is in the import list (it is used for `#argv-preview`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_test_ui.py::test_serve_shows_estimate_unavailable_when_no_gguf -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/serve.py tests/llamactl/test_test_ui.py
git commit -m "feat: pre-launch VRAM estimate line on Serve screen"
```

---

## Task 5: OOM-test VRAM monitor builder (PID resolution)

**Files:**
- Create: `llamactl/core/oomtest.py`
- Test: `tests/llamactl/test_oomtest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_oomtest.py
from __future__ import annotations

from llamactl.core.lifecycle import ServerInfo
from llamactl.core.oomtest import build_vram_monitor


def test_build_vram_monitor_native_uses_pid(monkeypatch):
    monkeypatch.setattr(
        "llamactl.core.oomtest.read_vram_kib",
        lambda pid, **kw: 2 * 1024 * 1024 if pid == "4242" else 0,  # 2 GiB
    )
    server = ServerInfo(model_id="m", backend="rocm", preset="", mode="native",
                        host="127.0.0.1", port=8080, started_at="", pid=4242)
    mon = build_vram_monitor(server, runtime=None, get_pid=None)
    assert abs(mon.read() - 2.0) < 1e-9


def test_build_vram_monitor_container_resolves_pid_once(monkeypatch):
    calls = {"inspect": 0}

    def fake_get_pid(name, rt, **kw):
        calls["inspect"] += 1
        return 999

    monkeypatch.setattr(
        "llamactl.core.oomtest.read_vram_kib",
        lambda pid, **kw: 1024 * 1024 if pid == "999" else 0,  # 1 GiB
    )
    server = ServerInfo(model_id="m", backend="rocm", preset="", mode="container",
                        host="0.0.0.0", port=8080, started_at="",
                        container_name="llamactl-m")
    mon = build_vram_monitor(server, runtime="podman", get_pid=fake_get_pid)
    assert abs(mon.read() - 1.0) < 1e-9
    mon.read()
    mon.read()
    assert calls["inspect"] == 1  # resolved once, not per read


def test_build_vram_monitor_returns_none_when_no_pid(monkeypatch):
    server = ServerInfo(model_id="m", backend="rocm", preset="", mode="container",
                        host="0.0.0.0", port=8080, started_at="",
                        container_name="llamactl-m")
    mon = build_vram_monitor(server, runtime="podman",
                             get_pid=lambda *a, **k: None)
    assert mon.read() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_oomtest.py -k vram_monitor -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamactl.core.oomtest'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/oomtest.py
"""Quick OOM boundary check against a running llama-server.

Drives the UNMODIFIED stress_harness phase classes, injecting llamactl's own
VRAM source (core/monitor.read_vram_kib) and a caller-supplied reporter through
the phases' existing constructor parameters. Works for native and container
servers.
"""
from __future__ import annotations

from stress_harness.monitoring import VramMonitor

from llamactl.core.lifecycle import ServerInfo
from llamactl.core.monitor import read_vram_kib
from llamactl.core.runtime import find_runtime, get_container_pid


def build_vram_monitor(
    server: ServerInfo,
    runtime: str | None,
    get_pid=get_container_pid,
) -> VramMonitor:
    """A harness VramMonitor whose reader reports the managed server's VRAM in
    GiB via /proc fdinfo. PID resolved ONCE up front (container inspect is a
    subprocess; the 200ms PeakVramSampler must not re-resolve per tick)."""
    if server.mode == "native":
        pid = str(server.pid) if server.pid else None
        mode = f"per-process native (PID: {pid})"
    else:
        rt = runtime or find_runtime()
        resolved = (
            get_pid(server.container_name, rt)
            if (rt and server.container_name and get_pid)
            else None
        )
        pid = str(resolved) if resolved else None
        mode = f"per-process container (PID: {pid})"

    def _reader() -> float | None:
        if pid is None:
            return None
        kib = read_vram_kib(pid)
        return kib / 1024 ** 2 if kib > 0 else None

    return VramMonitor(_reader, mode)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_oomtest.py -k vram_monitor -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "feat: oomtest VRAM monitor with one-shot PID resolution"
```

---

## Task 6: Native runtime-inspector adapter

**Files:**
- Modify: `llamactl/core/oomtest.py`
- Test: `tests/llamactl/test_oomtest.py`

**Why:** the harness phase constructors require a `runtime_inspector` whose `start_log_reader(info)` yields a log reader and whose `container_running(info)` is polled by the request watchdog. For native servers there is no container, so we supply an adapter that tails `server.log_path`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/llamactl/test_oomtest.py
from llamactl.core.oomtest import NativeInspector, NativeLogReader


def test_native_log_reader_tails_file(tmp_path):
    log = tmp_path / "srv.log"
    log.write_text("\n".join(f"line{i}" for i in range(10)) + "\n")
    reader = NativeLogReader(str(log))
    assert reader.line_count() == 10
    assert reader.dump_lines(3) == ["line7", "line8", "line9"]
    reader.stop()  # no-op, must not raise


def test_native_log_reader_handles_missing_path():
    reader = NativeLogReader(None)
    assert reader.line_count() == 0
    assert reader.dump_lines(5) == []
    reader.stop()


def test_native_inspector_container_running_is_none(tmp_path):
    insp = NativeInspector(str(tmp_path / "srv.log"))
    from stress_harness.models import RuntimeInfo
    info = RuntimeInfo(runtime=None, container_id=None, status_message="native")
    assert insp.container_running(info) is None
    reader = insp.start_log_reader(info)
    assert reader is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_oomtest.py -k native -v`
Expected: FAIL with `ImportError: cannot import name 'NativeInspector'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to llamactl/core/oomtest.py


class NativeLogReader:
    """Minimal stand-in for ContainerLogReader over a native server's log file."""

    def __init__(self, log_path: str | None) -> None:
        self._path = log_path

    def _lines(self) -> list[str]:
        if not self._path:
            return []
        try:
            with open(self._path, encoding="utf-8", errors="replace") as fh:
                return fh.read().splitlines()
        except OSError:
            return []

    def line_count(self) -> int:
        return len(self._lines())

    def dump_lines(self, limit: int) -> list[str]:
        lines = self._lines()
        return lines[-limit:] if limit > 0 else lines

    def stop(self) -> None:
        return None


class NativeInspector:
    """Runtime-inspector adapter for native servers (no container)."""

    def __init__(self, log_path: str | None) -> None:
        self._log_path = log_path

    def start_log_reader(self, info) -> NativeLogReader:
        return NativeLogReader(self._log_path)

    def container_running(self, info):
        return None  # native: watchdog treats None as "still running"

    def container_pids(self, info) -> list[str]:
        return []

    def api_host_port(self):
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_oomtest.py -k native -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "feat: native runtime-inspector + log-reader adapters for oomtest"
```

---

## Task 7: Verdict classification (pure)

**Files:**
- Modify: `llamactl/core/oomtest.py`
- Test: `tests/llamactl/test_oomtest.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/llamactl/test_oomtest.py
from stress_harness.models import PhaseResult, PhaseSample
from llamactl.core.oomtest import OomTestResult, classify_verdict


def _phase(key, success=True, samples=None, last_ok=None, summary=""):
    return PhaseResult(key=key, title=key.title(), samples=samples or [],
                       success=success, summary=summary, last_ok_tokens=last_ok)


def test_classify_ok_when_under_budget():
    phases = [_phase("ramp", last_ok=120000),
              _phase("boundary")]
    res = classify_verdict(phases, peak_vram_gb=9.5, budget_gb=11.0, vram_available=True)
    assert res.verdict == "OK"
    assert res.peak_vram_gb == 9.5
    assert res.last_ok_tokens == 120000


def test_classify_warn_when_at_or_over_budget():
    res = classify_verdict([_phase("ramp")], peak_vram_gb=11.2, budget_gb=11.0,
                           vram_available=True)
    assert res.verdict == "WARN"


def test_classify_fail_on_phase_failure():
    phases = [_phase("ramp"), _phase("cold-start", success=False,
                                     summary="OOM at 130k tokens")]
    res = classify_verdict(phases, peak_vram_gb=10.0, budget_gb=11.0,
                           vram_available=True)
    assert res.verdict == "FAIL"
    assert res.failed_phase == "cold-start"
    assert "OOM" in res.detail


def test_classify_degraded_when_vram_unavailable():
    res = classify_verdict([_phase("ramp")], peak_vram_gb=None, budget_gb=11.0,
                           vram_available=False)
    assert res.verdict == "OK (degraded)"
    assert res.peak_vram_gb is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_oomtest.py -k classify -v`
Expected: FAIL with `ImportError: cannot import name 'classify_verdict'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to llamactl/core/oomtest.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OomTestResult:
    verdict: str            # "OK" | "WARN" | "FAIL" | "OK (degraded)"
    peak_vram_gb: float | None
    last_ok_tokens: int | None
    failed_phase: str | None
    detail: str


def classify_verdict(phases, peak_vram_gb, budget_gb, vram_available) -> OomTestResult:
    last_ok = next((p.last_ok_tokens for p in phases if p.last_ok_tokens), None)
    failed = next((p for p in phases if not p.success), None)
    if failed is not None:
        detail = failed.summary or " / ".join(failed.log_excerpt[-3:]) or "phase failed"
        return OomTestResult("FAIL", peak_vram_gb, last_ok, failed.key, detail)
    if not vram_available:
        return OomTestResult(
            "OK (degraded)", None, last_ok, None,
            "All phases passed. VRAM unavailable — boundary + liveness only; "
            "leak detection skipped.",
        )
    if peak_vram_gb is not None and peak_vram_gb >= budget_gb:
        return OomTestResult(
            "WARN", peak_vram_gb, last_ok, None,
            f"All phases passed but peak {peak_vram_gb:.2f} GB ≥ budget {budget_gb:.0f} GB.",
        )
    peak_str = f"{peak_vram_gb:.2f} GB" if peak_vram_gb is not None else "n/a"
    return OomTestResult(
        "OK", peak_vram_gb, last_ok, None,
        f"All phases passed. Peak VRAM {peak_str}, under the {budget_gb:.0f} GB budget.",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_oomtest.py -k classify -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "feat: oomtest verdict classification"
```

---

## Task 8: Phase orchestration + `run_oom_check`

**Files:**
- Modify: `llamactl/core/oomtest.py`
- Test: `tests/llamactl/test_oomtest.py`

**Why injectable phases:** the real harness phases do blocking HTTP + sleeps, so we make the phase classes injectable via a `PhaseSet` so the orchestration (ramp-first → thread `last_ok_tokens` into the rest → short-circuit on failure → cancel between phases) is unit-testable with fakes. The default `PhaseSet` is the real harness chain.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/llamactl/test_oomtest.py
from stress_harness.monitoring import VramMonitor
from llamactl.core.oomtest import PhaseSet, run_phases


class _FakePhase:
    """Returns a canned PhaseResult; records that it ran and with what arg."""
    def __init__(self, key, success=True, last_ok=None, peak=None):
        self._key, self._success, self._last_ok, self._peak = key, success, last_ok, peak

    def make(self, *_args, **_kw):
        phase = self
        class _Runner:
            def run(self, _arg):
                samples = ([PhaseSample(label="s", peak_vram_gb=phase._peak)]
                           if phase._peak is not None else [])
                return _phase(phase._key, success=phase._success,
                              samples=samples, last_ok=phase._last_ok)
        return _Runner()


def _phase_set(specs):
    return PhaseSet(**{k: v.make for k, v in specs.items()})


def _noop_collaborators():
    mon = VramMonitor(lambda: 9.0, "test")
    return dict(client=object(), prompt_builder=object(), vram_monitor=mon,
                runtime_inspector=object(),
                runtime_info=__import__("stress_harness.models", fromlist=["RuntimeInfo"]).RuntimeInfo(None, None, "x"))


def test_run_phases_runs_full_chain_when_all_pass():
    specs = {"ramp": _FakePhase("ramp", last_ok=100, peak=9.0),
             "sustained": _FakePhase("sustained", peak=9.5),
             "cold_start": _FakePhase("cold-start", peak=10.0),
             "defrag": _FakePhase("defrag", peak=9.8),
             "boundary": _FakePhase("boundary", peak=9.0)}
    phases, peak, vram_available = run_phases(
        config_steps=[10, 20], phase_set=_phase_set(specs),
        cancel=lambda: False, reporter=_RecordingReporter(), **_noop_collaborators())
    assert [p.key for p in phases] == ["ramp", "sustained", "cold-start", "defrag", "boundary"]
    assert peak == 10.0
    assert vram_available is True


def test_run_phases_short_circuits_on_ramp_failure():
    specs = {"ramp": _FakePhase("ramp", success=False),
             "sustained": _FakePhase("sustained"),
             "cold_start": _FakePhase("cold-start"),
             "defrag": _FakePhase("defrag"),
             "boundary": _FakePhase("boundary")}
    phases, _, _ = run_phases(config_steps=[10], phase_set=_phase_set(specs),
                              cancel=lambda: False, reporter=_RecordingReporter(),
                              **_noop_collaborators())
    assert [p.key for p in phases] == ["ramp"]


def test_run_phases_stops_on_cancel():
    specs = {"ramp": _FakePhase("ramp", last_ok=100, peak=9.0),
             "sustained": _FakePhase("sustained"),
             "cold_start": _FakePhase("cold-start"),
             "defrag": _FakePhase("defrag"),
             "boundary": _FakePhase("boundary")}
    phases, _, _ = run_phases(config_steps=[10], phase_set=_phase_set(specs),
                              cancel=lambda: True, reporter=_RecordingReporter(),
                              **_noop_collaborators())
    assert [p.key for p in phases] == ["ramp"]  # cancelled after first


class _RecordingReporter:
    def start_run(self, r): ...
    def start_phase(self, p): ...
    def record_sample(self, k, s, w): ...
    def finish_phase(self, p): ...
    def finish_run(self, r): ...
    def error(self, m): ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_oomtest.py -k run_phases -v`
Expected: FAIL with `ImportError: cannot import name 'PhaseSet'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to llamactl/core/oomtest.py
from dataclasses import dataclass as _dataclass

from stress_harness.phases import (
    BoundaryPhase, ColdStartPhase, DefragPhase, RampPhase, SustainedPhase,
)


@_dataclass(frozen=True)
class PhaseSet:
    ramp: object = RampPhase
    sustained: object = SustainedPhase
    cold_start: object = ColdStartPhase
    defrag: object = DefragPhase
    boundary: object = BoundaryPhase


DEFAULT_PHASES = PhaseSet()


def _peak_from(samples) -> float | None:
    vals = [s.peak_vram_gb for s in samples if s.peak_vram_gb is not None]
    vals += [s.post_vram_gb for s in samples if getattr(s, "post_vram_gb", None) is not None]
    return max(vals) if vals else None


def run_phases(
    *,
    config_steps,
    phase_set,
    cancel,
    reporter,
    client,
    prompt_builder,
    vram_monitor,
    runtime_inspector,
    runtime_info,
    config=None,
):
    """Run ramp→sustained→cold→defrag→boundary, short-circuiting on failure or
    cancel. Returns (phases, peak_vram_gb, vram_available)."""
    kw = dict(config=config, client=client, prompt_builder=prompt_builder,
              vram_monitor=vram_monitor, runtime_inspector=runtime_inspector,
              runtime_info=runtime_info, reporter=reporter)
    phases = []
    peaks: list[float] = []
    any_reading = False

    def _track(result):
        nonlocal any_reading
        phases.append(result)
        reporter.finish_phase(result)
        p = _peak_from(result.samples)
        if p is not None:
            peaks.append(p)
            any_reading = True

    ramp = phase_set.ramp(**kw).run(config_steps)
    _track(ramp)
    if not ramp.success or not ramp.last_ok_tokens or cancel():
        return phases, (max(peaks) if peaks else None), any_reading
    last_ok = ramp.last_ok_tokens

    for attr, arg in (("sustained", last_ok), ("cold_start", last_ok),
                      ("defrag", last_ok)):
        result = getattr(phase_set, attr)(**kw).run(arg)
        _track(result)
        if not result.success or cancel():
            return phases, (max(peaks) if peaks else None), any_reading

    ctx = config.ctx_size_override if (config and config.ctx_size_override) else last_ok
    boundary = phase_set.boundary(**kw).run(ctx)
    _track(boundary)
    return phases, (max(peaks) if peaks else None), any_reading
```

> Note: the `_FakePhase` in the test ignores `**kw`, so `config=None` is fine in tests. In `run_oom_check` (next step) a real `StressConfig` is passed.

Now add the public entry point:

```python
# add to llamactl/core/oomtest.py
from pathlib import Path

from stress_harness.config import StressConfig
from stress_harness.prompting import PromptBuilder
from stress_harness.runtime import ContainerRuntimeInspector
from stress_harness.server import LlamaServerClient

from llamactl.core.config import GlobalConfig


def run_oom_check(
    server: ServerInfo,
    reporter,
    *,
    global_cfg: GlobalConfig,
    cancel=lambda: False,
    phase_set: PhaseSet = DEFAULT_PHASES,
) -> OomTestResult:
    """Run the QUICK OOM boundary check against the running server."""
    api_url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
    config = StressConfig.from_env({
        "QUICK": "1",
        "API_URL": api_url,
        "VRAM_WARN_GB": str(global_cfg.vram_budget_gb),
        # NB: do NOT set CTX_SIZE — let the harness auto-detect from /slots.
    })
    client = LlamaServerClient(config)
    if not client.server_healthy():
        return OomTestResult("FAIL", None, None, None,
                             f"Server not reachable at {api_url}")

    prompt_builder = PromptBuilder(config.filler_chunk, config.tokens_per_chunk)

    if server.mode == "native":
        inspector = NativeInspector(server.log_path)
        from stress_harness.models import RuntimeInfo
        runtime_info = RuntimeInfo(runtime=None, container_id=None,
                                   status_message="native server")
        runtime = None
    else:
        inspector = ContainerRuntimeInspector(api_url)
        runtime_info = inspector.detect()
        runtime = runtime_info.runtime

    vram_monitor = build_vram_monitor(server, runtime)
    ctx_size = client.server_ctx_size()
    config = StressConfig.from_env({
        "QUICK": "1", "API_URL": api_url,
        "VRAM_WARN_GB": str(global_cfg.vram_budget_gb),
    })
    steps = config.build_steps(ctx_size)

    reporter.start_run(_run_summary(config, ctx_size, runtime_info,
                                    vram_monitor.read()))

    phases, peak, vram_available = run_phases(
        config_steps=steps, phase_set=phase_set, cancel=cancel,
        reporter=reporter, client=client, prompt_builder=prompt_builder,
        vram_monitor=vram_monitor, runtime_inspector=inspector,
        runtime_info=runtime_info, config=config,
    )
    return classify_verdict(phases, peak, global_cfg.vram_budget_gb, vram_available)


def _run_summary(config, ctx_size, runtime_info, baseline):
    """Build a StressRunResult for reporter.start_run (it reads .config/.ctx_size/
    .runtime/.baseline_vram_gb)."""
    from stress_harness.models import StressRunResult
    return StressRunResult(config=config, ctx_size=ctx_size, steps=[],
                           runtime=runtime_info, baseline_vram_gb=baseline)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_oomtest.py -v`
Expected: PASS (all). Also run the full suite: `pytest tests/llamactl/test_oomtest.py tests/llamactl/test_estimate.py -v`

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/oomtest.py tests/llamactl/test_oomtest.py
git commit -m "feat: oomtest phase orchestration + run_oom_check entry point"
```

---

## Task 9: Test tab UI — `TestScreen` + `TextualReporter`

**Files:**
- Create: `llamactl/ui/screens/test.py`
- Test: covered by Task 10's Pilot test (no separate failing test here; this task builds the widget the Task 10 test exercises)

**Thread-safety contract:** the worker runs in a `@work(thread=True)` worker; `TextualReporter` must NEVER touch widgets directly. It posts Textual messages via `self._post(...)` (a callback that calls `app.call_from_thread` / `post_message`). The screen handles those messages on the event loop.

- [ ] **Step 1: Create the screen module**

```python
# llamactl/ui/screens/test.py
"""Test tab — quick OOM boundary check against the running server."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, DataTable, Label, Static

from llamactl.core.lifecycle import find_running
from llamactl.core.oomtest import OomTestResult, run_oom_check


class _PhaseRow(Message):
    def __init__(self, cells: tuple[str, ...]) -> None:
        self.cells = cells
        super().__init__()


class _PhaseHeader(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class _Verdict(Message):
    def __init__(self, result: OomTestResult) -> None:
        self.result = result
        super().__init__()


class TextualReporter:
    """Drop-in for ConsoleReporter; posts messages to the screen instead of
    printing. Called from the worker + sampler threads — only posts messages."""

    def __init__(self, screen: "TestScreen") -> None:
        self._screen = screen

    def _post(self, message: Message) -> None:
        self._screen.post_message(message)

    def start_run(self, result) -> None:
        self._post(_PhaseHeader(f"ctx={result.ctx_size:,}  "
                                f"baseline={_fmt(result.baseline_vram_gb)}"))

    def start_phase(self, phase) -> None:
        self._post(_PhaseHeader(f"── {phase.title} ──"))

    def record_sample(self, phase_key, sample, warn_at) -> None:
        req = sample.request
        prefill = req.prefill_display() if req else "—"
        gen = req.gen_toks_display() if req else "—"
        self._post(_PhaseRow((
            phase_key, str(sample.label), prefill, gen,
            _fmt(sample.peak_vram_gb), _fmt(sample.post_vram_gb), sample.status,
        )))

    def finish_phase(self, phase) -> None:
        return None

    def finish_run(self, result) -> None:
        return None  # verdict banner is set from run_oom_check's return value

    def error(self, message) -> None:
        # dual-purpose (info + fatal): show as an informational row, never a banner
        self._post(_PhaseRow(("", "", "", "", "", "", message.strip())))


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


class TestScreen(Widget):
    DEFAULT_CSS = """
    TestScreen { height: 1fr; layout: vertical; }
    TestScreen #test-table { height: 1fr; border: solid $panel; }
    TestScreen #verdict { padding: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("OOM boundary check", id="test-title")
        yield Static("", id="test-precondition")
        yield Button("Run check", id="btn-run-test", disabled=True)
        table = DataTable(id="test-table")
        table.add_columns("phase", "step", "prefill", "gen tok/s",
                          "peak", "post", "status")
        yield table
        yield Static("", id="verdict")
        yield Static(
            "[dim]Full multi-phase stress suite: run `python stress_test.py` "
            "(set QUICK=1 for a faster pass). This tab runs a quick subset and "
            "can take a few minutes at large context.[/dim]",
            id="test-footer",
        )

    def on_mount(self) -> None:
        self._refresh_precondition()

    def _server(self):
        app = self.app
        return find_running(app._global_cfg, app._state_dir)

    def _refresh_precondition(self) -> None:
        server = self._server()
        pre = self.query_one("#test-precondition", Static)
        btn = self.query_one("#btn-run-test", Button)
        if server is None:
            pre.update("[yellow]Start a server on the Serve tab first.[/yellow]")
            btn.disabled = True
        else:
            pre.update(f"Target: {server.model_id} / {server.backend} "
                       f"/ {server.mode} / port {server.port}")
            btn.disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-run-test":
            return
        if event.button.label == "Stop":
            self._cancelled = True
            event.button.label = "Stopping…"
            return
        self._start_run(event.button)

    def _start_run(self, button: Button) -> None:
        server = self._server()
        if server is None:
            return
        self._cancelled = False
        self.query_one("#test-table", DataTable).clear()
        self.query_one("#verdict", Static).update("[dim]Running…[/dim]")
        button.label = "Stop"
        self._run_worker(server)

    def _run_worker(self, server) -> None:
        from textual import work

        @work(thread=True, exclusive=True)
        def _worker() -> None:
            reporter = TextualReporter(self)
            try:
                result = run_oom_check(
                    server, reporter,
                    global_cfg=self.app._global_cfg,
                    cancel=lambda: getattr(self, "_cancelled", False),
                )
            except Exception as exc:  # never hang in "running"
                result = OomTestResult("FAIL", None, None, None,
                                       f"Test crashed: {exc!r}")
            self.post_message(_Verdict(result))

        _worker()

    # ── message handlers (event loop) ───────────────────────────────────────
    def on_test_screen_phase_header(self, message: "_PhaseHeader") -> None:
        ...  # headers are optional; rows carry the phase column

    def on_test_screen_phase_row(self, message: "_PhaseRow") -> None:
        self.query_one("#test-table", DataTable).add_row(*message.cells)

    def on_test_screen_verdict(self, message: "_Verdict") -> None:
        result = message.result
        colour = {"OK": "green", "WARN": "yellow"}.get(result.verdict, "red")
        if result.verdict.startswith("OK"):
            colour = "green"
        self.query_one("#verdict", Static).update(
            f"[{colour} bold]{result.verdict}[/{colour} bold]  {result.detail}")
        self.query_one("#btn-run-test", Button).label = "Run check"
```

> Textual message-handler naming: a `Message` subclass `_PhaseRow` nested in module scope is dispatched to `on__phase_row` by default. To get stable handler names, define the messages at module level (as above) and add explicit `class Meta` is unnecessary — instead, register handlers with `@on`. Simpler: use `self.post_message` + handle via `on(...)`. Implementer: if the snake_case handler names don't match, switch to the decorator form:
> ```python
> from textual import on
> @on(_PhaseRow)
> def _row(self, message: _PhaseRow) -> None: ...
> ```
> Use whichever Textual dispatch the codebase already uses (check `serve.py` for the pattern: it uses `on_select_changed` / `on_button_pressed`, i.e. event-name dispatch). Prefer the `@on(MessageClass)` decorator for custom messages.

- [ ] **Step 2: Quick import check**

Run: `python -c "import llamactl.ui.screens.test"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add llamactl/ui/screens/test.py
git commit -m "feat: Test tab screen + thread-safe TextualReporter"
```

---

## Task 10: Mount the Test tab + Pilot smoke test

**Files:**
- Modify: `llamactl/ui/app.py:69-70`
- Test: `tests/llamactl/test_test_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/llamactl/test_test_ui.py
@pytest.mark.asyncio
async def test_test_tab_disabled_without_server(tmp_path):
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import Button, TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        btn = screen.query_one("#btn-run-test", Button)
        assert btn.disabled is True


@pytest.mark.asyncio
async def test_verdict_banner_renders(tmp_path):
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
            OomTestResult("WARN", 11.3, 120000, None, "peak over budget")))
        await pilot.pause()
        verdict = screen.query_one("#verdict", Static)
        assert "WARN" in str(verdict.renderable)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_test_ui.py -k "test_tab or verdict_banner" -v`
Expected: FAIL — the Test tab still shows the placeholder `Static`, no `TestScreen`.

- [ ] **Step 3: Write minimal implementation**

In `llamactl/ui/app.py`, add the import and replace the placeholder:

```python
# top imports
from llamactl.ui.screens.test import TestScreen
```

```python
            with TabPane("Test", id="test"):
                yield TestScreen()
```

(replacing `yield Static("OOM boundary test — coming in Phase 5")` at `app.py:70`). If `Static` becomes unused, leave the import — it is still used elsewhere; verify with ruff in the next step.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_test_ui.py -v`
Expected: PASS (all)

- [ ] **Step 5: Run the whole suite + linters**

Run:
```bash
pytest tests/llamactl -q
ruff check llamactl/
```
Expected: all tests pass; ruff clean (fix any unused-import / formatting nits).

- [ ] **Step 6: Commit**

```bash
git add llamactl/ui/app.py tests/llamactl/test_test_ui.py
git commit -m "feat: mount Test tab in app shell"
```

---

## Task 11: Manual verification + docs

**Files:**
- Modify: `README.md` (if it documents tabs) or `docs/` as appropriate

- [ ] **Step 1: Manual smoke (requires a real server)**

Start a server from the Serve tab, switch to Test, press "Run check", confirm: live rows stream in, the run can be Stopped between phases, and a verdict banner appears. With a model GGUF present in `~/.cache/llama.cpp`, confirm the Serve tab shows a concrete `Est: N GB …` line; with an un-downloaded model, confirm it shows "unavailable".

- [ ] **Step 2: Note any calibration adjustments**

If the estimate is materially off vs. the live gauge after load, record the discrepancy (do not silently tune constants — they are shared with `vram_calc.py`).

- [ ] **Step 3: Commit any doc updates**

```bash
git add -A
git commit -m "docs: document Phase 5 Test tab + VRAM estimate"
```

---

## Self-Review

**Spec coverage** (against rev 2 of the design):

- §2 estimate + corrected GGUF resolver → Tasks 1–3 (resolver uses `llama_cache` then `hf_cache`; `global_cfg` in signature). ✓
- §2 Serve wiring (`#estimate-line`, warn-never-block, GiB note) → Task 4. ✓
- §3 slim runner over real phases, no harness edits → Tasks 5–8. ✓
- §3 container VRAM PID resolved once via `get_container_pid` → Task 5. ✓
- §3 native runtime-inspector + log-reader adapter → Task 6. ✓
- §3 ctx auto-detect (no forced `CTX_SIZE`), `VRAM_WARN_GB` from budget → Task 8. ✓
- §3 full reporter surface, dual-purpose `error`, `finish_run` not load-bearing → Task 9 (`TextualReporter`). ✓
- §3 thread-safety (`post_message`) → Task 9. ✓
- §3 cancellation between phases → Tasks 8 (`cancel()` checks) + 9 (Stop button). ✓
- §4 Test tab (precondition/disabled, table, verdict banner, full-suite pointer) → Tasks 9–10. ✓
- §5 verdict OK/WARN/FAIL + degraded → Task 7. ✓
- §6 error handling (server gone → FAIL; bad GGUF → unavailable; VRAM None → degraded; worker crash → FAIL banner) → Tasks 3, 7, 8, 9. ✓
- §7 testing (estimate math + resolver; pid; verdict; orchestration; Pilot smoke) → Tasks 1–10. ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases" left; the one implementer judgment call (Textual custom-message dispatch in Task 9) is given a concrete decision (`@on(MessageClass)`) plus the codebase pattern to follow. The cache-filename verification in Task 2 is a concrete inspection step with a working default glob, not a deferred decision.

**Type consistency:** `Estimate` fields (`total_gb/model_gb/kv_gb/compute_gb/overhead_gb`) consistent across Tasks 1, 3, 4. `OomTestResult` fields (`verdict/peak_vram_gb/last_ok_tokens/failed_phase/detail`) consistent across Tasks 7, 8, 9, 10. `build_vram_monitor`, `classify_verdict`, `run_phases`, `run_oom_check`, `PhaseSet`, `NativeInspector`, `NativeLogReader` names match between definitions and call sites/tests. `estimate_vram(model, resolved_settings, global_cfg)` signature consistent in Tasks 3 and 4.
