# llamactl Phase 2 — Serve Tab & Server Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Serve tab to llamactl — a live dashboard for launching, monitoring, and stopping llama-server (container or native) on ROCm or Vulkan, replacing daily `run.py` use.

**Architecture:** `llamactl/core/runtime.py` (container ops abstraction) → `llamactl/core/lifecycle.py` (launch/stop/re-attach, owns `ServerState`) → `llamactl/core/monitor.py` (VRAM via fdinfo + `/health` polling) → `llamactl/ui/app.py` + `llamactl/ui/screens/serve.py` (Textual TUI). The other three tabs (Models/Builds/Test) show "coming in Phase N" placeholders. `python -m llamactl` with no arguments now opens the TUI.

**Tech Stack:** Python 3.11+, Textual ≥ 0.80 (new dep), tomlkit (existing), urllib.request for HTTP health checks (stdlib — no httpx), pytest-asyncio (new test dep).

---

## Prerequisites

Run before starting Task 1:

```bash
# 1. Archive the completed Phase 1 OpenSpec change (clean boundary)
openspec archive "add-llamactl-core"

# 2. Install new runtime and test dependencies
pip install "textual>=0.80" pytest-asyncio
```

After reviewing this plan and before the first commit, create the Phase 2 OpenSpec change:

```bash
openspec new "add-llamactl-serve"
```

Then write a delta spec in `openspec/changes/add-llamactl-serve/specs/llamactl/` covering
the six requirements below (runtime detection, container launch, native launch, re-attach,
health polling, Serve tab). Run `openspec validate --type change add-llamactl-serve`
before each commit.

---

## File Map

```
New files:
  llamactl/core/runtime.py          container runtime detection + ops (podman/docker)
  llamactl/core/lifecycle.py        ServerState, ServerInfo, launch/stop/find, validate
  llamactl/core/monitor.py          read_vram_kib + check_health
  llamactl/ui/__init__.py           (empty)
  llamactl/ui/app.py                LlamaCtlApp(App) with TabbedContent
  llamactl/ui/screens/__init__.py   (empty)
  llamactl/ui/screens/serve.py      full Serve tab
  tests/llamactl/test_runtime.py
  tests/llamactl/test_lifecycle.py
  tests/llamactl/test_monitor.py
  tests/llamactl/test_serve_ui.py   Textual Pilot smoke tests

Modified files:
  llamactl/__main__.py              no-args → open dashboard; migrate unchanged
```

Runtime state (gitignored — `state/` is already in `.gitignore`):

```
state/native-server.json      native server metadata (pid, model, backend, log path)
state/logs/<ts>-<id>.log      native server stdout + stderr
state/registry.toml           build artifact registry (from Phase 1)
```

---

## Task 1: `runtime.py` — Container runtime abstraction

**Files:**
- Create: `llamactl/core/runtime.py`
- Create: `tests/llamactl/test_runtime.py`

All subprocess calls accept an injectable `runner` callable so tests never need a real
container runtime.

- [ ] **Step 1: Write the failing tests**

```python
# tests/llamactl/test_runtime.py
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from llamactl.core.runtime import (
    ContainerInfo,
    container_logs_cmd,
    container_stop,
    dri_passthrough_flags,
    find_runtime,
    list_managed,
)


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _err() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 1, stdout="", stderr="error")


# ── find_runtime ──────────────────────────────────────────────────────────────

def test_find_runtime_returns_podman(monkeypatch):
    from llamactl.core import runtime as rt_mod
    monkeypatch.setattr(rt_mod.shutil, "which", lambda n: "/usr/bin/podman" if n == "podman" else None)
    assert find_runtime() == "/usr/bin/podman"


def test_find_runtime_falls_back_to_docker(monkeypatch):
    from llamactl.core import runtime as rt_mod
    monkeypatch.setattr(rt_mod.shutil, "which", lambda n: "/usr/bin/docker" if n == "docker" else None)
    assert find_runtime() == "/usr/bin/docker"


def test_find_runtime_returns_none_when_neither_present(monkeypatch):
    from llamactl.core import runtime as rt_mod
    monkeypatch.setattr(rt_mod.shutil, "which", lambda _n: None)
    assert find_runtime() is None


# ── list_managed ──────────────────────────────────────────────────────────────

PODMAN_JSON = '''[
  {
    "Names": ["llamactl-qwen3.5-9b"],
    "State": "running",
    "Labels": {
      "llamactl.managed": "1",
      "llamactl.model": "qwen3.5-9b",
      "llamactl.backend": "rocm",
      "llamactl.preset": "thinking-budgeted"
    }
  }
]'''


def test_list_managed_parses_podman_json():
    result = list_managed("/usr/bin/podman", "llamactl", runner=lambda _: _ok(PODMAN_JSON))
    assert len(result) == 1
    c = result[0]
    assert c.name == "llamactl-qwen3.5-9b"
    assert c.state == "running"
    assert c.model_id == "qwen3.5-9b"
    assert c.backend == "rocm"
    assert c.preset == "thinking-budgeted"


def test_list_managed_returns_empty_on_nonzero_exit():
    result = list_managed("/usr/bin/podman", "llamactl", runner=lambda _: _err())
    assert result == []


def test_list_managed_returns_empty_on_invalid_json():
    result = list_managed("/usr/bin/podman", "llamactl", runner=lambda _: _ok("not json"))
    assert result == []


def test_list_managed_returns_empty_on_empty_output():
    result = list_managed("/usr/bin/podman", "llamactl", runner=lambda _: _ok("[]"))
    assert result == []


def test_list_managed_passes_label_filter_to_runner():
    captured: list[list[str]] = []
    def runner(cmd):
        captured.append(cmd)
        return _ok("[]")
    list_managed("/usr/bin/podman", "llamactl", runner=runner)
    assert captured, "runner was never called"
    flat = " ".join(captured[0])
    assert "llamactl.managed=1" in flat
    assert "--format" in captured[0]


# ── container_stop ────────────────────────────────────────────────────────────

def test_container_stop_calls_stop_subcommand():
    captured: list[list[str]] = []
    container_stop("/usr/bin/podman", "llamactl-mymodel", runner=lambda cmd: (captured.append(cmd), _ok())[1])
    assert captured[0] == ["/usr/bin/podman", "stop", "llamactl-mymodel"]


# ── container_logs_cmd ────────────────────────────────────────────────────────

def test_container_logs_cmd_includes_follow_and_name():
    cmd = container_logs_cmd("/usr/bin/podman", "llamactl-mymodel")
    assert "-f" in cmd
    assert "llamactl-mymodel" in cmd
    assert cmd[0] == "/usr/bin/podman"


# ── dri_passthrough_flags ─────────────────────────────────────────────────────

def _fake_stat(gid: int):
    s = MagicMock()
    s.st_gid = gid
    return s


def test_dri_passthrough_flags_device_and_group():
    nodes = ["/dev/dri/renderD128", "/dev/dri/card0"]
    device_flags, group_flags = dri_passthrough_flags(
        _glob=lambda _pat: nodes,
        _stat=lambda _p: _fake_stat(44),
    )
    assert "--device" in device_flags
    assert "/dev/dri/renderD128:/dev/dri/renderD128" in device_flags
    assert "--group-add" in group_flags
    assert "44" in group_flags


def test_dri_passthrough_deduplicates_same_gid():
    nodes = ["/dev/dri/renderD128", "/dev/dri/card0"]
    _, group_flags = dri_passthrough_flags(
        _glob=lambda _pat: nodes,
        _stat=lambda _p: _fake_stat(44),  # both nodes share GID
    )
    assert group_flags.count("44") == 1


def test_dri_passthrough_empty_dir_returns_empty_lists():
    device_flags, group_flags = dri_passthrough_flags(
        _glob=lambda _pat: [],
        _stat=lambda _p: _fake_stat(0),
    )
    assert device_flags == []
    assert group_flags == []
```

- [ ] **Step 2: Run tests to confirm they fail (import error expected)**

```bash
pytest tests/llamactl/test_runtime.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'llamactl.core.runtime'`

- [ ] **Step 3: Write `llamactl/core/runtime.py`**

```python
"""Container runtime detection and label-based operations.

All functions that invoke subprocess accept an injectable `runner` callable
so tests never need a real container runtime installed.
"""
from __future__ import annotations

import glob as _glob_module
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


@dataclass(frozen=True)
class ContainerInfo:
    name: str
    state: str      # "running" | "exited" | …
    model_id: str
    backend: str
    preset: str     # "" when no preset


def find_runtime() -> str | None:
    """Return path to podman or docker, preferring podman. None if neither found."""
    return shutil.which("podman") or shutil.which("docker")


def list_managed(
    runtime: str,
    name_prefix: str,
    runner: Runner = _default_runner,
) -> list[ContainerInfo]:
    """Return all llamactl-managed containers (running and stopped)."""
    result = runner([
        runtime, "ps", "-a",
        "--filter", "label=llamactl.managed=1",
        "--format", "json",
    ])
    if result.returncode != 0:
        return []
    raw = (result.stdout or "").strip()
    if not raw:
        return []

    try:
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        # Docker returns newline-delimited JSON, one object per line
        try:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        except json.JSONDecodeError:
            return []

    out: list[ContainerInfo] = []
    for row in rows:
        labels = row.get("Labels", {}) or {}
        names = row.get("Names", []) or []
        name = names[0] if names else row.get("Name", "")
        if not name.startswith(name_prefix):
            continue
        out.append(ContainerInfo(
            name=name,
            state=row.get("State", ""),
            model_id=labels.get("llamactl.model", ""),
            backend=labels.get("llamactl.backend", ""),
            preset=labels.get("llamactl.preset", ""),
        ))
    return out


def container_stop(
    runtime: str,
    name: str,
    runner: Runner = _default_runner,
) -> None:
    runner([runtime, "stop", name])


def container_logs_cmd(runtime: str, name: str) -> list[str]:
    """Command to stream container logs; caller opens as subprocess.Popen."""
    return [runtime, "logs", "-f", name]


def dri_passthrough_flags(
    _glob=None,
    _stat=None,
) -> tuple[list[str], list[str]]:
    """Return (device_flags, group_flags) for /dev/dri passthrough (Vulkan).

    _glob and _stat are injectable for testing; default to the real syscalls.
    """
    glob_fn = _glob if _glob is not None else _glob_module.glob
    stat_fn = _stat if _stat is not None else os.stat

    device_flags: list[str] = []
    group_flags: list[str] = []
    seen_gids: set[int] = set()

    nodes = glob_fn("/dev/dri/renderD*") + glob_fn("/dev/dri/card*")
    for node in nodes:
        device_flags += ["--device", f"{node}:{node}"]
        gid = stat_fn(node).st_gid
        if gid not in seen_gids:
            group_flags += ["--group-add", str(gid)]
            seen_gids.add(gid)

    return device_flags, group_flags
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/llamactl/test_runtime.py -v
```

Expected: all 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/runtime.py tests/llamactl/test_runtime.py
git commit -m "feat: container runtime abstraction with injectable runner"
```

---

## Task 2: `lifecycle.py` — Server launch, stop, re-attach

**Files:**
- Create: `llamactl/core/lifecycle.py`
- Create: `tests/llamactl/test_lifecycle.py`

`lifecycle.py` owns `ServerState` (imported by `monitor.py`). It composes
`runtime.py` + `config.py` + `mapper.py`. All subprocess calls use the injectable
`runner`; the native spawner is a separate injectable so tests never start real processes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/llamactl/test_lifecycle.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from llamactl.core.config import GlobalConfig, ModelConfig
from llamactl.core.lifecycle import (
    DEFAULT_ROCM_IMAGE,
    DEFAULT_VULKAN_IMAGE,
    VALID_BACKENDS,
    VALID_MODES,
    ServerInfo,
    ServerState,
    find_running,
    launch_container,
    launch_native,
    resolve_image,
    stop_server,
    validate_global_enums,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _err(stderr: str = "oops") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 1, stdout="", stderr=stderr)


def _global_cfg(**kwargs) -> GlobalConfig:
    defaults = dict(
        default_backend="rocm", default_mode="container",
        port=8080, vram_budget_gb=11.0,
        hf_cache=Path("/tmp/hf"), llama_cache=Path("/tmp/llama"),
        name_prefix="llamactl",
    )
    return GlobalConfig(**{**defaults, **kwargs})


def _model(**kwargs) -> ModelConfig:
    defaults = dict(
        id="qwen3-9b", name="Qwen 9B", hf="org/model:file",
        settings={"ctx_size": 4096, "n_gpu_layers": -1},
        backends={}, presets={}, images={},
        path=Path("/fake/qwen3-9b.toml"),
    )
    return ModelConfig(**{**defaults, **kwargs})


# ── validate_global_enums ─────────────────────────────────────────────────────

def test_validate_rejects_invalid_backend():
    from llamactl.core.config import ConfigError
    cfg = _global_cfg(default_backend="cuda")
    with pytest.raises(ConfigError, match="default_backend"):
        validate_global_enums(cfg)


def test_validate_rejects_invalid_mode():
    from llamactl.core.config import ConfigError
    cfg = _global_cfg(default_mode="daemon")
    with pytest.raises(ConfigError, match="default_mode"):
        validate_global_enums(cfg)


def test_validate_passes_valid_values():
    validate_global_enums(_global_cfg(default_backend="rocm", default_mode="container"))
    validate_global_enums(_global_cfg(default_backend="vulkan", default_mode="native"))


# ── resolve_image ─────────────────────────────────────────────────────────────

def test_resolve_image_uses_default_when_no_model_image():
    m = _model(images={})
    assert resolve_image(m, "rocm") == DEFAULT_ROCM_IMAGE
    assert resolve_image(m, "vulkan") == DEFAULT_VULKAN_IMAGE


def test_resolve_image_prefers_model_image_over_default():
    m = _model(images={"rocm": "llama-cpp-gfx1031:b9000"})
    assert resolve_image(m, "rocm") == "llama-cpp-gfx1031:b9000"


def test_resolve_image_override_wins_over_model_image():
    m = _model(images={"rocm": "llama-cpp-gfx1031:b9000"})
    assert resolve_image(m, "rocm", override="custom:tag") == "custom:tag"


# ── launch_container ──────────────────────────────────────────────────────────

def test_launch_container_rocm_device_flags(tmp_path):
    captured: list[list[str]] = []
    runner = lambda cmd: (captured.append(cmd), _ok("abc123\n"))[1]

    launch_container(
        runtime="/usr/bin/podman",
        global_cfg=_global_cfg(),
        model=_model(),
        resolved_settings={"ctx_size": 4096, "n_gpu_layers": -1},
        backend="rocm",
        preset="",
        image=DEFAULT_ROCM_IMAGE,
        state_dir=tmp_path,
        runner=runner,
    )

    flat = " ".join(captured[0])
    assert "/dev/kfd" in flat
    assert "video" in flat
    assert "render" in flat


def test_launch_container_includes_labels(tmp_path):
    captured: list[list[str]] = []
    runner = lambda cmd: (captured.append(cmd), _ok("abc123\n"))[1]

    launch_container(
        runtime="/usr/bin/podman",
        global_cfg=_global_cfg(),
        model=_model(id="qwen3-9b"),
        resolved_settings={"ctx_size": 4096},
        backend="rocm",
        preset="thinking",
        image=DEFAULT_ROCM_IMAGE,
        state_dir=tmp_path,
        runner=runner,
    )

    flat = " ".join(captured[0])
    assert "llamactl.managed=1" in flat
    assert "llamactl.model=qwen3-9b" in flat
    assert "llamactl.backend=rocm" in flat
    assert "llamactl.preset=thinking" in flat


def test_launch_container_raises_on_nonzero_exit(tmp_path):
    runner = lambda _cmd: _err(stderr="image not found")
    with pytest.raises(RuntimeError, match="image not found"):
        launch_container(
            runtime="/usr/bin/podman",
            global_cfg=_global_cfg(),
            model=_model(),
            resolved_settings={},
            backend="rocm",
            preset="",
            image=DEFAULT_ROCM_IMAGE,
            state_dir=tmp_path,
            runner=runner,
        )


def test_launch_container_vulkan_calls_dri_passthrough(tmp_path):
    captured: list[list[str]] = []
    runner = lambda cmd: (captured.append(cmd), _ok("abc123\n"))[1]
    fake_dri = lambda **_: (["--device", "/dev/dri/renderD128:/dev/dri/renderD128"], ["--group-add", "44"])

    launch_container(
        runtime="/usr/bin/podman",
        global_cfg=_global_cfg(),
        model=_model(),
        resolved_settings={"ctx_size": 4096},
        backend="vulkan",
        preset="",
        image=DEFAULT_VULKAN_IMAGE,
        state_dir=tmp_path,
        runner=runner,
        dri_flags_fn=fake_dri,
    )

    flat = " ".join(captured[0])
    assert "/dev/dri/renderD128" in flat
    assert "44" in flat


def test_launch_container_returns_server_info(tmp_path):
    runner = lambda _cmd: _ok("abc123\n")
    info = launch_container(
        runtime="/usr/bin/podman",
        global_cfg=_global_cfg(port=8080, name_prefix="llamactl"),
        model=_model(id="qwen3-9b"),
        resolved_settings={"ctx_size": 4096},
        backend="rocm",
        preset="",
        image=DEFAULT_ROCM_IMAGE,
        state_dir=tmp_path,
        runner=runner,
    )
    assert info.mode == "container"
    assert info.container_name == "llamactl-qwen3-9b"
    assert info.port == 8080
    assert info.model_id == "qwen3-9b"


# ── launch_native ─────────────────────────────────────────────────────────────

def test_launch_native_rocm_sets_hsa_env(tmp_path):
    spawned: list[tuple] = []
    def fake_spawner(cmd, env_extra, log_path):
        spawned.append((cmd, env_extra, log_path))
        return 12345

    launch_native(
        global_cfg=_global_cfg(),
        model=_model(),
        resolved_settings={"ctx_size": 4096},
        backend="rocm",
        preset="",
        binary="/usr/bin/llama-server",
        state_dir=tmp_path,
        spawner=fake_spawner,
    )

    _, env_extra, _ = spawned[0]
    assert env_extra.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"


def test_launch_native_vulkan_does_not_set_hsa_env(tmp_path):
    spawned: list[tuple] = []
    def fake_spawner(cmd, env_extra, log_path):
        spawned.append((cmd, env_extra, log_path))
        return 12345

    launch_native(
        global_cfg=_global_cfg(),
        model=_model(),
        resolved_settings={"ctx_size": 4096},
        backend="vulkan",
        preset="",
        binary="/usr/bin/llama-server",
        state_dir=tmp_path,
        spawner=fake_spawner,
    )

    _, env_extra, _ = spawned[0]
    assert "HSA_OVERRIDE_GFX_VERSION" not in env_extra


def test_launch_native_writes_meta_json(tmp_path):
    def fake_spawner(cmd, env_extra, log_path):
        return 99999

    launch_native(
        global_cfg=_global_cfg(port=8080),
        model=_model(id="qwen3-9b"),
        resolved_settings={"ctx_size": 4096},
        backend="rocm",
        preset="thinking",
        binary="/usr/bin/llama-server",
        state_dir=tmp_path,
        spawner=fake_spawner,
    )

    meta_path = tmp_path / "native-server.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["pid"] == 99999
    assert meta["model_id"] == "qwen3-9b"
    assert meta["backend"] == "rocm"
    assert meta["preset"] == "thinking"
    assert meta["port"] == 8080


def test_launch_native_returns_server_info(tmp_path):
    def fake_spawner(cmd, env_extra, log_path):
        return 99999

    info = launch_native(
        global_cfg=_global_cfg(),
        model=_model(id="qwen3-9b"),
        resolved_settings={},
        backend="rocm",
        preset="",
        binary="/usr/bin/llama-server",
        state_dir=tmp_path,
        spawner=fake_spawner,
    )

    assert info.mode == "native"
    assert info.pid == 99999
    assert info.model_id == "qwen3-9b"
    assert info.log_path is not None


# ── stop_server ───────────────────────────────────────────────────────────────

def test_stop_server_container_calls_runtime_stop():
    captured: list[list[str]] = []
    runner = lambda cmd: (captured.append(cmd), subprocess.CompletedProcess(cmd, 0))[1]

    info = ServerInfo(
        model_id="qwen3-9b", backend="rocm", preset="", mode="container",
        host="0.0.0.0", port=8080, started_at="2026-06-16T14:00:00",
        container_name="llamactl-qwen3-9b",
    )
    stop_server(info, runtime="/usr/bin/podman", runner=runner)
    assert captured[0] == ["/usr/bin/podman", "stop", "llamactl-qwen3-9b"]


def test_stop_server_native_sends_sigterm(tmp_path, monkeypatch):
    killed: list[tuple] = []
    import signal as _signal
    monkeypatch.setattr("llamactl.core.lifecycle.os.kill",
                        lambda pid, sig: killed.append((pid, sig)))

    info = ServerInfo(
        model_id="qwen3-9b", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="2026-06-16T14:00:00",
        pid=54321,
    )
    stop_server(info)
    assert killed[0] == (54321, _signal.SIGTERM)


# ── find_running ──────────────────────────────────────────────────────────────

RUNNING_CONTAINER_JSON = '''[
  {
    "Names": ["llamactl-qwen3-9b"],
    "State": "running",
    "Labels": {
      "llamactl.managed": "1",
      "llamactl.model": "qwen3-9b",
      "llamactl.backend": "rocm",
      "llamactl.preset": ""
    }
  }
]'''


def test_find_running_from_container(tmp_path):
    runner = lambda _cmd: subprocess.CompletedProcess([], 0, stdout=RUNNING_CONTAINER_JSON, stderr="")
    info = find_running(_global_cfg(), tmp_path, runtime="/usr/bin/podman", runner=runner)
    assert info is not None
    assert info.mode == "container"
    assert info.model_id == "qwen3-9b"
    assert info.container_name == "llamactl-qwen3-9b"


def test_find_running_returns_none_when_no_state(tmp_path):
    runner = lambda _cmd: subprocess.CompletedProcess([], 0, stdout="[]", stderr="")
    info = find_running(_global_cfg(), tmp_path, runtime="/usr/bin/podman", runner=runner)
    assert info is None


def test_find_running_native_from_pidfile(tmp_path, monkeypatch):
    meta = {
        "pid": 77777, "model_id": "qwen3-9b", "backend": "vulkan",
        "preset": "", "host": "0.0.0.0", "port": 8080,
        "log_path": str(tmp_path / "logs/test.log"),
        "started_at": "2026-06-16T14:00:00",
    }
    (tmp_path / "native-server.json").write_text(json.dumps(meta))
    # No container found
    runner = lambda _cmd: subprocess.CompletedProcess([], 0, stdout="[]", stderr="")
    # Fake that the process is alive
    monkeypatch.setattr("llamactl.core.lifecycle._is_pid_alive", lambda pid: pid == 77777)

    info = find_running(_global_cfg(), tmp_path, runtime="/usr/bin/podman", runner=runner)
    assert info is not None
    assert info.mode == "native"
    assert info.pid == 77777
    assert info.backend == "vulkan"
```

- [ ] **Step 2: Run tests to confirm they fail (import error expected)**

```bash
pytest tests/llamactl/test_lifecycle.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'llamactl.core.lifecycle'`

- [ ] **Step 3: Write `llamactl/core/lifecycle.py`**

```python
"""Server lifecycle: launch, stop, re-attach for container and native modes.

ServerState is defined here and imported by monitor.py — no circular deps.
All subprocess calls accept injectable `runner`/`spawner` for test isolation.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from llamactl.core.config import ConfigError, GlobalConfig, ModelConfig
from llamactl.core.mapper import build_server_argv
from llamactl.core.runtime import (
    Runner,
    _default_runner,
    container_stop,
    dri_passthrough_flags,
    find_runtime,
    list_managed,
)

# Callable that spawns a detached native process and returns its PID.
# Args: (cmd, env_extra, log_path)
NativeSpawner = Callable[[list[str], dict[str, str], Path], int]

VALID_BACKENDS: frozenset[str] = frozenset({"rocm", "vulkan"})
VALID_MODES: frozenset[str] = frozenset({"container", "native"})

DEFAULT_ROCM_IMAGE = "llama-cpp-gfx1031:latest"
DEFAULT_VULKAN_IMAGE = "llama-cpp-vulkan:latest"


class ServerState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    EXITED = "exited"


@dataclass(frozen=True)
class ServerInfo:
    model_id: str
    backend: str
    preset: str
    mode: str           # "container" | "native"
    host: str
    port: int
    started_at: str     # ISO-8601; "" when re-attached from running container
    container_name: str | None = None
    log_path: str | None = None
    pid: int | None = None


def validate_global_enums(cfg: GlobalConfig) -> None:
    """Raise ConfigError if default_backend or default_mode are not legal values."""
    if cfg.default_backend not in VALID_BACKENDS:
        raise ConfigError(
            f"default_backend '{cfg.default_backend}' must be one of: "
            f"{', '.join(sorted(VALID_BACKENDS))}"
        )
    if cfg.default_mode not in VALID_MODES:
        raise ConfigError(
            f"default_mode '{cfg.default_mode}' must be one of: "
            f"{', '.join(sorted(VALID_MODES))}"
        )


def resolve_image(
    model: ModelConfig, backend: str, override: str | None = None
) -> str:
    """built-in default → model images.{backend} → explicit override."""
    default = DEFAULT_ROCM_IMAGE if backend == "rocm" else DEFAULT_VULKAN_IMAGE
    return override or model.images.get(backend) or default


def _container_name(name_prefix: str, model_id: str) -> str:
    return f"{name_prefix}-{model_id}"


def _default_spawner(
    cmd: list[str], env_extra: dict[str, str], log_path: Path
) -> int:
    env = {**os.environ, **env_extra}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    return proc.pid


def _is_pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"llama-server" in f.read()
    except OSError:
        return False


def launch_container(
    runtime: str,
    global_cfg: GlobalConfig,
    model: ModelConfig,
    resolved_settings: dict[str, Any],
    backend: str,
    preset: str,
    image: str,
    state_dir: Path,
    runner: Runner = _default_runner,
    dri_flags_fn=dri_passthrough_flags,
) -> ServerInfo:
    name = _container_name(global_cfg.name_prefix, model.id)

    label_flags = [
        "--label", "llamactl.managed=1",
        "--label", f"llamactl.model={model.id}",
        "--label", f"llamactl.backend={backend}",
        "--label", f"llamactl.preset={preset}",
    ]

    if backend == "rocm":
        device_flags = ["--device", "/dev/kfd", "--device", "/dev/dri"]
        group_flags = ["--group-add", "video", "--group-add", "render"]
    else:
        device_flags, group_flags = dri_flags_fn()

    volume_flags = [
        "-v", f"{global_cfg.hf_cache}:/root/.cache/huggingface",
        "-v", f"{global_cfg.llama_cache}:/root/.cache/llama.cpp",
    ]

    argv = build_server_argv(model.hf, resolved_settings, "0.0.0.0", global_cfg.port)

    cmd = [
        runtime, "run", "-d",
        "--name", name,
        *label_flags,
        *device_flags,
        *group_flags,
        *volume_flags,
        "-p", f"{global_cfg.port}:{global_cfg.port}",
        image,
        *argv,
    ]

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = runner(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Container launch failed (exit {result.returncode}):\n{result.stderr}"
        )

    return ServerInfo(
        model_id=model.id,
        backend=backend,
        preset=preset,
        mode="container",
        host="0.0.0.0",
        port=global_cfg.port,
        started_at=started_at,
        container_name=name,
    )


def launch_native(
    global_cfg: GlobalConfig,
    model: ModelConfig,
    resolved_settings: dict[str, Any],
    backend: str,
    preset: str,
    binary: str,
    state_dir: Path,
    spawner: NativeSpawner = _default_spawner,
) -> ServerInfo:
    argv = build_server_argv(model.hf, resolved_settings, "0.0.0.0", global_cfg.port)
    cmd = [binary, *argv]

    env_extra: dict[str, str] = {}
    if backend == "rocm":
        env_extra["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_name = started_at.replace(":", "-") + f"-{model.id}.log"
    log_path = state_dir / "logs" / log_name

    pid = spawner(cmd, env_extra, log_path)

    meta = {
        "pid": pid,
        "model_id": model.id,
        "backend": backend,
        "preset": preset,
        "host": "0.0.0.0",
        "port": global_cfg.port,
        "log_path": str(log_path),
        "started_at": started_at,
    }
    meta_path = state_dir / "native-server.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ServerInfo(
        model_id=model.id,
        backend=backend,
        preset=preset,
        mode="native",
        host="0.0.0.0",
        port=global_cfg.port,
        started_at=started_at,
        log_path=str(log_path),
        pid=pid,
    )


def stop_server(
    info: ServerInfo,
    runtime: str | None = None,
    runner: Runner = _default_runner,
) -> None:
    if info.mode == "container" and info.container_name:
        rt = runtime or find_runtime()
        if rt:
            container_stop(rt, info.container_name, runner)
    elif info.mode == "native" and info.pid is not None:
        try:
            os.kill(info.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def find_running(
    global_cfg: GlobalConfig,
    state_dir: Path,
    runtime: str | None = None,
    runner: Runner = _default_runner,
) -> ServerInfo | None:
    """Re-attach to a running server: checks containers first, then native pidfile."""
    rt = runtime or find_runtime()
    if rt:
        containers = list_managed(rt, global_cfg.name_prefix, runner)
        running = [c for c in containers if c.state == "running"]
        if running:
            c = running[0]
            return ServerInfo(
                model_id=c.model_id,
                backend=c.backend,
                preset=c.preset,
                mode="container",
                host="0.0.0.0",
                port=global_cfg.port,
                started_at="",
                container_name=c.name,
            )

    meta_path = state_dir / "native-server.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pid = int(meta["pid"])
            if _is_pid_alive(pid):
                return ServerInfo(
                    model_id=meta["model_id"],
                    backend=meta["backend"],
                    preset=meta.get("preset", ""),
                    mode="native",
                    host=meta.get("host", "0.0.0.0"),
                    port=meta.get("port", global_cfg.port),
                    started_at=meta.get("started_at", ""),
                    log_path=meta.get("log_path"),
                    pid=pid,
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    return None
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/llamactl/test_lifecycle.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/lifecycle.py tests/llamactl/test_lifecycle.py
git commit -m "feat: server lifecycle — launch/stop/find for container and native modes"
```

---

## Task 3: `monitor.py` — VRAM reading and health polling

**Files:**
- Create: `llamactl/core/monitor.py`
- Create: `tests/llamactl/test_monitor.py`

`monitor.py` imports `ServerState` from `lifecycle.py`. The fdinfo reader is a focused
port of `vram_inspect.py`: memory fields only, deduplication by `drm-client-id`, no
printing. `fdinfo_root` is injectable so tests provide fake `/proc` trees under `tmp_path`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/llamactl/test_monitor.py
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llamactl.core.lifecycle import ServerState
from llamactl.core.monitor import check_health, read_vram_kib


# ── read_vram_kib ─────────────────────────────────────────────────────────────

def _write_fdinfo(proc_dir: Path, fd: str, client_id: str, fields: dict[str, int]) -> None:
    fdinfo_dir = proc_dir / "fdinfo"
    fdinfo_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"drm-client-id:\t{client_id}\n"]
    for key, val in fields.items():
        lines.append(f"{key}:\t{val} KiB\n")
    (fdinfo_dir / fd).write_text("".join(lines))


def test_read_vram_kib_sums_memory_fields(tmp_path):
    pid_dir = tmp_path / "12345"
    _write_fdinfo(pid_dir, "10", "client-1", {
        "drm-memory-vram": 2097152,
        "drm-memory-gtt":   524288,
    })
    total = read_vram_kib("12345", fdinfo_root=str(tmp_path))
    assert total == 2097152 + 524288


def test_read_vram_kib_deduplicates_same_client_id(tmp_path):
    pid_dir = tmp_path / "12345"
    # Two fds sharing the same drm-client-id: should count once
    _write_fdinfo(pid_dir, "10", "shared-client", {"drm-memory-vram": 1000000})
    _write_fdinfo(pid_dir, "11", "shared-client", {"drm-memory-vram": 1000000})
    total = read_vram_kib("12345", fdinfo_root=str(tmp_path))
    assert total == 1000000  # not 2000000


def test_read_vram_kib_ignores_non_memory_fields(tmp_path):
    pid_dir = tmp_path / "12345"
    _write_fdinfo(pid_dir, "10", "c1", {
        "drm-memory-vram": 500000,
        "drm-engine-gfx": 999,  # not a memory field — must be ignored
    })
    # drm-engine-gfx has no "memory" in its name; only drm-memory-* is summed
    total = read_vram_kib("12345", fdinfo_root=str(tmp_path))
    assert total == 500000


def test_read_vram_kib_returns_zero_for_missing_pid(tmp_path):
    total = read_vram_kib("99999", fdinfo_root=str(tmp_path))
    assert total == 0


# ── check_health ──────────────────────────────────────────────────────────────

def _mock_http_200():
    resp = MagicMock()
    resp.status = 200
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_check_health_returns_ready_on_200():
    with patch("urllib.request.urlopen", return_value=_mock_http_200()):
        state = check_health(8080)
    assert state == ServerState.READY


def test_check_health_returns_loading_on_503():
    err = urllib.error.HTTPError(url=None, code=503, msg="Loading", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=err):
        state = check_health(8080)
    assert state == ServerState.LOADING


def test_check_health_returns_starting_on_connection_refused():
    err = urllib.error.URLError(reason="Connection refused")
    with patch("urllib.request.urlopen", side_effect=err):
        state = check_health(8080)
    assert state == ServerState.STARTING


def test_check_health_returns_starting_on_timeout():
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        state = check_health(8080)
    assert state == ServerState.STARTING


def test_check_health_returns_unhealthy_on_other_http_error():
    err = urllib.error.HTTPError(url=None, code=500, msg="Internal Error", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=err):
        state = check_health(8080)
    assert state == ServerState.UNHEALTHY
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/llamactl/test_monitor.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'llamactl.core.monitor'`

- [ ] **Step 3: Write `llamactl/core/monitor.py`**

```python
"""VRAM inspection and health polling for a running llama-server.

read_vram_kib: ports the memory-only subset of vram_inspect.py's fdinfo reader.
check_health:  polls /health and maps HTTP status to ServerState.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

from llamactl.core.lifecycle import ServerState


def _parse_fdinfo_file(path: str) -> tuple[str | None, dict[str, int]]:
    """Return (drm_client_id, {memory_field: kib_value}) for one fdinfo file."""
    client_id: str | None = None
    fields: dict[str, int] = {}
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("drm-"):
                    continue
                key, sep, raw = line.partition(":")
                if not sep:
                    continue
                key = key.strip()
                raw = raw.strip()
                if key == "drm-client-id":
                    client_id = raw
                    continue
                if "memory" not in key:
                    continue
                parts = raw.split(None, 1)
                if not parts:
                    continue
                try:
                    fields[key] = int(parts[0])
                except ValueError:
                    continue
    except OSError:
        pass
    return client_id, fields


def read_vram_kib(pid: str, fdinfo_root: str = "/proc") -> int:
    """Sum all drm-*-memory-* KiB fields across unique DRM clients for PID.

    Returns 0 if the PID doesn't exist or has no DRM file descriptors.
    """
    fdinfo_dir = f"{fdinfo_root}/{pid}/fdinfo"
    try:
        fd_names = os.listdir(fdinfo_dir)
    except OSError:
        return 0

    seen_clients: dict[str, dict[str, int]] = {}
    anon_index = 0
    for fd_name in fd_names:
        client_id, fields = _parse_fdinfo_file(f"{fdinfo_dir}/{fd_name}")
        if not fields:
            continue
        key = client_id if client_id is not None else f"_anon_{anon_index}"
        if client_id is None:
            anon_index += 1
        if key not in seen_clients:
            seen_clients[key] = fields

    total = 0
    for fields in seen_clients.values():
        total += sum(fields.values())
    return total


def check_health(port: int, timeout: float = 2.0) -> ServerState:
    """Poll http://127.0.0.1:{port}/health and return the derived ServerState."""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return ServerState.READY
            return ServerState.UNHEALTHY
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return ServerState.LOADING
        return ServerState.UNHEALTHY
    except (urllib.error.URLError, OSError, TimeoutError):
        return ServerState.STARTING
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/llamactl/test_monitor.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
pytest tests/llamactl/ -v --ignore=tests/llamactl/test_parity.py
```

Expected: all tests pass (parity test is skipped — it needs the container image).

- [ ] **Step 6: Commit**

```bash
git add llamactl/core/monitor.py tests/llamactl/test_monitor.py
git commit -m "feat: VRAM fdinfo reader and /health poller for live server monitoring"
```

---

## Task 4: App skeleton — Textual wiring + `__main__.py` update

**Files:**
- Create: `llamactl/ui/__init__.py`
- Create: `llamactl/ui/screens/__init__.py`
- Create: `llamactl/ui/app.py`
- Create: `llamactl/ui/screens/serve.py` (skeleton — placeholder content)
- Modify: `llamactl/__main__.py`
- Create: `tests/llamactl/test_serve_ui.py`

- [ ] **Step 1: Write the failing smoke tests**

```python
# tests/llamactl/test_serve_ui.py
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Minimal repo layout: empty configs/models dir so load_all returns no models."""
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "state").mkdir()
    # Minimal global config (all defaults)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    return tmp_path


@pytest.mark.asyncio
async def test_app_boots_without_error(repo_root: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    app = LlamaCtlApp(repo_root=repo_root)
    async with app.run_test(headless=True) as pilot:
        assert pilot.app.is_running


@pytest.mark.asyncio
async def test_serve_tab_is_active_on_startup(repo_root: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import TabbedContent
    app = LlamaCtlApp(repo_root=repo_root)
    async with app.run_test(headless=True) as pilot:
        tabs = app.query_one(TabbedContent)
        assert tabs.active == "serve"


@pytest.mark.asyncio
async def test_tab_switching_works(repo_root: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    app = LlamaCtlApp(repo_root=repo_root)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")   # cycle to next tab
        # App must still be running after tab switch (no crash)
        assert pilot.app.is_running
```

- [ ] **Step 2: Run tests to confirm they fail (import error)**

```bash
pytest tests/llamactl/test_serve_ui.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'llamactl.ui'`

- [ ] **Step 3: Create empty `__init__.py` files**

```bash
touch llamactl/ui/__init__.py
touch llamactl/ui/screens/__init__.py
```

- [ ] **Step 4: Write `llamactl/ui/screens/serve.py` (skeleton)**

```python
"""Serve tab — Phase 2 implementation. Full layout added in Task 5."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class ServeScreen(Widget):
    DEFAULT_CSS = "ServeScreen { height: 1fr; padding: 1 2; }"

    def compose(self) -> ComposeResult:
        yield Static("Serve tab loading…", id="serve-placeholder")
```

- [ ] **Step 5: Write `llamactl/ui/app.py`**

```python
"""LlamaCtlApp — Textual TUI entry point.

Accepts repo_root so tests can pass a tmp_path instead of the real repo.
"""
from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from llamactl.core.config import GlobalConfig, load_all, load_global
from llamactl.core.registry import load_registry
from llamactl.ui.screens.serve import ServeScreen


class LlamaCtlApp(App):
    TITLE = "llamactl"
    SUB_TITLE = "llama.cpp on ROCm / Vulkan"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("?", "help", "Help", show=False),
    ]

    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        self._repo_root = repo_root
        self._config_dir = repo_root / "configs"
        self._state_dir = repo_root / "state"

    def on_mount(self) -> None:
        self._global_cfg = load_global(self._config_dir / "llamactl.toml")
        self._models, self._model_errors = load_all(self._config_dir / "models")
        try:
            self._artifacts = load_registry(self._state_dir / "registry.toml")
        except Exception:
            self._artifacts = []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="serve"):
            with TabPane("Serve", id="serve"):
                yield ServeScreen()
            with TabPane("Models", id="models"):
                yield Static("Models editor — coming in Phase 4")
            with TabPane("Builds", id="builds"):
                yield Static("Build manager — coming in Phase 3")
            with TabPane("Test", id="test"):
                yield Static("OOM boundary test — coming in Phase 5")
        yield Footer()

    def action_quit(self) -> None:
        if self.query_one(ServeScreen).has_running_server:
            self.notify(
                "Server is still running. Stop it first or close this window.",
                severity="warning",
                timeout=4,
            )
        self.exit()
```

- [ ] **Step 6: Update `llamactl/__main__.py`**

Read the existing file first, then apply this replacement:

```python
"""CLI entry point. No-args → TUI dashboard. 'migrate' sub-command → one-shot import."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llamactl.core.migrate import migrate

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llamactl")
    sub = parser.add_subparsers(dest="command")
    mig = sub.add_parser(
        "migrate",
        help="One-shot import of models/*.json into configs/models/*.toml",
    )
    mig.add_argument("--src", type=Path, default=REPO_ROOT / "models")
    mig.add_argument("--dest", type=Path, default=REPO_ROOT / "configs" / "models")
    mig.add_argument("--force", action="store_true", help="Overwrite existing .toml files")

    args = parser.parse_args(argv)

    if args.command == "migrate":
        failed = False
        for dest, status, warnings in migrate(args.src, args.dest, force=args.force):
            print(f"{status:>8}  {dest}")
            for w in warnings:
                print(f"          WARNING: {w}")
            failed = failed or status == "failed"
        return 1 if failed else 0

    # No sub-command → open dashboard
    from llamactl.ui.app import LlamaCtlApp
    LlamaCtlApp(repo_root=REPO_ROOT).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Add `has_running_server` property stub to `ServeScreen`**

`app.py` references `ServeScreen.has_running_server`. Add it to the skeleton so the
app boots without `AttributeError`:

In `llamactl/ui/screens/serve.py`, add inside `ServeScreen`:
```python
    @property
    def has_running_server(self) -> bool:
        return False  # Task 6 sets this from lifecycle state
```

- [ ] **Step 8: Run the smoke tests**

```bash
pytest tests/llamactl/test_serve_ui.py -v
```

Expected: all 3 smoke tests pass.

- [ ] **Step 9: Commit**

```bash
git add llamactl/ui/__init__.py llamactl/ui/screens/__init__.py \
        llamactl/ui/app.py llamactl/ui/screens/serve.py \
        llamactl/__main__.py tests/llamactl/test_serve_ui.py
git commit -m "feat: Textual app skeleton with tab navigation and dashboard entry point"
```

---

## Task 5: Serve tab — full layout and launch form

**Files:**
- Modify: `llamactl/ui/screens/serve.py` (replace skeleton with full layout)
- Modify: `tests/llamactl/test_serve_ui.py` (add form tests)

The Serve tab layout:
- Status header (state badge, model, backend, mode, port, uptime)
- VRAM gauge (ProgressBar vs `vram_budget_gb`)
- Launch form (model selector, backend selector, mode selector, preset selector,
  optional image/binary override, argv preview)
- Log pane (RichLog, scrollable)
- Action buttons: Launch, Stop, Copy argv

- [ ] **Step 1: Add form tests to `tests/llamactl/test_serve_ui.py`**

Append these tests to the existing file:

```python
@pytest.mark.asyncio
async def test_model_selector_shows_available_models(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp

    # Write one model TOML so load_all finds it
    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "test-model.toml").write_text(
        'name = "Test Model"\nhf = "org/model:file"\n\n[settings]\nctx_size = 4096\n'
    )
    (tmp_path / "state").mkdir(exist_ok=True)

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Select
        model_select = app.query_one("#model-select", Select)
        # The select should have at least one option (our test model)
        assert model_select._options  # non-empty options list


@pytest.mark.asyncio
async def test_argv_preview_visible_after_model_selection(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp

    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "test-model.toml").write_text(
        'name = "Test Model"\nhf = "org/model:file"\n\n[settings]\nctx_size = 4096\n'
    )
    (tmp_path / "state").mkdir(exist_ok=True)

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Select, Static
        model_select = app.query_one("#model-select", Select)
        # Select the first model value
        await pilot.click("#model-select")
        await pilot.press("enter")
        # Argv preview widget must be present and contain something
        preview = app.query_one("#argv-preview", Static)
        assert preview is not None
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/llamactl/test_serve_ui.py::test_model_selector_shows_available_models -v 2>&1 | tail -5
```

Expected: `textual.css.query.NoMatches: No nodes match '#model-select'`

- [ ] **Step 3: Replace `llamactl/ui/screens/serve.py` with the full layout**

```python
"""Serve tab — launch, monitor, and stop llama-server from the TUI."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from llamactl.core.config import ModelConfig, resolve_settings
from llamactl.core.lifecycle import ServerInfo, ServerState, resolve_image
from llamactl.core.mapper import build_server_argv

if TYPE_CHECKING:
    from llamactl.ui.app import LlamaCtlApp


class _StatusHeader(Widget):
    DEFAULT_CSS = """
    _StatusHeader {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
    }
    ._state-badge { width: 12; content-align: center middle; }
    ._state-badge.stopped  { color: $text-muted; }
    ._state-badge.starting { color: $warning; }
    ._state-badge.loading  { color: $warning; }
    ._state-badge.ready    { color: $success; }
    ._state-badge.unhealthy { color: $error; }
    ._state-badge.exited   { color: $error; }
    ._meta { width: 1fr; content-align: left middle; }
    """

    state: reactive[ServerState] = reactive(ServerState.STOPPED)
    info: reactive[ServerInfo | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Static("● STOPPED", id="state-badge", classes="_state-badge stopped")
        yield Static("No server running", id="server-meta", classes="_meta")

    def watch_state(self, state: ServerState) -> None:
        badge = self.query_one("#state-badge", Static)
        badge.update(f"● {state.value.upper()}")
        badge.set_class(True, state.value)
        for other in ServerState:
            if other != state:
                badge.set_class(False, other.value)

    def watch_info(self, info: ServerInfo | None) -> None:
        meta = self.query_one("#server-meta", Static)
        if info is None:
            meta.update("No server running")
        else:
            preset_str = f"  preset={info.preset}" if info.preset else ""
            meta.update(
                f"{info.model_id}  backend={info.backend}  "
                f"mode={info.mode}  port={info.port}{preset_str}"
            )


class _VramGauge(Widget):
    DEFAULT_CSS = """
    _VramGauge { height: 3; padding: 0 2; }
    _VramGauge Label { width: 20; }
    _VramGauge ProgressBar { width: 1fr; }
    ._vram-over ProgressBar Bar { color: $error; }
    """

    vram_kib: reactive[int] = reactive(0)
    budget_kib: reactive[int] = reactive(11 * 1024 * 1024)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("VRAM")
            yield ProgressBar(total=self.budget_kib, show_eta=False, id="vram-bar")
            yield Static("0.0 / 11.0 GiB", id="vram-label")

    def watch_vram_kib(self, kib: int) -> None:
        bar = self.query_one("#vram-bar", ProgressBar)
        bar.update(total=self.budget_kib, progress=kib)
        used_gib = kib / (1024 ** 2)
        budget_gib = self.budget_kib / (1024 ** 2)
        self.query_one("#vram-label", Static).update(
            f"{used_gib:.1f} / {budget_gib:.1f} GiB"
        )
        self.set_class(kib > self.budget_kib, "_vram-over")


class _LaunchForm(Widget):
    DEFAULT_CSS = """
    _LaunchForm { height: auto; padding: 1 2; border: tall $panel; }
    _LaunchForm .form-row { height: 3; layout: horizontal; }
    _LaunchForm Label { width: 12; content-align: right middle; padding-right: 1; }
    _LaunchForm Select { width: 30; }
    _LaunchForm Button { margin: 0 1; }
    #argv-preview { height: 4; background: $surface; padding: 1; margin-top: 1; }
    """

    def __init__(
        self,
        models: list[ModelConfig],
        model_errors: dict[Path, str],
    ) -> None:
        super().__init__()
        self._models = models
        self._model_errors = model_errors

    def compose(self) -> ComposeResult:
        model_opts: list[tuple[str, str]] = [
            (m.name, m.id) for m in self._models
        ]
        if self._model_errors:
            model_opts.append((f"⚠ {len(self._model_errors)} error(s)", "__errors__"))

        yield Label("Model")
        yield Select(
            options=model_opts,
            id="model-select",
            allow_blank=not model_opts,
        )
        with Horizontal(classes="form-row"):
            yield Label("Backend")
            yield Select(
                options=[("ROCm", "rocm"), ("Vulkan", "vulkan")],
                id="backend-select",
                value="rocm",
            )
        with Horizontal(classes="form-row"):
            yield Label("Mode")
            yield Select(
                options=[("Container", "container"), ("Native", "native")],
                id="mode-select",
                value="container",
            )
        with Horizontal(classes="form-row"):
            yield Label("Preset")
            yield Select(options=[], id="preset-select", allow_blank=True)
        with Horizontal(classes="form-row"):
            yield Button("Launch", id="btn-launch", variant="success")
            yield Button("Stop",   id="btn-stop",   variant="error",   disabled=True)
            yield Button("Copy argv", id="btn-copy-argv", variant="default")
        yield Static("(select a model to preview the launch command)", id="argv-preview")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("model-select", "backend-select", "preset-select"):
            self._refresh_preset_options()
            self._refresh_argv_preview()

    def _selected_model(self) -> ModelConfig | None:
        model_id = self.query_one("#model-select", Select).value
        if not model_id or model_id == Select.BLANK or model_id == "__errors__":
            return None
        return next((m for m in self._models if m.id == model_id), None)

    def _selected_backend(self) -> str:
        return str(self.query_one("#backend-select", Select).value or "rocm")

    def _selected_preset(self) -> str | None:
        v = self.query_one("#preset-select", Select).value
        return str(v) if v and v != Select.BLANK else None

    def _refresh_preset_options(self) -> None:
        model = self._selected_model()
        preset_select = self.query_one("#preset-select", Select)
        if model and model.presets:
            preset_select.set_options([(name, name) for name in model.presets])
        else:
            preset_select.set_options([])

    def _refresh_argv_preview(self) -> None:
        model = self._selected_model()
        preview = self.query_one("#argv-preview", Static)
        if model is None:
            preview.update("(select a model to preview the launch command)")
            return
        backend = self._selected_backend()
        preset = self._selected_preset()
        try:
            settings = resolve_settings(model, preset, backend, {})
        except Exception as exc:
            preview.update(f"[red]Config error: {exc}[/red]")
            return
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        host = "0.0.0.0"
        port = app._global_cfg.port
        argv = build_server_argv(model.hf, settings, host, port)
        image = resolve_image(model, backend)
        preview.update(f"[dim]image:[/dim] {image}\n[dim]argv:[/dim]  {' '.join(argv)}")

    def get_launch_params(self) -> tuple[ModelConfig | None, str, str | None]:
        """Return (model, backend, preset) for the current form selection."""
        return self._selected_model(), self._selected_backend(), self._selected_preset()


class ServeScreen(Widget):
    DEFAULT_CSS = """
    ServeScreen { height: 1fr; layout: vertical; }
    _LogPane { height: 1fr; border: tall $panel; padding: 0 1; }
    """

    @property
    def has_running_server(self) -> bool:
        return self._server_info is not None

    def __init__(self) -> None:
        super().__init__()
        self._server_info: ServerInfo | None = None
        self._server_state: ServerState = ServerState.STOPPED

    def compose(self) -> ComposeResult:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        yield _StatusHeader()
        yield _VramGauge()
        yield _LaunchForm(
            models=getattr(app, "_models", []),
            model_errors=getattr(app, "_model_errors", {}),
        )
        yield RichLog(highlight=True, markup=True, id="log-pane")
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/llamactl/test_serve_ui.py -v
```

Expected: all 5 tests pass (3 from Task 4 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/serve.py tests/llamactl/test_serve_ui.py
git commit -m "feat: Serve tab layout — status header, VRAM gauge, launch form, log pane"
```

---

## Task 6: Wire launch/stop/VRAM/logs — make the Serve tab live

**Files:**
- Modify: `llamactl/ui/screens/serve.py` (add action handlers, background workers, re-attach on mount)
- Modify: `tests/llamactl/test_serve_ui.py` (add lifecycle integration tests)

- [ ] **Step 1: Add lifecycle integration tests**

Append to `tests/llamactl/test_serve_ui.py`:

```python
@pytest.mark.asyncio
async def test_serve_screen_reattaches_on_mount(tmp_path: Path, monkeypatch) -> None:
    """find_running is called on mount; if a server is already up the status updates."""
    from llamactl.core.lifecycle import ServerInfo, ServerState
    from llamactl.ui.app import LlamaCtlApp

    fake_info = ServerInfo(
        model_id="qwen3-9b", backend="rocm", preset="",
        mode="container", host="0.0.0.0", port=8080,
        started_at="2026-06-16T14:00:00",
        container_name="llamactl-qwen3-9b",
    )

    import llamactl.core.lifecycle as lc_mod
    monkeypatch.setattr(lc_mod, "find_running", lambda *_a, **_kw: fake_info)

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\n'
        'port = 8080\nvram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        from llamactl.ui.screens.serve import ServeScreen
        serve = app.query_one(ServeScreen)
        assert serve.has_running_server
        assert serve._server_info == fake_info
```

- [ ] **Step 2: Run the new test to confirm it fails**

```bash
pytest tests/llamactl/test_serve_ui.py::test_serve_screen_reattaches_on_mount -v 2>&1 | tail -8
```

Expected: `AssertionError: assert False` (re-attach not yet wired)

- [ ] **Step 3: Add lifecycle wiring to `llamactl/ui/screens/serve.py`**

Add these methods inside `ServeScreen` (after the existing `compose` method):

```python
    def on_mount(self) -> None:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        from llamactl.core.lifecycle import find_running
        info = find_running(app._global_cfg, app._state_dir)
        if info is not None:
            self._set_server(info)
        self.set_interval(2.0, self._poll_vram_and_health)

    def _set_server(self, info: ServerInfo | None) -> None:
        self._server_info = info
        header = self.query_one(_StatusHeader)
        header.info = info
        stop_btn = self.query_one("#btn-stop", Button)
        launch_btn = self.query_one("#btn-launch", Button)
        if info is not None:
            stop_btn.disabled = False
            launch_btn.disabled = True
        else:
            stop_btn.disabled = True
            launch_btn.disabled = False
            header.state = ServerState.STOPPED

    async def _poll_vram_and_health(self) -> None:
        if self._server_info is None:
            return
        import asyncio
        from llamactl.core.monitor import check_health, read_vram_kib
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        port = self._server_info.port
        pid = str(self._server_info.pid) if self._server_info.pid else None

        state, vram_kib = await asyncio.gather(
            asyncio.to_thread(check_health, port),
            asyncio.to_thread(read_vram_kib, pid) if pid else asyncio.sleep(0, result=0),
        )
        header = self.query_one(_StatusHeader)
        header.state = state
        if state == ServerState.EXITED:
            self._set_server(None)
            return
        gauge = self.query_one(_VramGauge)
        gauge.vram_kib = vram_kib
        gauge.budget_kib = int(app._global_cfg.vram_budget_gb * 1024 * 1024)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-launch":
            self._action_launch()
        elif event.button.id == "btn-stop":
            self._action_stop()
        elif event.button.id == "btn-copy-argv":
            self._action_copy_argv()

    def _action_launch(self) -> None:
        from llamactl.core.lifecycle import (
            find_runtime,
            launch_container,
            launch_native,
            resolve_image,
        )
        from llamactl.core.config import resolve_settings
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        form = self.query_one(_LaunchForm)
        model, backend, preset = form.get_launch_params()
        if model is None:
            self.notify("Select a model first.", severity="warning")
            return
        mode_select = form.query_one("#mode-select", Select)
        mode = str(mode_select.value or "container")
        log = self.query_one("#log-pane", RichLog)
        try:
            settings = resolve_settings(model, preset, backend, {})
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        try:
            if mode == "container":
                rt = find_runtime()
                if rt is None:
                    self.notify("No container runtime found (podman/docker).", severity="error")
                    return
                image = resolve_image(model, backend)
                info = launch_container(
                    runtime=rt,
                    global_cfg=app._global_cfg,
                    model=model,
                    resolved_settings=settings,
                    backend=backend,
                    preset=preset or "",
                    image=image,
                    state_dir=app._state_dir,
                )
                log.write(f"[green]Container started:[/green] {info.container_name}")
            else:
                import shutil
                binary = shutil.which("llama-server") or ""
                if not binary:
                    self.notify("llama-server not found in PATH.", severity="error")
                    return
                info = launch_native(
                    global_cfg=app._global_cfg,
                    model=model,
                    resolved_settings=settings,
                    backend=backend,
                    preset=preset or "",
                    binary=binary,
                    state_dir=app._state_dir,
                )
                log.write(f"[green]Native server started:[/green] PID {info.pid}")
        except Exception as exc:
            self.notify(f"Launch failed: {exc}", severity="error")
            log.write(f"[red]Launch error:[/red] {exc}")
            return
        self._set_server(info)

    def _action_stop(self) -> None:
        if self._server_info is None:
            return
        from llamactl.core.lifecycle import find_runtime, stop_server
        log = self.query_one("#log-pane", RichLog)
        try:
            rt = find_runtime() if self._server_info.mode == "container" else None
            stop_server(self._server_info, runtime=rt)
            log.write("[yellow]Server stopped.[/yellow]")
        except Exception as exc:
            self.notify(f"Stop failed: {exc}", severity="error")
            log.write(f"[red]Stop error:[/red] {exc}")
        self._set_server(None)

    def _action_copy_argv(self) -> None:
        form = self.query_one(_LaunchForm)
        preview = form.query_one("#argv-preview", Static)
        text = str(preview.renderable)
        import pyperclip  # optional; fail gracefully
        try:
            import pyperclip
            pyperclip.copy(text)
            self.notify("argv copied to clipboard.")
        except Exception:
            self.notify("Install pyperclip to enable copy.", severity="warning")
```

Also add these missing imports at the top of `serve.py` (inside the existing import block):

```python
from llamactl.core.lifecycle import ServerInfo, ServerState, resolve_image
from llamactl.core.mapper import build_server_argv
```

(These are already present from Task 5 — verify they exist; add only the ones missing.)

- [ ] **Step 4: Run all Serve tab tests**

```bash
pytest tests/llamactl/test_serve_ui.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/llamactl/ -v --ignore=tests/llamactl/test_parity.py
```

Expected: all tests pass.

- [ ] **Step 6: Manual smoke test (optional but strongly recommended)**

```bash
python -m llamactl
```

The TUI should open, show the Serve tab with the model selector populated from
`configs/models/*.toml`, and display `● STOPPED` in the status header. Press `q`
to quit.

- [ ] **Step 7: Commit**

```bash
git add llamactl/ui/screens/serve.py tests/llamactl/test_serve_ui.py
git commit -m "feat: Serve tab live — launch, stop, VRAM poll, health state, re-attach"
```

---

## Self-Review Against Design Spec

Checking each Phase 2 requirement from `2026-06-10-llamactl-tui-design.md`:

| Requirement | Covered in |
|-------------|-----------|
| Container launch with ROCm device flags + labels | Task 2 `launch_container` |
| Container launch with Vulkan DRI passthrough | Task 1 `dri_passthrough_flags` + Task 2 |
| Native launch with HSA env var + pidfile + log | Task 2 `launch_native` |
| Re-attach on TUI restart (container labels + native pidfile) | Task 2 `find_running` |
| Health state machine (STOPPED→STARTING→LOADING→READY→UNHEALTHY) | Task 3 `check_health` |
| VRAM gauge vs `vram_budget_gb` | Task 3 `read_vram_kib` + Task 5 `_VramGauge` |
| Serve tab: status header, gauge, launch form, argv preview, log pane | Task 5 |
| Launch/stop actions with error surfaced in-UI | Task 6 |
| Re-attach on `on_mount` | Task 6 |
| `python -m llamactl` opens dashboard | Task 4 `__main__.py` |
| `validate_global_enums` (deferred from Phase 1) | Task 2 |
| Registry load error degrades gracefully | Task 4 `app.py` try/except |
| `load_all` errors rendered as disabled models | Task 5 `_LaunchForm` |

Deferred (known from handoff, not Phase 2 scope):
- `resolve_settings` shallow copy aliasing → Phase 4 (settings editor)
- `images` table type validation → validated implicitly at launch time (podman rejects non-strings)
- Log streaming pane (native: tail file, container: `podman logs -f`) → polish pass after Phase 2
- `flash_attn` str/bool inconsistency → Phase 4 settings editor must handle
