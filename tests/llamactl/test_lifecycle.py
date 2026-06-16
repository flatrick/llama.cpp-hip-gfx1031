from __future__ import annotations

import json
import signal
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
            runner=runner,
        )


def test_launch_container_vulkan_calls_dri_passthrough(tmp_path):
    captured: list[list[str]] = []
    runner = lambda cmd: (captured.append(cmd), _ok("abc123\n"))[1]
    fake_dri = lambda: (["--device", "/dev/dri/renderD128:/dev/dri/renderD128"], ["--group-add", "44"])

    launch_container(
        runtime="/usr/bin/podman",
        global_cfg=_global_cfg(),
        model=_model(),
        resolved_settings={"ctx_size": 4096},
        backend="vulkan",
        preset="",
        image=DEFAULT_VULKAN_IMAGE,
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
    monkeypatch.setattr("llamactl.core.lifecycle.os.kill",
                        lambda pid, sig: killed.append((pid, sig)))

    info = ServerInfo(
        model_id="qwen3-9b", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="2026-06-16T14:00:00",
        pid=54321,
    )
    stop_server(info)
    assert killed[0] == (54321, signal.SIGTERM)


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
    runner = lambda _cmd: subprocess.CompletedProcess([], 0, stdout="[]", stderr="")
    monkeypatch.setattr("llamactl.core.lifecycle._is_pid_alive", lambda pid: pid == 77777)

    info = find_running(_global_cfg(), tmp_path, runtime="/usr/bin/podman", runner=runner)
    assert info is not None
    assert info.mode == "native"
    assert info.pid == 77777
    assert info.backend == "vulkan"
