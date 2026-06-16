"""Server lifecycle management for llamactl.

Handles launching, stopping, and discovering llama-server processes in both
container (podman/docker) and native modes.
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
from dataclasses import dataclass
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

# Type alias for native process spawner (injectable for tests)
# Args: (cmd, env_extra, log_path) → pid (int)
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


# ── Private helpers ───────────────────────────────────────────────────────────

def _default_spawner(cmd: list[str], env_extra: dict[str, str], log_path: Path) -> int:
    """Start detached process: env={**os.environ, **env_extra}, stdout/stderr → log_path, start_new_session=True. Return pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            env={**os.environ, **env_extra},
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def _is_pid_alive(pid: int) -> bool:
    """Read /proc/{pid}/cmdline; return True if 'llama-server' in content. Return False on OSError."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"llama-server" in f.read()
    except OSError:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def validate_global_enums(cfg: GlobalConfig) -> None:
    """Raise ConfigError if default_backend or default_mode are not in the valid sets."""
    if cfg.default_backend not in VALID_BACKENDS:
        raise ConfigError(
            f"default_backend '{cfg.default_backend}' is not valid; "
            f"must be one of: {sorted(VALID_BACKENDS)}"
        )
    if cfg.default_mode not in VALID_MODES:
        raise ConfigError(
            f"default_mode '{cfg.default_mode}' is not valid; "
            f"must be one of: {sorted(VALID_MODES)}"
        )


def resolve_image(model: ModelConfig, backend: str, override: str | None = None) -> str:
    """Resolution order: explicit override → model.images[backend] → built-in default."""
    if override is not None:
        return override
    if backend in model.images:
        return model.images[backend]
    if backend == "rocm":
        return DEFAULT_ROCM_IMAGE
    return DEFAULT_VULKAN_IMAGE


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
    """
    Runs: runtime run -d --name llamactl-{model.id}
        --label llamactl.managed=1 --label llamactl.model=... --label llamactl.backend=...
        --label llamactl.preset=...
        [ROCm: --device /dev/kfd --device /dev/dri --group-add video --group-add render]
        [Vulkan: device_flags + group_flags from dri_flags_fn()]
        -v {hf_cache}:/root/.cache/huggingface -v {llama_cache}:/root/.cache/llama.cpp
        -p {port}:{port}
        {image}
        [build_server_argv output]

    Raises RuntimeError on nonzero exit (include exit code and stderr in message).
    Returns ServerInfo with mode="container", container_name="llamactl-{model.id}".
    """
    container_name = f"{global_cfg.name_prefix}-{model.id}"
    port = global_cfg.port

    cmd = [
        runtime, "run", "-d",
        "--name", container_name,
        "--label", "llamactl.managed=1",
        "--label", f"llamactl.model={model.id}",
        "--label", f"llamactl.backend={backend}",
        "--label", f"llamactl.preset={preset}",
    ]

    if backend == "rocm":
        cmd += [
            "--device", "/dev/kfd",
            "--device", "/dev/dri",
            "--group-add", "video",
            "--group-add", "render",
        ]
    else:
        # Vulkan: use dri passthrough flags
        device_flags, group_flags = dri_flags_fn()
        cmd += device_flags + group_flags

    cmd += [
        "-v", f"{global_cfg.hf_cache}:/root/.cache/huggingface",
        "-v", f"{global_cfg.llama_cache}:/root/.cache/llama.cpp",
        "-p", f"{port}:{port}",
        image,
        *build_server_argv(model.hf, resolved_settings, "0.0.0.0", port),
    ]

    result = runner(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Container launch failed (exit {result.returncode}): {result.stderr}"
        )

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return ServerInfo(
        model_id=model.id,
        backend=backend,
        preset=preset,
        mode="container",
        host="0.0.0.0",
        port=port,
        started_at=started_at,
        container_name=container_name,
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
    """
    Builds cmd: [binary, *build_server_argv(model.hf, resolved_settings, "0.0.0.0", port)]
    env_extra: {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"} for ROCm only.
    log_path: state_dir / "logs" / f"{started_at.replace(':', '-')}-{model.id}.log"
    Calls spawner(cmd, env_extra, log_path) → pid.
    Writes state_dir / "native-server.json" with keys:
        pid, model_id, backend, preset, host ("0.0.0.0"), port, log_path (str), started_at
    Returns ServerInfo with mode="native".
    """
    port = global_cfg.port
    host = "0.0.0.0"
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    cmd = [binary, *build_server_argv(model.hf, resolved_settings, host, port)]

    env_extra: dict[str, str] = {}
    if backend == "rocm":
        env_extra["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    log_filename = f"{started_at.replace(':', '-')}-{model.id}.log"
    log_path = state_dir / "logs" / log_filename

    pid = spawner(cmd, env_extra, log_path)

    meta = {
        "pid": pid,
        "model_id": model.id,
        "backend": backend,
        "preset": preset,
        "host": host,
        "port": port,
        "log_path": str(log_path),
        "started_at": started_at,
    }
    (state_dir / "native-server.json").write_text(json.dumps(meta))

    return ServerInfo(
        model_id=model.id,
        backend=backend,
        preset=preset,
        mode="native",
        host=host,
        port=port,
        started_at=started_at,
        log_path=str(log_path),
        pid=pid,
    )


def stop_server(
    info: ServerInfo,
    runtime: str | None = None,
    runner: Runner = _default_runner,
) -> None:
    """
    Container mode: container_stop(runtime or find_runtime(), info.container_name, runner)
    Native mode: os.kill(info.pid, signal.SIGTERM); swallow ProcessLookupError
    """
    if info.mode == "container" and info.container_name:
        rt = runtime or find_runtime()
        if rt is None:
            raise RuntimeError("No container runtime (podman/docker) found")
        container_stop(rt, info.container_name, runner)
    else:
        if info.pid is None:
            return
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
    """
    1. Check managed containers: (runtime or find_runtime()) → list_managed → first with state=="running"
       → return ServerInfo(mode="container", started_at="", container_name=c.name, ...)
    2. Check native pidfile: state_dir/"native-server.json" → json.loads →
       _is_pid_alive(pid) → return ServerInfo(mode="native", pid=..., ...)
    3. Return None if neither found.
    Swallow json.JSONDecodeError, KeyError, TypeError, ValueError from pidfile reads.
    """
    rt = runtime or find_runtime()
    if rt is not None:
        containers = list_managed(rt, global_cfg.name_prefix, runner)
        for c in containers:
            if c.state == "running":
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

    pidfile = state_dir / "native-server.json"
    if pidfile.exists():
        try:
            meta = json.loads(pidfile.read_text(encoding="utf-8"))
            pid = int(meta["pid"])
            if _is_pid_alive(pid):
                return ServerInfo(
                    model_id=meta["model_id"],
                    backend=meta["backend"],
                    preset=meta["preset"],
                    mode="native",
                    host=meta["host"],
                    port=meta["port"],
                    started_at=meta["started_at"],
                    log_path=meta.get("log_path"),
                    pid=pid,
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    return None
