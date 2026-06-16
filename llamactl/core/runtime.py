"""Container runtime abstraction for llamactl.

Provides detection of podman/docker and label-based container operations.
All subprocess calls accept an injectable runner parameter for testability.
"""

from __future__ import annotations

import glob as _glob_mod
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
    return shutil.which("podman") or shutil.which("docker") or None


def list_managed(
    runtime: str,
    name_prefix: str,
    runner: Runner = _default_runner,
) -> list[ContainerInfo]:
    """Return all llamactl-managed containers (running and stopped).

    Runs: runtime ps -a --filter label=llamactl.managed=1 --format json

    Handles both Podman JSON (array) and Docker JSON (newline-delimited objects).
    Returns [] on nonzero exit, empty output, or unparseable JSON.
    Only returns containers whose name starts with name_prefix.
    Labels used: llamactl.model, llamactl.backend, llamactl.preset
    """
    cmd = [
        runtime,
        "ps",
        "-a",
        "--filter",
        "label=llamactl.managed=1",
        "--format",
        "json",
    ]
    result = runner(cmd)

    if result.returncode != 0:
        return []

    stdout = result.stdout.strip()
    if not stdout:
        return []

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        # Try newline-delimited JSON (Docker format)
        try:
            parsed = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        # Docker may return a single object
        parsed = [parsed]

    containers: list[ContainerInfo] = []
    for item in parsed:
        # Extract name — Podman uses "Names" (list), Docker uses "Names" (string)
        names_field = item.get("Names", [])
        if isinstance(names_field, list):
            name = names_field[0] if names_field else ""
        else:
            name = str(names_field).lstrip("/")

        if not name.startswith(name_prefix):
            continue

        labels = item.get("Labels") or {}
        state = item.get("State", "")

        containers.append(
            ContainerInfo(
                name=name,
                state=state,
                model_id=labels.get("llamactl.model", ""),
                backend=labels.get("llamactl.backend", ""),
                preset=labels.get("llamactl.preset", ""),
            )
        )

    return containers


def container_stop(runtime: str, name: str, runner: Runner = _default_runner) -> None:
    """Runs: runtime stop name"""
    runner([runtime, "stop", name])


def container_logs_cmd(runtime: str, name: str) -> list[str]:
    """Returns [runtime, 'logs', '-f', name] — caller opens as Popen."""
    return [runtime, "logs", "-f", name]


def dri_passthrough_flags(
    _glob=None,   # injectable: defaults to glob.glob
    _stat=None,   # injectable: defaults to os.stat
) -> tuple[list[str], list[str]]:
    """Return (device_flags, group_flags) for /dev/dri/* passthrough (Vulkan).

    For each node in glob("/dev/dri/renderD*") + glob("/dev/dri/card*"):
      device_flags += ["--device", f"{node}:{node}"]
      deduplicated group_flags += ["--group-add", str(gid)]

    Returns ([], []) if no nodes found.
    """
    if _glob is None:
        _glob = _glob_mod.glob
    if _stat is None:
        _stat = os.stat

    nodes = _glob("/dev/dri/renderD*") + _glob("/dev/dri/card*")

    if not nodes:
        return [], []

    device_flags: list[str] = []
    group_flags: list[str] = []
    seen_gids: set[int] = set()

    for node in nodes:
        device_flags += ["--device", f"{node}:{node}"]
        gid = _stat(node).st_gid
        if gid not in seen_gids:
            seen_gids.add(gid)
            group_flags += ["--group-add", str(gid)]

    return device_flags, group_flags
