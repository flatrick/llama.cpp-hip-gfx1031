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
