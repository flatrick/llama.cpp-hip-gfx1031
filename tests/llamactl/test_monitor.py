from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llamactl.core.lifecycle import ServerState
from llamactl.core.monitor import (
    check_health,
    read_container_vram_kib,
    read_vram_kib,
)

import subprocess


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


def test_read_vram_kib_returns_none_on_permission_denied(monkeypatch):
    # Reading another user's (e.g. a root-owned container) /proc/<pid>/fdinfo
    # raises PermissionError; that must surface as None, not be mistaken for 0.
    def _denied(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("os.listdir", _denied)
    assert read_vram_kib("18145") is None


# ── read_container_vram_kib ───────────────────────────────────────────────────

def _ok(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _err() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 1, stdout="", stderr="no such container")


_EXEC_TWO_CLIENTS = (
    "@@@/proc/1/fdinfo/3\n"
    "drm-client-id:\t100\n"
    "drm-memory-vram:\t1000000 KiB\n"
    "@@@/proc/1/fdinfo/4\n"
    "drm-client-id:\t101\n"
    "drm-memory-vram:\t500000 KiB\n"
)


def test_read_container_vram_kib_sums_from_exec():
    captured: list[list[str]] = []

    def runner(cmd):
        captured.append(cmd)
        return _ok(_EXEC_TWO_CLIENTS)

    total = read_container_vram_kib("llamactl-qwen3.5-9b", "/usr/bin/docker", runner=runner)
    assert total == 1500000
    # Must run inside the container, not on the host.
    assert captured[0][:3] == ["/usr/bin/docker", "exec", "llamactl-qwen3.5-9b"]


def test_read_container_vram_kib_dedups_client_id():
    same_client = (
        "@@@/proc/1/fdinfo/3\n"
        "drm-client-id:\t100\n"
        "drm-memory-vram:\t1000000 KiB\n"
        "@@@/proc/1/fdinfo/4\n"
        "drm-client-id:\t100\n"
        "drm-memory-vram:\t1000000 KiB\n"
    )
    total = read_container_vram_kib("c", "/usr/bin/docker", runner=lambda _: _ok(same_client))
    assert total == 1000000  # counted once


def test_read_container_vram_kib_returns_none_on_exec_failure():
    total = read_container_vram_kib("c", "/usr/bin/docker", runner=lambda _: _err())
    assert total is None


def test_read_container_vram_kib_parses_output_despite_nonzero_exit():
    # The in-container shell loop can exit non-zero (a trailing `cat` on a
    # transient fd fails) yet still print valid fdinfo. Output present must be
    # parsed, not discarded — this was the real-world failure.
    result = subprocess.CompletedProcess([], 1, stdout=_EXEC_TWO_CLIENTS, stderr="cat: ...")
    total = read_container_vram_kib("c", "/usr/bin/docker", runner=lambda _: result)
    assert total == 1500000


def test_parse_fdinfo_normalises_mib_to_kib(tmp_path):
    from llamactl.core.monitor import _parse_fdinfo_file
    p = tmp_path / "fd0"
    p.write_text("drm-client-id:\t42\ndrm-memory-vram:\t2 MiB\n")
    _, fields = _parse_fdinfo_file(str(p))
    assert fields["drm-memory-vram"] == 2 * 1024


def test_parse_fdinfo_normalises_gib_to_kib(tmp_path):
    from llamactl.core.monitor import _parse_fdinfo_file
    p = tmp_path / "fd0"
    p.write_text("drm-client-id:\t42\ndrm-memory-vram:\t1 GiB\n")
    _, fields = _parse_fdinfo_file(str(p))
    assert fields["drm-memory-vram"] == 1024 * 1024


def test_parse_fdinfo_unknown_unit_treated_as_kib(tmp_path):
    from llamactl.core.monitor import _parse_fdinfo_file
    p = tmp_path / "fd0"
    p.write_text("drm-client-id:\t42\ndrm-memory-vram:\t500 XUNIT\n")
    _, fields = _parse_fdinfo_file(str(p))
    assert fields["drm-memory-vram"] == 500  # treated as KiB


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


# ── read_server_vram_kib ──────────────────────────────────────────────────────

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

    # container mode but no runtime -> None
    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="container",
        host="0.0.0.0", port=8080, started_at="", container_name="llamactl-m",
    )
    assert mon.read_server_vram_kib(server, None) is None
