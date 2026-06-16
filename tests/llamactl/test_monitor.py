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
