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
