from __future__ import annotations

from llamactl.core.lifecycle import ServerInfo
from llamactl.core.oomtest import (
    NativeInspector,
    NativeLogReader,
    OomTestResult,
    build_vram_monitor,
    classify_verdict,
)
from stress_harness.models import PhaseResult


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


def test_build_vram_monitor_native_no_pid_returns_none():
    server = ServerInfo(model_id="m", backend="rocm", preset="", mode="native",
                        host="127.0.0.1", port=8080, started_at="", pid=None)
    mon = build_vram_monitor(server, runtime=None)
    assert mon.read() is None


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
    assert isinstance(reader, NativeLogReader)


def test_native_log_reader_handles_nonexistent_file(tmp_path):
    reader = NativeLogReader(str(tmp_path / "does-not-exist.log"))
    assert reader.line_count() == 0
    assert reader.dump_lines(5) == []


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
    over = classify_verdict([_phase("ramp")], peak_vram_gb=11.2, budget_gb=11.0,
                            vram_available=True)
    assert over.verdict == "WARN"
    # exact-budget boundary is also WARN (>= budget)
    at = classify_verdict([_phase("ramp")], peak_vram_gb=11.0, budget_gb=11.0,
                          vram_available=True)
    assert at.verdict == "WARN"


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
