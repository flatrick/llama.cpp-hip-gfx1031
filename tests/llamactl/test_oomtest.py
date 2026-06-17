from __future__ import annotations

from llamactl.core.lifecycle import ServerInfo
from llamactl.core.oomtest import (
    NativeInspector,
    NativeLogReader,
    OomTestResult,
    PhaseSet,
    build_vram_monitor,
    classify_verdict,
    run_phases,
)
from stress_harness.models import PhaseSample, PhaseResult, RuntimeInfo
from stress_harness.monitoring import VramMonitor


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


class _FakePhase:
    """Returns a canned PhaseResult; records that it ran and with what arg."""

    def __init__(self, key, success=True, last_ok=None, peak=None):
        self._key, self._success, self._last_ok, self._peak = key, success, last_ok, peak

    def make(self, *_args, **_kw):
        phase = self

        class _Runner:
            def run(self, _arg):
                samples = (
                    [PhaseSample(label="s", peak_vram_gb=phase._peak)]
                    if phase._peak is not None
                    else []
                )
                return _phase(
                    phase._key,
                    success=phase._success,
                    samples=samples,
                    last_ok=phase._last_ok,
                )

        return _Runner()


def _phase_set(specs):
    return PhaseSet(**{k: v.make for k, v in specs.items()})


def _noop_collaborators():
    mon = VramMonitor(lambda: 9.0, "test")
    return dict(
        client=object(),
        prompt_builder=object(),
        vram_monitor=mon,
        runtime_inspector=object(),
        runtime_info=RuntimeInfo(None, None, "x"),
    )


class _RecordingReporter:
    def start_run(self, r): ...
    def start_phase(self, p): ...
    def record_sample(self, k, s, w): ...
    def finish_phase(self, p): ...
    def finish_run(self, r): ...
    def error(self, m): ...


def test_run_phases_runs_full_chain_when_all_pass():
    specs = {
        "ramp": _FakePhase("ramp", last_ok=100, peak=9.0),
        "sustained": _FakePhase("sustained", peak=9.5),
        "cold_start": _FakePhase("cold-start", peak=10.0),
        "defrag": _FakePhase("defrag", peak=9.8),
        "boundary": _FakePhase("boundary", peak=9.0),
    }
    phases, peak, vram_available = run_phases(
        config_steps=[10, 20],
        phase_set=_phase_set(specs),
        cancel=lambda: False,
        reporter=_RecordingReporter(),
        **_noop_collaborators(),
    )
    assert [p.key for p in phases] == [
        "ramp", "sustained", "cold-start", "defrag", "boundary"
    ]
    assert peak == 10.0
    assert vram_available is True


def test_run_phases_short_circuits_on_ramp_failure():
    specs = {
        "ramp": _FakePhase("ramp", success=False),
        "sustained": _FakePhase("sustained"),
        "cold_start": _FakePhase("cold-start"),
        "defrag": _FakePhase("defrag"),
        "boundary": _FakePhase("boundary"),
    }
    phases, _, _ = run_phases(
        config_steps=[10],
        phase_set=_phase_set(specs),
        cancel=lambda: False,
        reporter=_RecordingReporter(),
        **_noop_collaborators(),
    )
    assert [p.key for p in phases] == ["ramp"]


def test_run_phases_short_circuits_on_sustained_failure():
    specs = {
        "ramp": _FakePhase("ramp", last_ok=100, peak=9.0),
        "sustained": _FakePhase("sustained", success=False),
        "cold_start": _FakePhase("cold-start"),
        "defrag": _FakePhase("defrag"),
        "boundary": _FakePhase("boundary"),
    }
    phases, _, _ = run_phases(
        config_steps=[10],
        phase_set=_phase_set(specs),
        cancel=lambda: False,
        reporter=_RecordingReporter(),
        **_noop_collaborators(),
    )
    assert [p.key for p in phases] == ["ramp", "sustained"]


def test_run_phases_stops_on_cancel():
    specs = {
        "ramp": _FakePhase("ramp", last_ok=100, peak=9.0),
        "sustained": _FakePhase("sustained"),
        "cold_start": _FakePhase("cold-start"),
        "defrag": _FakePhase("defrag"),
        "boundary": _FakePhase("boundary"),
    }
    phases, _, _ = run_phases(
        config_steps=[10],
        phase_set=_phase_set(specs),
        cancel=lambda: True,
        reporter=_RecordingReporter(),
        **_noop_collaborators(),
    )
    assert [p.key for p in phases] == ["ramp"]  # cancelled after first


def test_run_phases_cancel_after_defrag_returns_peak():
    """Cancelling so that defrag is the last phase run must still return the
    peak accumulated so far (regression guard for the run_phases refactor)."""
    from llamactl.core.oomtest import run_phases, PhaseSet
    from stress_harness.models import PhaseResult, PhaseSample

    def make_phase(key, peak, *, success=True, last_ok=10):
        class _P:
            def __init__(self, **kw):
                pass
            def run(self, *a):
                return PhaseResult(
                    key=key, title=key, samples=[
                        PhaseSample(label="s", peak_vram_gb=peak,
                                    post_vram_gb=None, ok=success, status="ok")
                    ],
                    success=success, summary="", log_excerpt=[], details=[],
                    last_ok_tokens=last_ok,
                )
        return _P

    calls = {"n": 0}
    def cancel():
        calls["n"] += 1
        return calls["n"] > 3  # True after ramp+sustained+cold+defrag have run

    pset = PhaseSet(
        ramp=make_phase("ramp", 5.0), sustained=make_phase("sustained", 6.0),
        cold_start=make_phase("cold", 7.0), defrag=make_phase("defrag", 9.0),
        boundary=make_phase("boundary", 99.0),  # must NOT run
    )

    class _Rep:
        def finish_phase(self, r): pass

    phases, peak, vram_avail = run_phases(
        config_steps=[1], phase_set=pset, cancel=cancel, reporter=_Rep(),
        client=None, prompt_builder=None, vram_monitor=None,
        runtime_inspector=None, runtime_info=None, config=None,
    )
    assert [p.key for p in phases] == ["ramp", "sustained", "cold", "defrag"]
    assert peak == 9.0
    assert vram_avail is True
