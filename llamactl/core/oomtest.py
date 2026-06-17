"""Quick OOM boundary check against a running llama-server.

Drives the UNMODIFIED stress_harness phase classes, injecting llamactl's own
VRAM source (core/monitor.read_vram_kib) and a caller-supplied reporter through
the phases' existing constructor parameters. Works for native and container
servers.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from stress_harness.config import StressConfig
from stress_harness.models import PhaseResult, PhaseSample, RuntimeInfo, StressRunResult
from stress_harness.monitoring import VramMonitor
from stress_harness.phases import (
    BoundaryPhase,
    ColdStartPhase,
    DefragPhase,
    RampPhase,
    SustainedPhase,
)
from stress_harness.prompting import PromptBuilder
from stress_harness.runtime import ContainerRuntimeInspector
from stress_harness.server import LlamaServerClient

from llamactl.core.config import GlobalConfig
from llamactl.core.lifecycle import ServerInfo
from llamactl.core.monitor import read_container_vram_kib, read_vram_kib
from llamactl.core.runtime import find_runtime


# Container VRAM is read via `exec` (per tick). That is far heavier than a
# native /proc read, so sample it less often than the 200ms native default.
CONTAINER_VRAM_SAMPLE_INTERVAL_MS = 1000


class Reporter(Protocol):
    """Duck-typed reporter surface the phases + runner drive.

    `error` is dual-purpose (informational and fatal); `finish_run` is called
    only when all phases pass.
    """

    def start_run(self, result: StressRunResult) -> None: ...
    def start_phase(self, phase: PhaseResult) -> None: ...
    def record_sample(self, phase_key: str, sample: PhaseSample, warn_at: float | None) -> None: ...
    def finish_phase(self, phase: PhaseResult) -> None: ...
    def finish_run(self, result: StressRunResult) -> None: ...
    def error(self, message: str) -> None: ...


def build_vram_monitor(
    server: ServerInfo,
    runtime: str | None,
) -> VramMonitor:
    """A harness VramMonitor whose reader reports the managed server's VRAM in GiB.

    Native: read the server's own /proc/<pid>/fdinfo (cheap, 200ms cadence).
    Container: read fdinfo *inside* the container via `exec` (no host sudo). That
    exec is heavy, so use a slower sample cadence.
    """
    if server.mode == "native":
        pid = str(server.pid) if server.pid else None
        mode = f"per-process native (PID: {pid or 'unknown'})"

        def _reader() -> float | None:
            if pid is None:
                return None
            kib = read_vram_kib(pid)
            return kib / 1024 ** 2 if kib else None

        return VramMonitor(_reader, mode)

    rt = runtime or find_runtime()
    cname = server.container_name
    mode = f"per-container exec (container: {cname or 'unknown'})"

    def _reader() -> float | None:
        if not (rt and cname):
            return None
        kib = read_container_vram_kib(cname, rt)
        # None = unreadable, 0 = no DRM memory; both mean "no usable reading".
        return kib / 1024 ** 2 if kib else None

    return VramMonitor(
        _reader, mode, sample_interval_ms=CONTAINER_VRAM_SAMPLE_INTERVAL_MS
    )


class NativeLogReader:
    """Minimal stand-in for ContainerLogReader over a native server's log file."""

    def __init__(self, log_path: str | None) -> None:
        self._path = log_path

    def _lines(self) -> list[str]:
        # Re-reads the file on every call. Acceptable: native log reads are a
        # low-frequency, cold-path failure-excerpt feature, and logs are short
        # during an OOM test (no in-memory deque needed as ContainerLogReader has).
        if not self._path:
            return []
        try:
            with open(self._path, encoding="utf-8", errors="replace") as fh:
                return fh.read().splitlines()
        except OSError:
            return []

    def line_count(self) -> int:
        return len(self._lines())

    def dump_lines(self, limit: int) -> list[str]:
        lines = self._lines()
        return lines[-limit:] if limit > 0 else lines

    def stop(self) -> None:
        return None


class NativeInspector:
    """Runtime-inspector adapter for native servers (no container)."""

    def __init__(self, log_path: str | None) -> None:
        self._log_path = log_path

    def start_log_reader(self, info) -> NativeLogReader:
        return NativeLogReader(self._log_path)

    def container_running(self, info) -> bool | None:
        return None  # native: watchdog treats None as "still running"

    def container_pids(self, info) -> list[str]:
        return []

    def api_host_port(self) -> int | None:
        return None


@dataclass(frozen=True, slots=True)
class OomTestResult:
    verdict: str            # "OK" | "WARN" | "FAIL" | "OK (degraded)"
    peak_vram_gb: float | None
    last_ok_tokens: int | None
    failed_phase: str | None
    detail: str


def classify_verdict(
    phases: list[PhaseResult],
    peak_vram_gb: float | None,
    budget_gb: float,
    vram_available: bool,
) -> OomTestResult:
    """Classify a completed OOM test run into a verdict.

    Precedence: FAIL > OK (degraded) > WARN > OK.
    """
    last_ok = next((p.last_ok_tokens for p in phases if p.last_ok_tokens), None)
    failed = next((p for p in phases if not p.success), None)
    if failed is not None:
        detail = (
            failed.summary
            or " / ".join(failed.log_excerpt[-3:])
            or "phase failed"
        )
        return OomTestResult("FAIL", peak_vram_gb, last_ok, failed.key, detail)
    if not vram_available:
        return OomTestResult(
            "OK (degraded)", None, last_ok, None,
            "All phases passed. VRAM unavailable — boundary + liveness only; "
            "leak detection skipped.",
        )
    if peak_vram_gb is not None and peak_vram_gb >= budget_gb:
        return OomTestResult(
            "WARN", peak_vram_gb, last_ok, None,
            f"All phases passed but peak {peak_vram_gb:.2f} GB ≥ budget {budget_gb:.0f} GB.",
        )
    peak_str = f"{peak_vram_gb:.2f} GB" if peak_vram_gb is not None else "n/a"
    return OomTestResult(
        "OK", peak_vram_gb, last_ok, None,
        f"All phases passed. Peak VRAM {peak_str}, under the {budget_gb:.0f} GB budget.",
    )


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSet:
    """Holds the five phase constructors (or test fakes) used by run_phases."""

    ramp: Callable[..., Any] = RampPhase
    sustained: Callable[..., Any] = SustainedPhase
    cold_start: Callable[..., Any] = ColdStartPhase
    defrag: Callable[..., Any] = DefragPhase
    boundary: Callable[..., Any] = BoundaryPhase


DEFAULT_PHASES = PhaseSet()


def _peak_from(samples: list[PhaseSample]) -> float | None:
    """Return the maximum VRAM reading across a list of PhaseSamples."""
    vals = [s.peak_vram_gb for s in samples if s.peak_vram_gb is not None]
    vals += [
        s.post_vram_gb
        for s in samples
        if getattr(s, "post_vram_gb", None) is not None
    ]
    return max(vals) if vals else None


def run_phases(
    *,
    config_steps: list[int],
    phase_set: PhaseSet,
    cancel: Callable[[], bool],
    reporter: Reporter,
    client: LlamaServerClient,
    prompt_builder: PromptBuilder,
    vram_monitor: VramMonitor,
    runtime_inspector: Any,
    runtime_info: RuntimeInfo,
    config: StressConfig | None = None,
    ctx_size: int | None = None,
) -> tuple[list[PhaseResult], float | None, bool]:
    """Run ramp → sustained → cold-start → defrag → boundary.

    Short-circuits on the first phase failure or when *cancel* returns True.
    Returns (phases_run, peak_vram_gb, vram_available).
    """
    kw = dict(
        config=config,
        client=client,
        prompt_builder=prompt_builder,
        vram_monitor=vram_monitor,
        runtime_inspector=runtime_inspector,
        runtime_info=runtime_info,
        reporter=reporter,
        cancel=cancel,
    )
    phases: list[PhaseResult] = []
    peaks: list[float] = []
    any_reading = False

    def _track(result: PhaseResult) -> None:
        nonlocal any_reading
        phases.append(result)
        reporter.finish_phase(result)
        p = _peak_from(result.samples)
        if p is not None:
            peaks.append(p)
            any_reading = True

    def _result() -> tuple[list[PhaseResult], float | None, bool]:
        return phases, (max(peaks) if peaks else None), any_reading

    # Phase 1: Ramp
    ramp = phase_set.ramp(**kw).run(config_steps)
    _track(ramp)
    if not ramp.success or not ramp.last_ok_tokens or cancel():
        return _result()
    last_ok = ramp.last_ok_tokens

    # Phases 2–4: Sustained, ColdStart, Defrag
    for attr, arg in (
        ("sustained", last_ok),
        ("cold_start", last_ok),
        ("defrag", last_ok),
    ):
        result = getattr(phase_set, attr)(**kw).run(arg)
        _track(result)
        if not result.success or cancel():
            return _result()

    # Phase 5: Boundary — must oversize against the REAL context window. Prefer an
    # explicit override, then the server-detected ctx_size; fall back to last_ok
    # only when ctx_size is unknown. (Using last_ok ~0.95*ctx never exceeds a large
    # context, so the server accepts the prompt and boundary spuriously fails.)
    ctx = (
        config.ctx_size_override
        if (config and config.ctx_size_override)
        else (ctx_size if ctx_size else last_ok)
    )
    boundary = phase_set.boundary(**kw).run(ctx)
    _track(boundary)
    return _result()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_oom_check(
    server: ServerInfo,
    reporter: Reporter,
    *,
    global_cfg: GlobalConfig,
    cancel: Callable[[], bool] = lambda: False,
    phase_set: PhaseSet = DEFAULT_PHASES,
    quick: bool = True,
) -> OomTestResult:
    """Run the OOM boundary check against the running server.

    quick=True (default) runs the reduced suite; quick=False runs full rounds.
    """
    api_url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
    config = StressConfig.from_env({
        "QUICK": "1" if quick else "0",
        "API_URL": api_url,
        "VRAM_WARN_GB": str(global_cfg.vram_budget_gb),
        # NB: do NOT set CTX_SIZE — let the harness auto-detect from /slots.
    })
    client = LlamaServerClient(config)
    if not client.server_healthy():
        return OomTestResult(
            "FAIL", None, None, None,
            f"Server not reachable at {api_url}",
        )

    prompt_builder = PromptBuilder(config.filler_chunk, config.tokens_per_chunk)

    if server.mode == "native":
        inspector = NativeInspector(server.log_path)
        runtime_info = RuntimeInfo(
            runtime=None, container_id=None, status_message="native server"
        )
        runtime = None
    else:
        inspector = ContainerRuntimeInspector(api_url)
        runtime_info = inspector.detect()
        runtime = runtime_info.runtime

    vram_monitor = build_vram_monitor(server, runtime)
    ctx_size = client.server_ctx_size()
    steps = config.build_steps(ctx_size)

    reporter.start_run(
        _run_summary(config, ctx_size, runtime_info, vram_monitor.read(), steps)
    )

    phases, peak, vram_available = run_phases(
        config_steps=steps,
        phase_set=phase_set,
        cancel=cancel,
        reporter=reporter,
        client=client,
        prompt_builder=prompt_builder,
        vram_monitor=vram_monitor,
        runtime_inspector=inspector,
        runtime_info=runtime_info,
        config=config,
        ctx_size=ctx_size,
    )
    if cancel():
        return OomTestResult(
            "STOPPED", peak, None, None, "Stopped by user."
        )
    return classify_verdict(phases, peak, global_cfg.vram_budget_gb, vram_available)


def _run_summary(
    config: StressConfig,
    ctx_size: int,
    runtime_info: RuntimeInfo,
    baseline: float | None,
    steps: list[int],
) -> StressRunResult:
    """Build a StressRunResult for reporter.start_run."""
    return StressRunResult(
        config=config,
        ctx_size=ctx_size,
        steps=steps,
        runtime=runtime_info,
        baseline_vram_gb=baseline,
    )
