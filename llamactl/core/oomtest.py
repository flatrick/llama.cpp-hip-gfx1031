"""Quick OOM boundary check against a running llama-server.

Drives the UNMODIFIED stress_harness phase classes, injecting llamactl's own
VRAM source (core/monitor.read_vram_kib) and a caller-supplied reporter through
the phases' existing constructor parameters. Works for native and container
servers.
"""
from __future__ import annotations

from collections.abc import Callable

from stress_harness.monitoring import VramMonitor

from llamactl.core.lifecycle import ServerInfo
from llamactl.core.monitor import read_vram_kib
from llamactl.core.runtime import find_runtime, get_container_pid


def build_vram_monitor(
    server: ServerInfo,
    runtime: str | None,
    get_pid: Callable[[str, str], int | None] | None = get_container_pid,
) -> VramMonitor:
    """A harness VramMonitor whose reader reports the managed server's VRAM in GiB.

    PID resolved ONCE up front (container inspect is a subprocess; the 200ms
    PeakVramSampler must not re-resolve per tick).
    """
    if server.mode == "native":
        pid = str(server.pid) if server.pid else None
        mode = f"per-process native (PID: {pid or 'unknown'})"
    else:
        rt = runtime or find_runtime()
        resolved = (
            get_pid(server.container_name, rt)
            if (rt and server.container_name and get_pid)
            else None
        )
        pid = str(resolved) if resolved else None
        mode = f"per-process container (PID: {pid or 'unknown'})"

    def _reader() -> float | None:
        if pid is None:
            return None
        kib = read_vram_kib(pid)
        return kib / 1024 ** 2 if kib > 0 else None

    return VramMonitor(_reader, mode)
