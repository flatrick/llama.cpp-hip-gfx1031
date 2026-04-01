from __future__ import annotations

import glob
import json
import os
import subprocess
import threading

from .models import RuntimeInfo
from .runtime import ContainerRuntimeInspector


class VramMonitor:
    def __init__(self, reader, mode: str) -> None:
        self._reader = reader
        self.mode = mode

    @classmethod
    def create(cls, inspector: ContainerRuntimeInspector, info: RuntimeInfo) -> "VramMonitor":
        pids = inspector.container_pids(info)
        if pids:
            test = cls._read_drm_vram_for_pids(pids)
            if test is not None:
                mode = f"per-process (PIDs: {', '.join(pids)})"
                info.vram_mode = mode
                return cls(lambda: cls._read_drm_vram_for_pids(pids), mode)

        # No container — try to find the local llama-server process by port.
        host_port = inspector.api_host_port()
        if host_port is not None:
            local_pids = cls._find_pids_for_port(host_port)
            if local_pids:
                test = cls._read_drm_vram_for_pids(local_pids)
                if test is not None:
                    mode = f"per-process local (PIDs: {', '.join(local_pids)})"
                    info.vram_mode = mode
                    return cls(lambda: cls._read_drm_vram_for_pids(local_pids), mode)

        mode = "system-wide (includes all GPU clients)"
        info.vram_mode = mode
        return cls(cls._system_vram_gb, mode)

    @staticmethod
    def _find_pids_for_port(port: int) -> list[str]:
        """Return PIDs of processes listening on *port* via /proc/net/tcp."""
        hex_port = f"{port:04X}"
        inodes: set[str] = set()

        # /proc/net/tcp covers the host; /proc/*/net/tcp covers per-namespace views.
        candidates = ["/proc/net/tcp"] + glob.glob("/proc/*/net/tcp")
        for tcp_path in candidates:
            try:
                with open(tcp_path) as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 10:
                            continue
                        local_addr, state, inode = parts[1], parts[3], parts[9]
                        # state 0A = TCP_LISTEN
                        if state == "0A" and local_addr.endswith(f":{hex_port}"):
                            inodes.add(inode)
            except (PermissionError, FileNotFoundError, OSError):
                continue

        if not inodes:
            return []

        pids: list[str] = []
        for fd_path in glob.glob("/proc/*/fd/*"):
            try:
                pid = fd_path.split("/")[2]
                if not pid.isdigit():
                    continue
                target = os.readlink(fd_path)
                if target.startswith("socket:[") and target[len("socket:["):-1] in inodes:
                    if pid not in pids:
                        pids.append(pid)
            except (PermissionError, FileNotFoundError, OSError):
                continue

        return pids

    def read(self) -> float | None:
        return self._reader()

    @staticmethod
    def _system_vram_gb() -> float | None:
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
            try:
                used = int(open(path).read().strip())
                return used / 1024 ** 3
            except Exception:
                continue

        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            data = json.loads(out)
            for card in data.values():
                if not isinstance(card, dict):
                    continue
                for key, value in card.items():
                    if "used" in key.lower() and "vram" in key.lower():
                        return int(value) / 1024 ** 3
        except Exception:
            return None
        return None

    @staticmethod
    def _read_drm_vram_for_pids(pids: list[str]) -> float | None:
        max_kib = 0
        for pid in pids:
            fdinfo_dir = f"/proc/{pid}/fdinfo"
            try:
                for fd_name in os.listdir(fdinfo_dir):
                    try:
                        with open(f"{fdinfo_dir}/{fd_name}") as handle:
                            for line in handle:
                                if line.startswith("drm-memory-vram:"):
                                    kib = int(line.split(":")[1].strip().split()[0])
                                    if kib > max_kib:
                                        max_kib = kib
                    except (PermissionError, FileNotFoundError, ValueError, OSError):
                        continue
            except (FileNotFoundError, PermissionError):
                continue
        return max_kib / 1024 ** 2 if max_kib > 0 else None


class PeakVramSampler:
    def __init__(self, monitor: VramMonitor, interval_ms: int = 200) -> None:
        self._monitor = monitor
        self._interval = interval_ms / 1000.0
        self._peak: float | None = None
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "PeakVramSampler":
        self._stop_evt.clear()
        self._peak = self._monitor.read()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            current = self._monitor.read()
            if current is not None and (self._peak is None or current > self._peak):
                self._peak = current
            self._stop_evt.wait(self._interval)

    def stop(self) -> float | None:
        self._stop_evt.set()
        self._thread.join(timeout=1)
        return self._peak
