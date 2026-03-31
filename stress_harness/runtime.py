from __future__ import annotations

import collections
import re
import shutil
import subprocess
import threading
import urllib.parse

from .models import RuntimeInfo


class ContainerLogReader:
    def __init__(self, runtime: str, container_id: str, max_lines: int = 200) -> None:
        self._buf: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._line_count = 0
        self._proc = subprocess.Popen(
            [runtime, "logs", "--follow", "--since", "0s", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        if self._proc.stdout is None:
            return
        for raw in self._proc.stdout:
            try:
                self._buf.append(raw.decode(errors="replace").rstrip())
                self._line_count += 1
            except Exception:
                continue

    def line_count(self) -> int:
        return self._line_count

    def dump_lines(self, limit: int) -> list[str]:
        return list(self._buf)[-limit:]

    def stop(self) -> None:
        try:
            self._proc.terminate()
        except Exception:
            pass


class ContainerRuntimeInspector:
    _HINT_RE = re.compile(r"(llama-cpp|llama-server|vulkan|gfx1031)", re.IGNORECASE)

    def __init__(self, api_url: str, log_lines: int = 30) -> None:
        self.api_url = api_url
        self.log_lines = log_lines

    @staticmethod
    def host_port_matches(ports: str, host_port: int) -> bool:
        if not ports:
            return False
        pattern = re.compile(rf"(^|[,\s])(?:\[[^\]]+\]|[^,\s:]+):{host_port}->")
        return bool(pattern.search(ports))

    def api_host_port(self) -> int | None:
        try:
            parsed = urllib.parse.urlparse(self.api_url)
            return parsed.port
        except Exception:
            return None

    def find_runtime(self) -> str | None:
        for runtime in ("podman", "docker"):
            if shutil.which(runtime):
                return runtime
        return None

    def find_container_id(self, runtime: str | None) -> str | None:
        if runtime is None:
            return None
        try:
            out = subprocess.check_output(
                [runtime, "ps", "--format", "{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Ports}}"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode()
        except Exception:
            return None

        rows: list[tuple[str, str, str, str]] = []
        for line in out.strip().splitlines():
            parts = line.split("\t", 3)
            if len(parts) != 4:
                continue
            rows.append((parts[0], parts[1], parts[2], parts[3]))

        host_port = self.api_host_port()
        if host_port is not None:
            for container_id, _image, _name, ports in rows:
                if self.host_port_matches(ports, host_port):
                    return container_id

        for container_id, image, name, _ports in rows:
            if self._HINT_RE.search(image) or self._HINT_RE.search(name):
                return container_id

        return None

    def detect(self) -> RuntimeInfo:
        runtime = self.find_runtime()
        container_id = self.find_container_id(runtime)
        if container_id and runtime:
            return RuntimeInfo(
                runtime=runtime,
                container_id=container_id,
                status_message=f"Container logs: tapping {container_id[:12]} via {runtime}",
            )
        if runtime:
            host_port = self.api_host_port()
            port_label = f"port {host_port}" if host_port is not None else self.api_url
            return RuntimeInfo(
                runtime=runtime,
                container_id=None,
                status_message=f"Container logs: no running container found for {port_label}",
            )
        return RuntimeInfo(
            runtime=None,
            container_id=None,
            status_message="Container logs: podman/docker not found in PATH",
        )

    def start_log_reader(self, info: RuntimeInfo) -> ContainerLogReader | None:
        if info.runtime and info.container_id:
            return ContainerLogReader(info.runtime, info.container_id)
        return None

    def container_running(self, info: RuntimeInfo) -> bool | None:
        if info.runtime is None or info.container_id is None:
            return None
        try:
            out = subprocess.check_output(
                [info.runtime, "inspect", "-f", "{{.State.Running}}", info.container_id],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip().lower()
        except Exception:
            return None
        if out in {"true", "false"}:
            return out == "true"
        return None

    def container_pids(self, info: RuntimeInfo) -> list[str]:
        if info.runtime is None or info.container_id is None:
            return []
        try:
            out = subprocess.check_output(
                [info.runtime, "top", info.container_id, "-o", "pid"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode()
        except Exception:
            return []

        pids: list[str] = []
        for line in out.strip().splitlines()[1:]:
            pid = line.strip().split()[0]
            if pid.isdigit():
                pids.append(pid)
        return pids
