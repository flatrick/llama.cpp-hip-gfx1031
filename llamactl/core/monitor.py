"""VRAM inspection and health polling for a running llama-server.

read_vram_kib: memory-only subset of vram_inspect.py's fdinfo reader.
check_health: polls /health and maps HTTP status to ServerState.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Iterable

from llamactl.core.lifecycle import ServerState
from llamactl.core.runtime import Runner, _default_runner

_UNIT_MULTIPLIERS: dict[str, float] = {
    "kib": 1.0,
    "mib": 1024.0,
    "gib": 1024.0 * 1024.0,
    "b": 1.0 / 1024.0,
    "bytes": 1.0 / 1024.0,
}


def _parse_fdinfo_text(text: str) -> tuple[str | None, dict[str, int]]:
    """
    Parse the contents of one fdinfo file. Returns (drm_client_id, {field: kib}).
    - Only processes lines starting with "drm-"
    - "drm-client-id" sets the client_id
    - Lines where "memory" is NOT in the key name are skipped
    - Value is the integer before the first space/tab on the value side
    - Lines that don't parse as int are skipped
    """
    client_id: str | None = None
    fields: dict[str, int] = {}

    for line in text.splitlines():
        if not line.startswith("drm-"):
            continue
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()

        if key == "drm-client-id":
            client_id = raw_value.split()[0] if raw_value.split() else raw_value
            continue

        if "memory" not in key:
            continue

        parts = raw_value.split()
        if not parts:
            continue
        try:
            raw_int = int(parts[0])
            if len(parts) >= 2:
                multiplier = _UNIT_MULTIPLIERS.get(parts[1].lower(), 1.0)
                fields[key] = int(raw_int * multiplier)
            else:
                fields[key] = raw_int  # assume KiB if no unit
        except ValueError:
            continue

    return (client_id, fields)


def _parse_fdinfo_file(path: str) -> tuple[str | None, dict[str, int]]:
    """Read one fdinfo file and parse it. OSError on open → (None, {})."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return _parse_fdinfo_text(fh.read())
    except OSError:
        return (None, {})


def _sum_unique_clients(
    parsed: Iterable[tuple[str | None, dict[str, int]]],
) -> int:
    """Sum memory fields across DRM clients, deduplicating by client-id.

    Fds sharing a drm-client-id are counted once (first seen wins). Entries with
    no client-id are each treated as unique. Entries with no fields are ignored.
    """
    seen_clients: dict[str, dict[str, int]] = {}
    anon_counter = 0

    for client_id, fields in parsed:
        if not fields:
            continue
        if client_id is not None:
            key = client_id
        else:
            key = f"_anon_{anon_counter}"
            anon_counter += 1
        if key not in seen_clients:
            seen_clients[key] = fields

    return sum(sum(fields.values()) for fields in seen_clients.values())


def read_vram_kib(pid: str, fdinfo_root: str = "/proc") -> int | None:
    """
    Sum all drm-*-memory-* KiB fields across unique DRM clients for PID.

    Reads {fdinfo_root}/{pid}/fdinfo/, parses each fd, deduplicates by
    drm-client-id, and sums memory fields.

    Returns:
      - int  → KiB in use (0 if the pid has no DRM fds)
      - None → the fdinfo could not be read because of a permission error
               (e.g. a root-owned container process read by a non-root user).
               None lets callers surface "unavailable" rather than show a
               misleading 0.

    A missing pid (process gone) returns 0, not None — liveness is handled
    separately by the caller.
    """
    fdinfo_dir = os.path.join(fdinfo_root, pid, "fdinfo")

    try:
        entries = os.listdir(fdinfo_dir)
    except PermissionError:
        return None
    except OSError:
        return 0

    return _sum_unique_clients(_parse_fdinfo_file(os.path.join(fdinfo_dir, e)) for e in entries)


# Lists every fdinfo file inside the container and prints each one's contents
# preceded by a delimiter line, so a single exec call captures all DRM clients.
# Running inside the container means we read as the container's (root) user,
# avoiding the host-side permission wall that blocks reading root-owned
# /proc/<pid>/fdinfo as a normal user.
_CONTAINER_FDINFO_SCRIPT = (
    'for f in /proc/[0-9]*/fdinfo/*; do '
    'echo "@@@$f"; cat "$f" 2>/dev/null; '
    'done'
)
_FDINFO_DELIMITER = "@@@"


def read_container_vram_kib(
    container_name: str,
    runtime: str,
    runner: Runner = _default_runner,
) -> int | None:
    """Sum DRM memory of all processes inside a running container.

    Runs `runtime exec <container> sh -c <script>` so fdinfo is read as the
    container's own (root) user, sidestepping the host permission wall.

    Returns summed KiB, or None if the exec failed (container not running,
    runtime error) so the caller can surface "unavailable".
    """
    result = runner([runtime, "exec", container_name, "sh", "-c", _CONTAINER_FDINFO_SCRIPT])

    # The in-container loop often exits non-zero because a trailing `cat` on a
    # transient/special fd fails — yet it has already printed valid fdinfo. So
    # gate on output, not exit code: only a non-zero exit with no output means
    # the exec itself failed (e.g. the container is not running).
    if result.returncode != 0 and not result.stdout.strip():
        return None

    chunks = result.stdout.split(_FDINFO_DELIMITER)
    parsed: list[tuple[str | None, dict[str, int]]] = []
    for chunk in chunks[1:]:  # chunks[0] is whatever preceded the first delimiter
        # First line is the file path emitted by `echo`; the rest is fdinfo content.
        _, _, content = chunk.partition("\n")
        parsed.append(_parse_fdinfo_text(content))

    return _sum_unique_clients(parsed)


def check_health(port: int, timeout: float = 2.0) -> ServerState:
    """
    GET http://127.0.0.1:{port}/health with timeout.

    200           → ServerState.READY
    HTTPError 503 → ServerState.LOADING
    HTTPError other → ServerState.UNHEALTHY
    URLError / OSError / TimeoutError → ServerState.STARTING
    """
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return ServerState.READY
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return ServerState.LOADING
        return ServerState.UNHEALTHY
    except (urllib.error.URLError, OSError, TimeoutError):
        return ServerState.STARTING
