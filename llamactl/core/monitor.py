"""VRAM inspection and health polling for a running llama-server.

read_vram_kib: memory-only subset of vram_inspect.py's fdinfo reader.
check_health: polls /health and maps HTTP status to ServerState.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from llamactl.core.lifecycle import ServerInfo, ServerState
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


def _dedupe_clients(
    parsed: Iterable[tuple[str | None, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    """Map of unique DRM client-id → memory fields (first seen wins).

    Fds sharing a drm-client-id are kept once. Entries with no client-id are
    each treated as unique (synthetic keys). Entries with no fields are ignored.
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

    return seen_clients


@dataclass(frozen=True, slots=True)
class VramSnapshot:
    """A point-in-time VRAM reading, broken down for leak investigation.

    total_kib: sum of all memory fields across unique clients.
    by_field:  per memory category (drm-memory-vram, drm-memory-gtt, …), summed.
    by_client: per drm-client-id breakdown (first-seen fields).
    """
    total_kib: int
    by_field: dict[str, int]
    by_client: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class VramDelta:
    """Signed change between two VramSnapshots."""
    total_kib: int
    by_field: dict[str, int]


def _snapshot_from_parsed(
    parsed: Iterable[tuple[str | None, dict[str, int]]],
) -> VramSnapshot:
    """Aggregate parsed fdinfo into a VramSnapshot (dedup by client-id)."""
    clients = _dedupe_clients(parsed)
    by_field: dict[str, int] = {}
    for fields in clients.values():
        for key, val in fields.items():
            by_field[key] = by_field.get(key, 0) + val
    return VramSnapshot(
        total_kib=sum(by_field.values()),
        by_field=by_field,
        by_client=clients,
    )


def vram_delta(before: VramSnapshot, after: VramSnapshot) -> VramDelta:
    """Per-category signed change (after − before), for `--delta`-style leak checks."""
    keys = set(before.by_field) | set(after.by_field)
    by_field = {
        key: after.by_field.get(key, 0) - before.by_field.get(key, 0)
        for key in keys
    }
    return VramDelta(total_kib=after.total_kib - before.total_kib, by_field=by_field)


def read_vram_snapshot(pid: str, fdinfo_root: str = "/proc") -> VramSnapshot | None:
    """
    Per-client / per-category VRAM snapshot for PID, or None.

    Reads {fdinfo_root}/{pid}/fdinfo/, parses each fd, deduplicates by
    drm-client-id, and aggregates memory fields.

    Returns:
      - VramSnapshot → breakdown (total_kib == 0 if the pid has no DRM fds)
      - None         → fdinfo unreadable due to a permission error (e.g. a
                       root-owned container process read by a non-root user),
                       so callers can surface "unavailable" rather than a 0.

    A missing pid (process gone) returns an empty snapshot, not None — liveness
    is handled separately by the caller.
    """
    fdinfo_dir = os.path.join(fdinfo_root, pid, "fdinfo")

    try:
        entries = os.listdir(fdinfo_dir)
    except PermissionError:
        return None
    except OSError:
        return _snapshot_from_parsed([])

    return _snapshot_from_parsed(
        _parse_fdinfo_file(os.path.join(fdinfo_dir, e)) for e in entries
    )


def read_vram_kib(pid: str, fdinfo_root: str = "/proc") -> int | None:
    """Total VRAM (KiB) for PID, or None if unreadable (permission error).

    Thin total-only view over read_vram_snapshot; see it for the full contract.
    """
    snap = read_vram_snapshot(pid, fdinfo_root)
    return None if snap is None else snap.total_kib


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


def read_container_vram_snapshot(
    container_name: str,
    runtime: str,
    runner: Runner = _default_runner,
) -> VramSnapshot | None:
    """Per-client / per-category VRAM snapshot for a running container, or None.

    Runs `runtime exec <container> sh -c <script>` so fdinfo is read as the
    container's own (root) user, sidestepping the host permission wall. Returns
    None if the exec failed (container not running, runtime error).
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

    return _snapshot_from_parsed(parsed)


def read_container_vram_kib(
    container_name: str,
    runtime: str,
    runner: Runner = _default_runner,
) -> int | None:
    """Total VRAM (KiB) inside a running container, or None if the exec failed.

    Thin total-only view over read_container_vram_snapshot.
    """
    snap = read_container_vram_snapshot(container_name, runtime, runner=runner)
    return None if snap is None else snap.total_kib


def read_server_vram_snapshot(
    server: ServerInfo, runtime: str | None
) -> VramSnapshot | None:
    """Per-client / per-category VRAM snapshot for a managed server, or None.

    Native: read the server's own /proc/<pid>/fdinfo.
    Container: read fdinfo inside the container via `exec` (needs runtime).
    """
    if server.mode == "native":
        if server.pid is None:
            return None
        return read_vram_snapshot(str(server.pid))
    if server.container_name and runtime is not None:
        return read_container_vram_snapshot(server.container_name, runtime)
    return None


def read_server_vram_kib(server: ServerInfo, runtime: str | None) -> int | None:
    """Total VRAM (KiB) for a managed server, or None if it cannot be read.

    Thin total-only view over read_server_vram_snapshot.
    """
    snap = read_server_vram_snapshot(server, runtime)
    return None if snap is None else snap.total_kib


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
