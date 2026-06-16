"""VRAM inspection and health polling for a running llama-server.

read_vram_kib: memory-only subset of vram_inspect.py's fdinfo reader.
check_health: polls /health and maps HTTP status to ServerState.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

from llamactl.core.lifecycle import ServerState


def _parse_fdinfo_file(path: str) -> tuple[str | None, dict[str, int]]:
    """
    Reads one fdinfo file. Returns (drm_client_id, {field_name: kib_value}).
    - Only processes lines starting with "drm-"
    - "drm-client-id" sets the client_id
    - Lines where "memory" is NOT in the key name are skipped
    - Value is the integer before the first space/tab on the value side
    - Lines that don't parse as int are skipped
    - OSError on open → return (None, {})
    """
    client_id: str | None = None
    fields: dict[str, int] = {}

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
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
                    fields[key] = int(parts[0])
                except ValueError:
                    continue
    except OSError:
        return (None, {})

    return (client_id, fields)


def read_vram_kib(pid: str, fdinfo_root: str = "/proc") -> int:
    """
    Sum all drm-*-memory-* KiB fields across unique DRM clients for PID.

    1. List files in {fdinfo_root}/{pid}/fdinfo/
    2. For each file, call _parse_fdinfo_file
    3. Deduplicate by drm-client-id: if two fds share the same client-id, count only once
       (anonymous fds without client-id each get a unique key "_anon_N")
    4. Sum all field values across unique clients
    5. Return 0 if pid doesn't exist or has no DRM fds
    """
    fdinfo_dir = os.path.join(fdinfo_root, pid, "fdinfo")

    try:
        entries = os.listdir(fdinfo_dir)
    except OSError:
        return 0

    seen_clients: dict[str, dict[str, int]] = {}
    anon_counter = 0

    for entry in entries:
        path = os.path.join(fdinfo_dir, entry)
        client_id, fields = _parse_fdinfo_file(path)

        if not fields:
            continue

        if client_id is not None:
            key = client_id
        else:
            key = f"_anon_{anon_counter}"
            anon_counter += 1

        # Deduplicate: first seen wins (all duplicates share identical data)
        if key not in seen_clients:
            seen_clients[key] = fields

    total = 0
    for client_fields in seen_clients.values():
        total += sum(client_fields.values())

    return total


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
