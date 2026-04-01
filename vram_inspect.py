#!/usr/bin/env python3
"""
vram_inspect.py — inspect GPU memory fields from /proc/{pid}/fdinfo.

Deduplicates DRM clients by drm-client-id so shared BOs are not double-counted.

Modes
─────
  (no args)       one snapshot of the auto-detected server process
  --watch [N]     refresh every N seconds (default 2); Ctrl-C to stop
  --delta         print baseline, wait for Enter, print after + delta
  --pid PID       target a specific PID instead of auto-detecting
  --port PORT     find process listening on PORT (default 8080)
  --all-fields    show engine/cycle/freq fields in addition to memory

Examples
────────
  python vram_inspect.py
  python vram_inspect.py --watch
  python vram_inspect.py --watch 0.5
  python vram_inspect.py --delta
  python vram_inspect.py --pid 12345 --all-fields
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime


# ── PID discovery ────────────────────────────────────────────────────────────

def _find_pid_for_port(port: int) -> str | None:
    hex_port = f"{port:04X}"
    inodes: set[str] = set()
    for tcp_path in ["/proc/net/tcp"] + glob.glob("/proc/*/net/tcp"):
        try:
            with open(tcp_path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    if parts[3] == "0A" and parts[1].endswith(f":{hex_port}"):
                        inodes.add(parts[9])
        except OSError:
            continue
    if not inodes:
        return None
    for fd_path in glob.glob("/proc/*/fd/*"):
        try:
            pid = fd_path.split("/")[2]
            if not pid.isdigit():
                continue
            target = os.readlink(fd_path)
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return pid
        except OSError:
            continue
    return None


def _find_pid_by_cmdline(name: str) -> str | None:
    for path in glob.glob("/proc/*/cmdline"):
        try:
            pid = path.split("/")[2]
            if not pid.isdigit():
                continue
            with open(path, "rb") as fh:
                cmdline = fh.read().decode(errors="replace").replace("\x00", " ")
            if name in cmdline:
                return pid
        except OSError:
            continue
    return None


def resolve_pid(pid_arg: str | None, port: int) -> str:
    if pid_arg:
        return pid_arg
    pid = _find_pid_for_port(port)
    if pid:
        return pid
    pid = _find_pid_by_cmdline("llama-server")
    if pid:
        return pid
    print(f"ERROR: no process found on port {port} and no llama-server in /proc",
          file=sys.stderr)
    sys.exit(1)


def process_name(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/comm") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


# ── fdinfo parsing ────────────────────────────────────────────────────────────

def _parse_fdinfo_file(path: str) -> tuple[str | None, dict[str, tuple[int, str]]]:
    """Return (client_id, {field: (value, unit)}) for one fdinfo file."""
    client_id: str | None = None
    fields: dict[str, tuple[int, str]] = {}
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("drm-"):
                    continue
                key, sep, raw = line.partition(":")
                if not sep:
                    continue
                key = key.strip()
                raw = raw.strip()
                if key == "drm-client-id":
                    client_id = raw
                    continue
                parts = raw.split(None, 1)
                if not parts:
                    continue
                try:
                    val = int(parts[0])
                except ValueError:
                    continue
                unit = parts[1].strip() if len(parts) > 1 else ""
                fields[key] = (val, unit)
    except OSError:
        pass
    return client_id, fields


def read_fdinfo(pid: str) -> tuple[dict[str, dict], int]:
    """
    Returns:
        (aggregated, n_clients)
        aggregated: {field: {"total": int, "per_client": [int,...], "unit": str}}
        n_clients: number of unique DRM clients found
    """
    fdinfo_dir = f"/proc/{pid}/fdinfo"
    try:
        fd_names = os.listdir(fdinfo_dir)
    except OSError:
        return {}, 0

    # Deduplicate by drm-client-id.  Fds sharing the same client-id report
    # identical values (same DRM context) and must not be summed.
    seen_clients: dict[str, dict[str, tuple[int, str]]] = {}
    anon_index = 0

    for fd_name in fd_names:
        client_id, fields = _parse_fdinfo_file(f"{fdinfo_dir}/{fd_name}")
        if not fields:
            continue
        key = client_id if client_id is not None else f"_anon_{anon_index}"
        if client_id is None:
            anon_index += 1
        if key not in seen_clients:
            seen_clients[key] = fields

    # Aggregate across unique clients
    agg: dict[str, dict] = {}
    for client_fields in seen_clients.values():
        for field, (val, unit) in client_fields.items():
            if field not in agg:
                agg[field] = {"total": 0, "per_client": [], "unit": unit}
            agg[field]["total"] += val
            agg[field]["per_client"].append(val)

    return agg, len(seen_clients)


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt_kib(kib: float, signed: bool = False) -> str:
    gib = kib / (1024 ** 2)
    if abs(gib) >= 0.05:
        s = f"{gib:+.3f} GiB" if signed else f"{gib:.3f} GiB"
    else:
        mib = kib / 1024
        s = f"{mib:+.2f} MiB" if signed else f"{mib:.2f} MiB"
    return s


def _fmt_ns(ns: int, signed: bool = False) -> str:
    if abs(ns) >= 1_000_000_000:
        s = f"{ns / 1e9:.2f} s"
    elif abs(ns) >= 1_000_000:
        s = f"{ns / 1e6:.1f} ms"
    elif abs(ns) >= 1_000:
        s = f"{ns / 1e3:.1f} µs"
    else:
        s = f"{ns} ns"
    return ("+" if signed and ns >= 0 else "") + s


def _fmt_generic(val: int, unit: str, signed: bool = False) -> str:
    prefix = "+" if signed and val >= 0 else ""
    if unit in ("KiB",):
        return _fmt_kib(val, signed)
    if unit == "ns":
        return _fmt_ns(val, signed)
    if unit in ("Hz",):
        if abs(val) >= 1_000_000:
            return f"{prefix}{val / 1e6:.0f} MHz"
        return f"{prefix}{val:,} {unit}"
    return f"{prefix}{val:,} {unit}".strip()


W_FIELD = 34
W_VAL = 14
SEP = "─"


def _section(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {SEP * (W_FIELD + W_VAL * 2 + 6)}")


def _header_row(col1: str, col2: str, col3: str) -> None:
    print(f"  {col1:<{W_FIELD}}  {col2:>{W_VAL}}  {col3:>{W_VAL}}")
    print(f"  {SEP * W_FIELD}  {SEP * W_VAL}  {SEP * W_VAL}")


# ── snapshot printing ─────────────────────────────────────────────────────────

def print_snapshot(pid: str, agg: dict, n_clients: int, label: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    name = process_name(pid)
    header = f"{name} (PID {pid})"
    if label:
        header += f"  —  {label}"
    print(f"\n{'═' * 70}")
    print(f"  {header}")
    print(f"  DRM clients: {n_clients}   |   {ts}")
    print(f"{'═' * 70}")

    mem = {k: v for k, v in agg.items() if "memory" in k}
    other = {k: v for k, v in agg.items() if "memory" not in k}

    if mem:
        _section("Memory  (unique DRM clients, summed)")
        _header_row("Field", "Total", "Per-client breakdown")
        mem_total_kib = 0.0
        for field in sorted(mem):
            d = mem[field]
            total_str = _fmt_generic(d["total"], d["unit"])
            per = "  ".join(_fmt_kib(v) for v in d["per_client"])
            print(f"  {field:<{W_FIELD}}  {total_str:>{W_VAL}}  {per}")
            if d["unit"] == "KiB":
                mem_total_kib += d["total"]
        print(f"  {SEP * W_FIELD}  {SEP * W_VAL}")
        print(f"  {'TOTAL (all memory fields)':<{W_FIELD}}  {_fmt_kib(mem_total_kib):>{W_VAL}}")

    if other:
        _section("Engine / cycles / freq  (since process start)")
        print(f"  {'Field':<{W_FIELD}}  {'Value':>{W_VAL}}  {'Unit'}")
        print(f"  {SEP * W_FIELD}  {SEP * W_VAL}  {SEP * 8}")
        for field in sorted(other):
            d = other[field]
            print(f"  {field:<{W_FIELD}}  {_fmt_generic(d['total'], d['unit']):>{W_VAL}}  {d['unit']}")


# ── delta printing ────────────────────────────────────────────────────────────

def print_delta(before: dict, after: dict) -> None:
    print(f"\n{'═' * 70}")
    print(f"  DELTA  (after − before)")
    print(f"{'═' * 70}")

    all_fields = sorted(set(before) | set(after))
    mem_fields = [f for f in all_fields if "memory" in f]
    other_fields = [f for f in all_fields if "memory" not in f]

    if mem_fields:
        _section("Memory delta")
        W3 = W_VAL
        print(f"  {'Field':<{W_FIELD}}  {'Before':>{W3}}  {'After':>{W3}}  {'Delta':>{W3}}")
        print(f"  {SEP*W_FIELD}  {SEP*W3}  {SEP*W3}  {SEP*W3}")
        for field in mem_fields:
            b_data = before.get(field, {"total": 0, "unit": "KiB"})
            a_data = after.get(field, {"total": 0, "unit": "KiB"})
            unit = a_data.get("unit") or b_data.get("unit", "KiB")
            b_val = b_data["total"]
            a_val = a_data["total"]
            delta = a_val - b_val
            b_s = _fmt_generic(b_val, unit)
            a_s = _fmt_generic(a_val, unit)
            d_s = _fmt_generic(delta, unit, signed=True)
            print(f"  {field:<{W_FIELD}}  {b_s:>{W3}}  {a_s:>{W3}}  {d_s:>{W3}}")

    if other_fields:
        _section("Engine / freq delta")
        print(f"  {'Field':<{W_FIELD}}  {'Before':>{W_VAL}}  {'After':>{W_VAL}}  {'Delta':>{W_VAL}}")
        print(f"  {SEP*W_FIELD}  {SEP*W_VAL}  {SEP*W_VAL}  {SEP*W_VAL}")
        for field in other_fields:
            b_data = before.get(field, {"total": 0, "unit": ""})
            a_data = after.get(field, {"total": 0, "unit": ""})
            unit = a_data.get("unit") or b_data.get("unit", "")
            b_val = b_data["total"]
            a_val = a_data["total"]
            delta = a_val - b_val
            b_s = _fmt_generic(b_val, unit)
            a_s = _fmt_generic(a_val, unit)
            d_s = _fmt_generic(delta, unit, signed=True)
            print(f"  {field:<{W_FIELD}}  {b_s:>{W_VAL}}  {a_s:>{W_VAL}}  {d_s:>{W_VAL}}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect GPU drm-fdinfo memory fields for a running llama-server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pid", help="Target PID (auto-detected if omitted)")
    parser.add_argument("--port", type=int, default=8080,
                        help="API port for auto-detection (default: 8080)")
    parser.add_argument("--watch", type=float, nargs="?", const=2.0, metavar="SECS",
                        help="Refresh every SECS seconds (default 2); Ctrl-C to stop")
    parser.add_argument("--delta", action="store_true",
                        help="Take baseline, wait for Enter, take after snapshot, show delta")
    args = parser.parse_args()

    pid = resolve_pid(args.pid, args.port)

    def snapshot(label: str = "") -> dict:
        agg, n = read_fdinfo(pid)
        print_snapshot(pid, agg, n, label)
        return agg

    if args.delta:
        before = snapshot("BEFORE")
        input("\n  ── Press Enter to capture the AFTER snapshot ──")
        after = snapshot("AFTER")
        print_delta(before, after)

    elif args.watch is not None:
        print(f"Watching PID {pid} every {args.watch}s — Ctrl-C to stop")
        try:
            while True:
                snapshot()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")

    else:
        snapshot()


if __name__ == "__main__":
    main()
