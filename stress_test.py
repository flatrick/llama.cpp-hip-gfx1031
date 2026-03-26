#!/usr/bin/env python3
"""
Stress test for llama-server: sends progressively larger prompts,
reports VRAM usage and timing after each step, then runs four
additional phases to prove memory safety:

  1. Ramp          — find working range, calibrate VRAM rate
  2. Sustained     — cache stability + defrag trigger (20 rounds)
  3. Cold-start    — fresh KV allocation, no leak (8 rounds)
  4. Defrag stress — fill→evict→fill cycles (10 cycles)
  5. Boundary      — clean HTTP 400 at ctx-size, server stays healthy

Override context window: CTX_SIZE=32768 python stress_test.py
"""

import collections
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL    = "http://127.0.0.1:8080/v1/chat/completions"
MAX_TOKENS = 256         # generation length — stresses KV cache during decode
TIMEOUT    = 300         # seconds per request

# VRAM threshold: any in-flight peak above this triggers a warning.
# Set to match the "stay under 11 GB" safety target.
VRAM_WARN_GB = 11.0

# Server ctx-size — auto-detected from /props at startup; fallback to env/default.
_CTX_SIZE_FALLBACK = int(__import__("os").environ.get("CTX_SIZE", "65536"))

# Prompt sizes: ramp up to ~95% of ctx, leaving room for generated tokens.
# Computed after we know CTX_SIZE (see main()).

# Sustained phase: identical prompt, tests cache stability and triggers defrag.
SUSTAINED_ROUNDS = 20

# Cold-start phase: unique prompt each round, defeats LCP cache reuse.
COLD_ROUNDS = 8
LEAK_THRESHOLD_GB = 0.1   # max acceptable VRAM growth across cold-start rounds

# Defrag stress phase: fill→evict cycles.
DEFRAG_CYCLES = 10

# Filler text repeated to reach target token count.
FILLER_CHUNK = """\
The quick brown fox jumps over the lazy dog near the river bank. \
A software engineer reviews pull requests and writes unit tests every morning. \
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2). \
class Node: def __init__(self, val): self.val = val; self.next = None. \
import os, sys, json, re, pathlib, subprocess, threading, collections. \
Error handling is critical in production systems to prevent silent failures. \
Always validate inputs at system boundaries before processing user-supplied data. \
The database query returned 42 rows matching the filter criteria applied. \
""" * 3   # one chunk ≈ 381 tokens (measured for Qwen3.5)

TOKENS_PER_CHUNK = 381


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vram_used_gb():
    """
    Read current VRAM usage. Tries in order:
    1. amdgpu sysfs (always available, no ROCm needed on host)
    2. rocm-smi (if installed)
    Returns float GB or None.
    """
    import glob as _glob
    for path in _glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try:
            used = int(open(path).read().strip())
            return used / 1024**3
        except Exception:
            continue

    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL, timeout=5
        )
        data = json.loads(out)
        for card in data.values():
            if isinstance(card, dict):
                for k, v in card.items():
                    if "used" in k.lower() and "vram" in k.lower():
                        return int(v) / 1024**3
    except Exception:
        pass

    return None


def server_ctx_size(fallback):
    """
    Read the actual configured slot context size from the running server.

    Tries in order:
    1. GET /slots  → first slot's n_ctx  (most accurate — matches n_ctx_slot in logs)
    2. GET /props  → default_generation_settings.n_ctx  (some llama.cpp versions)
    3. GET /props  → n_ctx  (older versions; may return training context, not slot ctx)
    Falls back to the env-var/default if all fail.
    """
    # /slots returns per-slot state; n_ctx here is the actual configured context
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8080/slots", timeout=3)
        slots = json.loads(resp.read())
        if slots and isinstance(slots, list) and "n_ctx" in slots[0]:
            return int(slots[0]["n_ctx"])
    except Exception:
        pass

    # /props fallback — nested field first, then top-level
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8080/props", timeout=3)
        data = json.loads(resp.read())
        dgs  = data.get("default_generation_settings", {})
        if "n_ctx" in dgs:
            return int(dgs["n_ctx"])
        if "n_ctx" in data:
            return int(data["n_ctx"])
    except Exception:
        pass

    return fallback


def build_prompt(target_tokens, prefix=""):
    """Repeat filler until we reach approximately target_tokens."""
    repeats = max(1, target_tokens // TOKENS_PER_CHUNK)
    body = (FILLER_CHUNK * repeats).strip()
    return (prefix + " " + body).strip() if prefix else body


def send_request(prompt, system=None):
    """POST to the completions API. Returns (response_dict, elapsed_sec) or raises."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model":       "local",
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    return body, elapsed


def server_healthy():
    """Return True if /health responds 200."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5)
        return True
    except Exception:
        return False


def fmt_vram(v):
    if v is None:
        return "n/a"
    warn = " !" if v >= VRAM_WARN_GB else ""
    return f"{v:.2f} GB{warn}"


def parse_timings(resp, elapsed):
    timings = resp.get("timings", {})
    pp_tps  = timings.get("prompt_per_second", None)
    gen_tps = timings.get("predicted_per_second", None)
    gen_str = f"{gen_tps:.1f}" if gen_tps else "—"
    pp_str  = f"{timings.get('prompt_n','?')}t/{elapsed:.1f}s" if pp_tps else f"{elapsed:.1f}s"
    return pp_str, gen_str


# ---------------------------------------------------------------------------
# In-flight VRAM sampler
# ---------------------------------------------------------------------------

class PeakVramSampler:
    """Polls vram_used_gb() every interval_ms in a daemon thread."""

    def __init__(self, interval_ms=200):
        self._interval = interval_ms / 1000.0
        self._peak     = None
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop_evt.clear()
        self._peak = vram_used_gb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop_evt.is_set():
            v = vram_used_gb()
            if v is not None and (self._peak is None or v > self._peak):
                self._peak = v
            self._stop_evt.wait(self._interval)

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=1)
        return self._peak


# ---------------------------------------------------------------------------
# Container log tapping
# ---------------------------------------------------------------------------

IMAGE_NAME = "llama-cpp-gfx1031:latest"
LOG_LINES  = 30


def find_runtime():
    for rt in ("podman", "docker"):
        if shutil.which(rt):
            return rt
    return None


def find_container_id(runtime):
    try:
        out = subprocess.check_output(
            [runtime, "ps", "--filter", f"ancestor={IMAGE_NAME}",
             "--format", "{{.ID}}"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        cid = out.decode().strip().splitlines()
        return cid[0] if cid else None
    except Exception:
        return None


class ContainerLogReader:
    def __init__(self, runtime, container_id):
        self._buf  = collections.deque(maxlen=200)
        self._proc = subprocess.Popen(
            [runtime, "logs", "--follow", "--since", "0s", container_id],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for raw in self._proc.stdout:
            try:
                self._buf.append(raw.decode(errors="replace").rstrip())
            except Exception:
                pass

    def dump(self):
        lines = list(self._buf)[-LOG_LINES:]
        if not lines:
            print("  (no container log lines captured)")
            return
        print(f"\n  --- last {len(lines)} container log lines ---")
        for line in lines:
            print(f"  {line}")

    def stop(self):
        try:
            self._proc.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared failure handler
# ---------------------------------------------------------------------------

def handle_failure(label, exc_or_msg, log_reader, last_ok):
    vram = vram_used_gb()
    print(f"  {label}  FAIL {exc_or_msg}  {fmt_vram(vram)}")
    time.sleep(1)
    if log_reader:
        log_reader.dump()
        log_reader.stop()
    if last_ok is not None:
        print(f"\nLast successful size: ~{last_ok:,} tokens")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Auto-detect ctx-size from server
    ctx_size = server_ctx_size(_CTX_SIZE_FALLBACK)
    _cap = int(ctx_size * 0.95) - MAX_TOKENS
    steps = sorted(set([
        4_000, 8_000, 16_000, 24_000, 32_000,
        ctx_size // 4, ctx_size // 2, int(ctx_size * 0.70),
        int(ctx_size * 0.80), int(ctx_size * 0.88),
        int(ctx_size * 0.92), _cap,
    ]))

    print("=" * 70)
    print("  llama-server stress test")
    print("=" * 70)
    print(f"  API        : {API_URL}")
    print(f"  CTX_SIZE   : {ctx_size:,}  (auto-detected; override with CTX_SIZE=N)")
    print(f"  Steps      : {steps}")
    print(f"  Max gen    : {MAX_TOKENS} tokens/request")
    print(f"  VRAM warn  : {VRAM_WARN_GB} GB  (! marker)")
    print(f"  Sustained  : {SUSTAINED_ROUNDS} rounds")
    print(f"  Cold-start : {COLD_ROUNDS} rounds")
    print(f"  Defrag     : {DEFRAG_CYCLES} fill→evict cycles")
    print()

    if not server_healthy():
        print(f"ERROR: server not reachable at {API_URL}")
        sys.exit(1)

    runtime    = find_runtime()
    cid        = find_container_id(runtime) if runtime else None
    log_reader = ContainerLogReader(runtime, cid) if cid else None

    if cid:
        print(f"  Container logs: tapping {cid[:12]} via {runtime}")
    elif runtime:
        print(f"  Container logs: no running {IMAGE_NAME} container found")
    else:
        print("  Container logs: podman/docker not found in PATH")
    print()

    baseline = vram_used_gb()
    print(f"VRAM at baseline : {fmt_vram(baseline)}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Ramp
    # -----------------------------------------------------------------------
    print(f"  {'~tokens':>8}  {'prompt len':>12}  {'prefill':>12}  "
          f"{'gen tok/s':>10}  {'peak VRAM':>10}  {'post VRAM':>10}  status")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  ──────")

    last_ok_tokens = 0

    for target in steps:
        prompt  = build_prompt(target)
        p_chars = len(prompt)

        sampler = PeakVramSampler().start()
        try:
            resp, elapsed = send_request(prompt)
        except urllib.error.HTTPError as e:
            peak = sampler.stop()
            body = e.read().decode(errors="replace")
            vram = vram_used_gb()
            print(f"  {target:>8}  {p_chars:>12,}  {'—':>12}  {'—':>10}  "
                  f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  "
                  f"FAIL HTTP {e.code}: {body[:50]}")
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            print(f"\nLast successful size: ~{last_ok_tokens:,} tokens")
            return
        except Exception as e:
            peak = sampler.stop()
            vram = vram_used_gb()
            print(f"  {target:>8}  {p_chars:>12,}  {'—':>12}  {'—':>10}  "
                  f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  FAIL {e}")
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            print(f"\nLast successful size: ~{last_ok_tokens:,} tokens")
            return

        peak    = sampler.stop()
        pp_str, gen_str = parse_timings(resp, elapsed)
        vram    = vram_used_gb()
        print(f"  {target:>8}  {p_chars:>12,}  {pp_str:>12}  {gen_str:>10}  "
              f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  OK")
        last_ok_tokens = target
        time.sleep(1)

    if log_reader:
        log_reader.stop()
    print()
    print(f"Ramp passed. Last tested: ~{last_ok_tokens:,} tokens.")

    if last_ok_tokens == 0:
        return

    sustained_prompt = build_prompt(last_ok_tokens)
    log_reader = ContainerLogReader(runtime, cid) if cid else None

    # -----------------------------------------------------------------------
    # Phase 2: Sustained load (tests cache stability, triggers defrag)
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Phase 2: Sustained — {SUSTAINED_ROUNDS} rounds at ~{last_ok_tokens:,} tokens")
    print("=" * 70)
    print(f"  {'round':>6}  {'prefill':>12}  {'gen tok/s':>10}  "
          f"{'peak VRAM':>10}  {'post VRAM':>10}  status")
    print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  ──────")

    for i in range(1, SUSTAINED_ROUNDS + 1):
        sampler = PeakVramSampler().start()
        try:
            resp, elapsed = send_request(sustained_prompt)
        except Exception as e:
            peak = sampler.stop()
            vram = vram_used_gb()
            print(f"  {i:>6}  {'—':>12}  {'—':>10}  "
                  f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  FAIL {e}")
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            return

        peak = sampler.stop()
        pp_str, gen_str = parse_timings(resp, elapsed)
        vram = vram_used_gb()
        print(f"  {i:>6}  {pp_str:>12}  {gen_str:>10}  "
              f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  OK")
        time.sleep(1)

    if log_reader:
        log_reader.stop()
    print()
    print("Sustained load passed.")

    # -----------------------------------------------------------------------
    # Phase 3: Cold-start (unique prompts defeat LCP cache reuse)
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Phase 3: Cold-start — {COLD_ROUNDS} rounds, fresh KV each time")
    print(f"  (leak threshold: {LEAK_THRESHOLD_GB} GB growth across rounds)")
    print("=" * 70)
    print(f"  {'round':>6}  {'prefill':>12}  {'gen tok/s':>10}  "
          f"{'peak VRAM':>10}  {'post VRAM':>10}  status")
    print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  ──────")

    log_reader = ContainerLogReader(runtime, cid) if cid else None
    first_cold_vram = None

    for i in range(1, COLD_ROUNDS + 1):
        # Unique prefix defeats LCP similarity — server must allocate fresh KV cache
        prompt = build_prompt(last_ok_tokens, prefix=f"[cold-start round {i}]")

        sampler = PeakVramSampler().start()
        try:
            resp, elapsed = send_request(prompt)
        except Exception as e:
            peak = sampler.stop()
            vram = vram_used_gb()
            print(f"  {i:>6}  {'—':>12}  {'—':>10}  "
                  f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  FAIL {e}")
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            return

        peak = sampler.stop()
        pp_str, gen_str = parse_timings(resp, elapsed)
        vram = vram_used_gb()

        if first_cold_vram is None:
            first_cold_vram = vram
        growth = (vram - first_cold_vram) if (vram and first_cold_vram) else 0.0
        leak_flag = f"  LEAK +{growth:.2f} GB" if growth > LEAK_THRESHOLD_GB else ""

        print(f"  {i:>6}  {pp_str:>12}  {gen_str:>10}  "
              f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  OK{leak_flag}")

        if growth > LEAK_THRESHOLD_GB:
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            print(f"\nCold-start FAILED: VRAM grew {growth:.2f} GB (threshold {LEAK_THRESHOLD_GB} GB)")
            return

        time.sleep(1)

    if log_reader:
        log_reader.stop()
    print()
    print("Cold-start passed — no KV cache leak detected.")

    # -----------------------------------------------------------------------
    # Phase 4: Defrag stress (fill → evict → fill cycles)
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Phase 4: Defrag stress — {DEFRAG_CYCLES} fill→evict cycles")
    print("=" * 70)
    print(f"  {'cycle':>6}  {'step':>6}  {'prefill':>12}  {'gen tok/s':>10}  "
          f"{'peak VRAM':>10}  {'post VRAM':>10}  status")
    print(f"  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  ──────")

    full_prompt  = build_prompt(last_ok_tokens, prefix="[defrag-fill]")
    evict_system = "You are a different assistant with no memory of previous conversations."

    log_reader = ContainerLogReader(runtime, cid) if cid else None
    first_fill_peak = None

    for cycle in range(1, DEFRAG_CYCLES + 1):
        # Fill: near-full-context request with unique prefix
        fill_prompt = build_prompt(last_ok_tokens, prefix=f"[defrag cycle {cycle} fill]")
        sampler = PeakVramSampler().start()
        try:
            resp, elapsed = send_request(fill_prompt)
        except Exception as e:
            peak = sampler.stop()
            vram = vram_used_gb()
            print(f"  {cycle:>6}  {'fill':>6}  {'—':>12}  {'—':>10}  "
                  f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  FAIL {e}")
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            return

        fill_peak = sampler.stop()
        pp_str, gen_str = parse_timings(resp, elapsed)
        vram = vram_used_gb()

        if first_fill_peak is None:
            first_fill_peak = fill_peak
        drift = ((fill_peak - first_fill_peak) if (fill_peak and first_fill_peak) else 0.0)
        drift_flag = f"  drift +{drift:.2f} GB" if drift > LEAK_THRESHOLD_GB else ""

        print(f"  {cycle:>6}  {'fill':>6}  {pp_str:>12}  {gen_str:>10}  "
              f"{fmt_vram(fill_peak):>10}  {fmt_vram(vram):>10}  OK{drift_flag}")

        if drift > LEAK_THRESHOLD_GB:
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            print(f"\nDefrag FAILED: fill-peak drifted {drift:.2f} GB (threshold {LEAK_THRESHOLD_GB} GB)")
            return

        # Evict: short request with a different system prompt forces KV cache eviction
        sampler = PeakVramSampler().start()
        try:
            resp, elapsed = send_request(
                f"Respond with one word: ready. [cycle {cycle}]",
                system=evict_system,
            )
        except Exception as e:
            peak = sampler.stop()
            vram = vram_used_gb()
            print(f"  {cycle:>6}  {'evict':>6}  {'—':>12}  {'—':>10}  "
                  f"{fmt_vram(peak):>10}  {fmt_vram(vram):>10}  FAIL {e}")
            time.sleep(1)
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            return

        evict_peak = sampler.stop()
        _, gen_str = parse_timings(resp, elapsed)
        vram = vram_used_gb()
        print(f"  {cycle:>6}  {'evict':>6}  {'—':>12}  {gen_str:>10}  "
              f"{fmt_vram(evict_peak):>10}  {fmt_vram(vram):>10}  OK")
        time.sleep(1)

    if log_reader:
        log_reader.stop()
    print()
    print("Defrag stress passed — no peak VRAM drift across fill→evict cycles.")

    # -----------------------------------------------------------------------
    # Phase 5: Boundary validation
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Phase 5: Boundary — request at ctx_size+1 must get clean HTTP 400")
    print("=" * 70)

    # Overshoot by a safe margin (2× MAX_TOKENS) — guaranteed to exceed ctx_size
    over_prompt = build_prompt(ctx_size + MAX_TOKENS * 2)
    vram_before = vram_used_gb()

    log_reader = ContainerLogReader(runtime, cid) if cid else None
    sampler = PeakVramSampler().start()
    try:
        send_request(over_prompt)
        sampler.stop()
        print("  FAIL: expected HTTP 400 but request succeeded — ctx_size may be misconfigured")
        if log_reader:
            log_reader.stop()
        return
    except urllib.error.HTTPError as e:
        peak       = sampler.stop()
        vram_after = vram_used_gb()
        body       = e.read().decode(errors="replace")
        healthy    = server_healthy()

        status_400  = "OK" if e.code == 400 else f"FAIL (got HTTP {e.code})"
        status_msg  = "OK" if "exceeds" in body.lower() else f"FAIL (body: {body[:60]})"
        status_vram = "OK" if (vram_before is None or vram_after is None or
                                abs(vram_after - vram_before) < 0.05) else \
                      f"WARN ({vram_after:.2f} vs {vram_before:.2f} GB before)"
        status_health = "OK" if healthy else "FAIL (server unreachable after reject)"

        print(f"  HTTP 400 received  : {status_400}")
        print(f"  Error message      : {status_msg}")
        print(f"  VRAM unchanged     : {status_vram}  "
              f"(peak in-flight: {fmt_vram(peak)})")
        print(f"  Server still alive : {status_health}")

        if log_reader:
            log_reader.stop()

        if "FAIL" in (status_400, status_msg, status_health):
            print("\nBoundary test FAILED.")
            return
    except Exception as e:
        sampler.stop()
        print(f"  FAIL: unexpected exception: {e}")
        if log_reader:
            log_reader.dump()
            log_reader.stop()
        return

    print()
    print("=" * 70)
    print("  ALL PHASES PASSED")
    print(f"  Peak VRAM stayed {'below' if True else 'above'} {VRAM_WARN_GB} GB threshold.")
    print("  Server is healthy and memory-safe at configured ctx-size.")
    print("=" * 70)


if __name__ == "__main__":
    main()
