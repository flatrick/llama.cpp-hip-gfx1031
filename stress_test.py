#!/usr/bin/env python3
"""
Stress test for llama-server: sends progressively larger prompts,
reports VRAM usage and timing after each step, then hammers the server
with repeated near-full-context requests to surface memory fragmentation.

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
MAX_TOKENS = 256         # generation length — longer to stress KV cache during decode
TIMEOUT    = 300         # seconds per request

# Server ctx-size. Steps are generated relative to this so we bracket the
# actual limit rather than wasting time on sizes that will always fail.
CTX_SIZE = int(__import__("os").environ.get("CTX_SIZE", "65536"))

# Prompt sizes: ramp up to ~95% of ctx, leaving room for generated tokens.
# Fine-grained near the limit to find the exact breaking point.
_cap = int(CTX_SIZE * 0.95) - MAX_TOKENS
STEPS = sorted(set([
    4_000, 8_000, 16_000, 24_000, 32_000,
    CTX_SIZE // 4, CTX_SIZE // 2, int(CTX_SIZE * 0.70),
    int(CTX_SIZE * 0.80), int(CTX_SIZE * 0.88),
    int(CTX_SIZE * 0.92), _cap,
]))

# After the ramp, hammer the server with this many back-to-back requests at
# the largest successful prompt size to surface memory leaks / fragmentation.
SUSTAINED_ROUNDS = 5

# Filler text repeated to reach target token count.
# ~200 words ≈ ~270 tokens (ratio 1.35 tokens/word is typical for code+prose).
FILLER_CHUNK = """\
The quick brown fox jumps over the lazy dog near the river bank. \
A software engineer reviews pull requests and writes unit tests every morning. \
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2). \
class Node: def __init__(self, val): self.val = val; self.next = None. \
import os, sys, json, re, pathlib, subprocess, threading, collections. \
Error handling is critical in production systems to prevent silent failures. \
Always validate inputs at system boundaries before processing user-supplied data. \
The database query returned 42 rows matching the filter criteria applied. \
""" * 3   # ~270 tokens per repetition of the above block

TOKENS_PER_CHUNK = 381   # measured: Qwen3.5 tokenises this filler at ~381 tok/chunk


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
    # --- amdgpu sysfs ---
    import glob as _glob
    for path in _glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try:
            used  = int(open(path).read().strip())
            return used / 1024**3
        except Exception:
            continue

    # --- rocm-smi fallback ---
    try:
        out  = subprocess.check_output(
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


def build_prompt(target_tokens):
    """Repeat filler until we reach approximately target_tokens."""
    repeats = max(1, target_tokens // TOKENS_PER_CHUNK)
    return (FILLER_CHUNK * repeats).strip()


def send_request(prompt):
    """POST to the completions API. Returns (response_dict, elapsed_sec) or raises."""
    payload = json.dumps({
        "model":       "local",
        "messages":    [{"role": "user", "content": prompt}],
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


def fmt_vram(v):
    return f"{v:.2f} GB" if v is not None else "n/a"


# ---------------------------------------------------------------------------
# Container log tapping
# ---------------------------------------------------------------------------

IMAGE_NAME = "llama-cpp-gfx1031:latest"
LOG_LINES  = 30   # lines to dump on failure


def find_runtime():
    for rt in ("podman", "docker"):
        if shutil.which(rt):
            return rt
    return None


def find_container_id(runtime):
    """Return the ID of the running container for our image, or None."""
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
    """
    Reads container logs in a background daemon thread into a ring buffer.
    On failure, call dump() to print the last LOG_LINES lines.
    """

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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  llama-server stress test — progressive context fill")
    print("=" * 65)
    print(f"  API      : {API_URL}")
    print(f"  CTX_SIZE : {CTX_SIZE:,}  (override with CTX_SIZE=N)")
    print(f"  Steps    : {STEPS}")
    print(f"  Max gen  : {MAX_TOKENS} tokens/request")
    print(f"  Sustained: {SUSTAINED_ROUNDS} rounds at peak size")
    print()

    # Verify server is up
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5)
    except Exception as e:
        print(f"ERROR: server not reachable at {API_URL}\n  {e}")
        sys.exit(1)

    # Attach to container logs (silent during normal run, dumped on failure)
    log_reader = None
    runtime    = find_runtime()
    if runtime:
        cid = find_container_id(runtime)
        if cid:
            log_reader = ContainerLogReader(runtime, cid)
            print(f"  Container logs: tapping {cid[:12]} via {runtime}")
        else:
            print(f"  Container logs: no running {IMAGE_NAME} container found")
    else:
        print("  Container logs: podman/docker not found in PATH")
    print()

    baseline = vram_used_gb()
    print(f"VRAM at baseline : {fmt_vram(baseline)}")
    print()
    print(f"  {'~tokens':>8}  {'prompt len':>12}  {'prefill s':>10}  "
          f"{'gen tok/s':>10}  {'VRAM':>9}  status")
    print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*9}  ──────")

    last_ok_tokens = 0

    for target in STEPS:
        prompt  = build_prompt(target)
        p_chars = len(prompt)

        try:
            resp, elapsed = send_request(prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            vram = vram_used_gb()
            print(f"  {target:>8}  {p_chars:>12,}  {'—':>10}  {'—':>10}  "
                  f"{fmt_vram(vram):>9}  FAIL HTTP {e.code}: {body[:60]}")
            time.sleep(1)   # let log reader catch the crash lines
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            print(f"\nLast successful size: ~{last_ok_tokens:,} tokens")
            return
        except Exception as e:
            vram = vram_used_gb()
            print(f"  {target:>8}  {p_chars:>12,}  {'—':>10}  {'—':>10}  "
                  f"{fmt_vram(vram):>9}  FAIL {e}")
            time.sleep(1)   # let log reader catch the crash lines
            if log_reader:
                log_reader.dump()
                log_reader.stop()
            print(f"\nLast successful size: ~{last_ok_tokens:,} tokens")
            return

        # Parse timing from response if available
        timings   = resp.get("timings", {})
        pp_tps    = timings.get("prompt_per_second", None)
        gen_tps   = timings.get("predicted_per_second", None)
        gen_str   = f"{gen_tps:.1f}" if gen_tps else "—"
        pp_str    = f"{timings.get('prompt_n', '?')}t/{elapsed:.1f}s" if pp_tps else f"{elapsed:.1f}s"

        vram = vram_used_gb()
        print(f"  {target:>8}  {p_chars:>12,}  {pp_str:>10}  {gen_str:>10}  "
              f"{fmt_vram(vram):>9}  OK")
        last_ok_tokens = target

        # Brief pause so VRAM reading settles
        time.sleep(1)

    if log_reader:
        log_reader.stop()
    print()
    print(f"All ramp steps passed. Last tested: ~{last_ok_tokens:,} tokens.")

    if last_ok_tokens == 0:
        return

    # ------------------------------------------------------------------
    # Sustained-load phase: repeated near-full-context requests
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"  Sustained load — {SUSTAINED_ROUNDS} rounds at ~{last_ok_tokens:,} tokens")
    print("=" * 65)
    print(f"  {'round':>6}  {'prefill':>10}  {'gen tok/s':>10}  {'VRAM':>9}  status")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*9}  ──────")

    sustained_prompt = build_prompt(last_ok_tokens)
    runtime = find_runtime()
    if runtime:
        cid = find_container_id(runtime)
        log_reader2 = ContainerLogReader(runtime, cid) if cid else None
    else:
        log_reader2 = None

    for i in range(1, SUSTAINED_ROUNDS + 1):
        try:
            resp, elapsed = send_request(sustained_prompt)
        except Exception as e:
            vram = vram_used_gb()
            print(f"  {i:>6}  {'—':>10}  {'—':>10}  {fmt_vram(vram):>9}  FAIL {e}")
            time.sleep(1)
            if log_reader2:
                log_reader2.dump()
                log_reader2.stop()
            return

        timings  = resp.get("timings", {})
        gen_tps  = timings.get("predicted_per_second", None)
        gen_str  = f"{gen_tps:.1f}" if gen_tps else "—"
        pp_tps   = timings.get("prompt_per_second", None)
        pp_str   = f"{timings.get('prompt_n','?')}t/{elapsed:.1f}s" if pp_tps else f"{elapsed:.1f}s"
        vram     = vram_used_gb()
        print(f"  {i:>6}  {pp_str:>10}  {gen_str:>10}  {fmt_vram(vram):>9}  OK")
        time.sleep(1)

    if log_reader2:
        log_reader2.stop()
    print()
    print("Sustained load passed — no fragmentation or OOM detected.")


if __name__ == "__main__":
    main()
