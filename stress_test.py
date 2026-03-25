#!/usr/bin/env python3
"""
Stress test for llama-server: sends progressively larger prompts,
reports VRAM usage and timing after each step.
Stops at first failure or when the target context size is reached.
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
MAX_TOKENS = 64          # generation length — keep short, we're testing prefill
TIMEOUT    = 300         # seconds per request

# Prompt sizes to test (in approximate tokens).
# Each step fills the context a bit more.
STEPS = [4_000, 8_000, 16_000, 24_000, 32_000, 48_000, 64_000,
         80_000, 96_000, 112_000, 124_000]

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

TOKENS_PER_CHUNK = 270   # approximate; adjust if model tokenises differently


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
    print(f"  API : {API_URL}")
    print(f"  Steps: {STEPS}")
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
    print(f"All steps passed. Last tested: ~{last_ok_tokens:,} tokens.")


if __name__ == "__main__":
    main()
