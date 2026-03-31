from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_FILLER_CHUNK = """\
The quick brown fox jumps over the lazy dog near the river bank. \
A software engineer reviews pull requests and writes unit tests every morning. \
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2). \
class Node: def __init__(self, val): self.val = val; self.next = None. \
import os, sys, json, re, pathlib, subprocess, threading, collections. \
Error handling is critical in production systems to prevent silent failures. \
Always validate inputs at system boundaries before processing user-supplied data. \
The database query returned 42 rows matching the filter criteria applied. \
""" * 3


def _env_int(environ: dict[str, str], key: str, default: int) -> int:
    return int(environ.get(key, str(default)))


def _env_float(environ: dict[str, str], key: str, default: float) -> float:
    return float(environ.get(key, str(default)))


@dataclass(frozen=True, slots=True)
class StressConfig:
    api_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    max_tokens: int = 256
    request_timeout: int = 1800
    stall_timeout: int = 180
    watchdog_poll_s: float = 5.0
    vram_warn_gb: float = 11.0
    ctx_size_override: int | None = None
    ctx_size_fallback: int = 65536
    sustained_rounds: int = 20
    cold_rounds: int = 8
    leak_threshold_gb: float = 0.1
    defrag_cycles: int = 10
    filler_chunk: str = DEFAULT_FILLER_CHUNK
    tokens_per_chunk: int = 381
    log_lines: int = 30

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "StressConfig":
        env = dict(os.environ if environ is None else environ)
        defaults = cls()
        ctx_override = env.get("CTX_SIZE")
        return cls(
            api_url=env.get("API_URL", defaults.api_url),
            max_tokens=_env_int(env, "MAX_TOKENS", defaults.max_tokens),
            request_timeout=_env_int(env, "REQUEST_TIMEOUT", defaults.request_timeout),
            stall_timeout=_env_int(env, "STALL_TIMEOUT", defaults.stall_timeout),
            watchdog_poll_s=_env_float(env, "WATCHDOG_POLL_S", defaults.watchdog_poll_s),
            vram_warn_gb=_env_float(env, "VRAM_WARN_GB", defaults.vram_warn_gb),
            ctx_size_override=int(ctx_override) if ctx_override is not None else defaults.ctx_size_override,
            ctx_size_fallback=int(ctx_override) if ctx_override is not None else defaults.ctx_size_fallback,
            sustained_rounds=_env_int(env, "SUSTAINED_ROUNDS", defaults.sustained_rounds),
            cold_rounds=_env_int(env, "COLD_ROUNDS", defaults.cold_rounds),
            leak_threshold_gb=_env_float(env, "LEAK_THRESHOLD_GB", defaults.leak_threshold_gb),
            defrag_cycles=_env_int(env, "DEFRAG_CYCLES", defaults.defrag_cycles),
        )

    def build_steps(self, ctx_size: int) -> list[int]:
        cap = int(ctx_size * 0.95) - self.max_tokens
        return sorted(set([
            4_000,
            8_000,
            16_000,
            24_000,
            32_000,
            ctx_size // 4,
            ctx_size // 2,
            int(ctx_size * 0.70),
            int(ctx_size * 0.80),
            int(ctx_size * 0.88),
            int(ctx_size * 0.92),
            cap,
        ]))
