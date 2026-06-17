"""Pre-launch VRAM estimate.

Ports the GGUF-metadata VRAM math from vram_calc.py (calibrated for this repo's
gfx1031 setup) and resolves the model's `org/repo:QUANT` -hf spec against the
local llama.cpp cache. Local-only: no network access. Returns None ("estimate
unavailable") when the GGUF is not on disk; never blocks launch.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from llamactl.core.config import GlobalConfig, ModelConfig

# Calibrated constants (ported from vram_calc.py).
CACHE_TYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "f32": 4.0, "q8_0": 1.0, "q5_1": 0.6875,
    "q5_0": 0.625, "q4_1": 0.5625, "q4_0": 0.5,
}
VRAM_OVERHEAD_GB = 0.6
COMPUTE_BUFFER_PER_512_GB = 0.9
_SAFE_SPEC_PART = re.compile(r"^[A-Za-z0-9._-]+$")


# ---------------------------------------------------------------------------
# GGUF metadata reader (ported verbatim from vram_calc.py)
# Self-contained: only stdlib struct + os. Reads only the header.
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"

# GGUF value-type IDs
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12


def _read_string(f: Any) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f: Any, vtype: int) -> Any:
    if vtype == _UINT8:   return struct.unpack("<B", f.read(1))[0]
    if vtype == _INT8:    return struct.unpack("<b", f.read(1))[0]
    if vtype == _UINT16:  return struct.unpack("<H", f.read(2))[0]
    if vtype == _INT16:   return struct.unpack("<h", f.read(2))[0]
    if vtype == _UINT32:  return struct.unpack("<I", f.read(4))[0]
    if vtype == _INT32:   return struct.unpack("<i", f.read(4))[0]
    if vtype == _FLOAT32: return struct.unpack("<f", f.read(4))[0]
    if vtype == _BOOL:    return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == _UINT64:  return struct.unpack("<Q", f.read(8))[0]
    if vtype == _INT64:   return struct.unpack("<q", f.read(8))[0]
    if vtype == _FLOAT64: return struct.unpack("<d", f.read(8))[0]
    if vtype == _STRING:  return _read_string(f)
    if vtype == _ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count     = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf_metadata(path: str) -> dict[str, Any]:
    """Return dict of all metadata key→value from a GGUF file header."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path}")
        version    = struct.unpack("<I", f.read(4))[0]
        _tc        = struct.unpack("<Q", f.read(8))[0]   # tensor count (unused)
        kv_count   = struct.unpack("<Q", f.read(8))[0]
        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key   = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            meta[key] = _read_value(f, vtype)
    return meta


def _resolve_kv_layers(meta: dict[str, Any], arch: str, block_count: int) -> int:
    """Number of layers that hold a context-scaling KV cache.

    Hybrid models (e.g. Qwen3.5 Gated Delta Net) interleave full-attention layers
    with SSM/linear-attention layers that have NO ctx-scaling KV cache. Counting
    every block as a KV layer massively overestimates VRAM.

    Order: explicit attention_layer_count → derive from full_attention_interval
    (1 full-attention layer every N blocks) → fall back to block_count.
    """
    explicit = meta.get(
        f"{arch}.attention_layer_count",
        meta.get(f"{arch}.attention.layer_count"),
    )
    if explicit:
        return explicit
    interval = meta.get(f"{arch}.full_attention_interval")
    if interval:
        return max(1, block_count // interval)
    return block_count


def model_params_from_gguf(path: str) -> dict[str, Any]:
    """
    Extract the parameters we need for VRAM calculation from a GGUF file.

    Returns dict with keys:
      arch, block_count, kv_layers, kv_heads, head_dim, weight_gb
    kv_layers may differ from block_count for hybrid SSM/attention models.
    """
    meta = read_gguf_metadata(path)
    arch = meta.get("general.architecture", "llm")

    block_count = meta.get(f"{arch}.block_count", 0)

    # Hybrid models (e.g. Qwen3.5 Gated Delta Net) have fewer attention layers
    # than total blocks — only those hold a ctx-scaling KV cache.
    kv_layers = _resolve_kv_layers(meta, arch, block_count)

    kv_heads = meta.get(
        f"{arch}.attention.head_count_kv",
        meta.get(f"{arch}.attention.kv_heads", 0)
    )

    # head_dim: prefer explicit key, else derive from embedding / total heads
    head_dim = meta.get(f"{arch}.attention.key_length", 0)
    if not head_dim:
        embed  = meta.get(f"{arch}.embedding_length", 0)
        n_head = meta.get(f"{arch}.attention.head_count", 1)
        head_dim = embed // n_head if n_head else 0

    weight_gb = os.path.getsize(path) / 1024**3

    return {
        "arch":         arch,
        "block_count":  block_count,
        "kv_layers":    kv_layers,
        "kv_heads":     kv_heads,
        "head_dim":     head_dim,
        "weight_gb":    weight_gb,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)

_DEFAULT_CTX = 4096
_DEFAULT_BATCH = 512


def _positive_int(value: Any) -> bool:
    """True for a usable scalar count. Rejects bools, non-ints, and arrays
    (some GGUFs store head_count_kv / key_length as per-layer arrays, which we
    don't model)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _compute_buffer_batch(settings: dict[str, Any]) -> int:
    """Token count that drives the graph compute buffer.

    llama.cpp sizes the compute buffer for the *physical* micro-batch
    (`--ubatch-size` / `n_ubatch`), not the logical `--batch-size`. Keying the
    estimate off batch_size overcounted VRAM whenever ubatch < batch (the common
    case: ubatch 256, batch 1024 over-stated the buffer by ~1.35 GB). Prefer
    ubatch_size, fall back to batch_size, then the default.
    """
    return int(
        settings.get("ubatch_size")
        or settings.get("batch_size")
        or _DEFAULT_BATCH
    )


@dataclass(frozen=True, slots=True)
class Estimate:
    total_gb: float
    model_gb: float
    kv_gb: float
    compute_gb: float
    overhead_gb: float


def compute_estimate(
    *,
    params: dict[str, Any],
    ctx_size: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
    batch_size: int,
) -> Estimate:
    """Pure VRAM math from GGUF params + resolved settings. Units are GiB."""
    # Unknown/None cache types fall back to f16 (2.0 B) — a conservative
    # upper-bound default for cache types not yet in CACHE_TYPE_BYTES.
    k_bytes = CACHE_TYPE_BYTES.get(cache_type_k or "f16", 2.0)
    v_bytes = CACHE_TYPE_BYTES.get(cache_type_v or "f16", 2.0)
    model_gb = float(params["weight_gb"])
    kv_gb = (
        (k_bytes + v_bytes)
        * params["kv_layers"]
        * params["kv_heads"]
        * params["head_dim"]
        * ctx_size
        / 1024 ** 3
    )
    compute_gb = COMPUTE_BUFFER_PER_512_GB * (batch_size / 512)
    total = model_gb + kv_gb + compute_gb + VRAM_OVERHEAD_GB
    return Estimate(
        total_gb=total,
        model_gb=model_gb,
        kv_gb=kv_gb,
        compute_gb=compute_gb,
        overhead_gb=VRAM_OVERHEAD_GB,
    )


def resolve_gguf_path(hf_spec: str, global_cfg: GlobalConfig) -> Path | None:
    """Resolve an `org/repo:QUANT` -hf spec to a local .gguf path, or None.

    Order: llama.cpp -hf cache (flattened names) → HF hub snapshot layout.
    Local-only; never raises.
    """
    repo_part, _, quant = hf_spec.partition(":")
    if not quant or "/" not in repo_part:
        return None
    org, repo = repo_part.split("/", 1)

    # Validate components before interpolating into glob/path patterns: reject
    # path-traversal and glob metacharacters in operator-supplied `hf` specs.
    if not all(_SAFE_SPEC_PART.match(part) for part in (org, repo, quant)):
        _log.warning("estimate: rejecting unsafe hf spec %r", hf_spec)
        return None

    # Resolution prefers REPO-SPECIFIC matches; the repo-less fallback is used
    # only when it is unambiguous. In a shared cache, several models share a
    # quant (e.g. UD-Q5_K_XL), so a greedy repo-less glob could resolve to a
    # completely different (much larger) model and wildly mis-estimate VRAM.

    # 1. llama.cpp -hf cache: flattened filenames containing repo + quant.
    llama_cache = global_cfg.llama_cache
    matches = sorted(glob.glob(str(llama_cache / f"*{repo}*{quant}*.gguf")))
    if matches:
        _log.debug("estimate: resolved %s in llama_cache (repo+quant) -> %s",
                   hf_spec, matches[-1])
        return Path(matches[-1])

    # 2. HF hub snapshot layout (huggingface-cli downloads), repo-specific. With
    #    multiple snapshots we take the lexicographically last; single-snapshot
    #    repos (the common case) are unambiguous.
    hf_cache = global_cfg.hf_cache
    hub = hf_cache / "hub" if (hf_cache / "hub").is_dir() else hf_cache
    pattern = str(hub / f"models--{org}--{repo}" / "snapshots" / "*" / f"*{quant}*.gguf")
    matches = sorted(glob.glob(pattern))
    if matches:
        _log.debug("estimate: resolved %s via HF hub layout -> %s",
                   hf_spec, matches[-1])
        return Path(matches[-1])

    # 3. Repo-less last resort for caches whose filenames omit the repo. Only
    #    accept it when EXACTLY ONE file carries this quant — otherwise the match
    #    is ambiguous across models and we must not guess.
    matches = sorted(glob.glob(str(llama_cache / f"*{quant}*.gguf")))
    if len(matches) == 1:
        _log.debug("estimate: resolved %s via unique repo-less quant match -> %s",
                   hf_spec, matches[0])
        return Path(matches[0])
    if len(matches) > 1:
        _log.warning(
            "estimate: %r matches %d models sharing quant %r in llama_cache; "
            "refusing to guess (estimate unavailable)", hf_spec, len(matches), quant
        )

    return None


def _load_params(
    model: ModelConfig, global_cfg: GlobalConfig
) -> dict[str, Any] | None:
    """Resolve a model's GGUF and read its VRAM params, or None if the file is
    absent / unreadable / lacks KV-cache metadata. Never raises.

    Shared by estimate_vram and sweep_vram so both apply the same
    "unavailable rather than wrong" guarantee.
    """
    path = resolve_gguf_path(model.hf, global_cfg)
    if path is None:
        return None
    try:
        params = model_params_from_gguf(str(path))
    except Exception as exc:  # corrupt/unreadable header
        _log.warning("estimate: cannot read GGUF %s: %s", path, exc)
        return None
    if not _positive_int(params.get("kv_heads")) or not _positive_int(params.get("head_dim")):
        _log.warning(
            "estimate: GGUF %s lacks usable scalar KV-cache metadata "
            "(kv_heads=%r, head_dim=%r); estimate unavailable rather than "
            "under-counting or mis-multiplying KV", path,
            params.get("kv_heads"), params.get("head_dim"),
        )
        return None
    return params


def estimate_vram(
    model: ModelConfig,
    resolved_settings: dict[str, Any],
    global_cfg: GlobalConfig,
) -> Estimate | None:
    """Estimate VRAM (GiB) for a model + resolved settings, or None if the GGUF
    is not locally available / not readable. Never raises."""
    params = _load_params(model, global_cfg)
    if params is None:
        return None
    return compute_estimate(
        params=params,
        ctx_size=int(resolved_settings.get("ctx_size", _DEFAULT_CTX)),
        cache_type_k=resolved_settings.get("cache_type_k"),
        cache_type_v=resolved_settings.get("cache_type_v"),
        batch_size=_compute_buffer_batch(resolved_settings),
    )


# ---------------------------------------------------------------------------
# VRAM what-if sweep (ports vram_calc.py's ctx-size / cache-type tables).
# A 2-D grid of total VRAM across context sizes × cache types, each classified
# OK / TIGHT / OOM against the configured VRAM budget — decision support for
# picking ctx_size and cache precision before launch.
# ---------------------------------------------------------------------------

# Default sweep axes (ported from vram_calc.py's what-if tables).
SWEEP_CTX_SIZES: tuple[int, ...] = (8192, 16384, 32768, 65536, 131072)
SWEEP_CACHE_TYPES: tuple[str, ...] = ("f16", "q8_0", "q5_0", "q4_0")
# Headroom below which a fitting estimate is flagged TIGHT (ported from vram_calc.py).
TIGHT_HEADROOM_GB = 0.5


@dataclass(frozen=True, slots=True)
class SweepCell:
    ctx_size: int
    cache_type: str
    estimate: Estimate
    status: str  # "ok" | "tight" | "oom"


def classify_headroom(total_gb: float, budget_gb: float) -> str:
    """Classify an estimate against a VRAM budget: oom / tight / ok.

    Negative headroom is OOM; less than TIGHT_HEADROOM_GB (inclusive of 0) is
    TIGHT; otherwise OK.
    """
    headroom = budget_gb - total_gb
    if headroom < 0:
        return "oom"
    if headroom < TIGHT_HEADROOM_GB:
        return "tight"
    return "ok"


def estimate_sweep(
    *,
    params: dict[str, Any],
    base_settings: dict[str, Any],
    budget_gb: float,
    ctx_sizes: Sequence[int] = SWEEP_CTX_SIZES,
    cache_types: Sequence[str] = SWEEP_CACHE_TYPES,
) -> list[SweepCell]:
    """Build the ctx × cache VRAM grid for already-loaded GGUF params.

    Reuses compute_estimate for every cell (no new VRAM math); both K and V
    cache use the row's cache type, matching vram_calc.py's cache table. The
    non-swept settings (e.g. ubatch/batch size) come from base_settings.
    """
    batch_size = _compute_buffer_batch(base_settings)
    cells: list[SweepCell] = []
    for ctx in ctx_sizes:
        for cache in cache_types:
            est = compute_estimate(
                params=params,
                ctx_size=ctx,
                cache_type_k=cache,
                cache_type_v=cache,
                batch_size=batch_size,
            )
            cells.append(
                SweepCell(ctx, cache, est, classify_headroom(est.total_gb, budget_gb))
            )
    return cells


def sweep_vram(
    model: ModelConfig,
    base_settings: dict[str, Any],
    global_cfg: GlobalConfig,
    *,
    ctx_sizes: Sequence[int] = SWEEP_CTX_SIZES,
    cache_types: Sequence[str] = SWEEP_CACHE_TYPES,
) -> list[SweepCell] | None:
    """What-if VRAM grid for a model, or None if its GGUF is unavailable.

    Resolves + reads the GGUF once (same availability guarantee as
    estimate_vram) and classifies against global_cfg.vram_budget_gb.
    """
    params = _load_params(model, global_cfg)
    if params is None:
        return None
    return estimate_sweep(
        params=params,
        base_settings=base_settings,
        budget_gb=global_cfg.vram_budget_gb,
        ctx_sizes=ctx_sizes,
        cache_types=cache_types,
    )
