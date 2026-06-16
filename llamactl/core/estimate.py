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
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llamactl.core.config import GlobalConfig, ModelConfig

# Calibrated constants (ported from vram_calc.py).
CACHE_TYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "f32": 4.0, "q8_0": 1.0, "q5_1": 0.6875,
    "q5_0": 0.625, "q4_1": 0.5625, "q4_0": 0.5,
}
VRAM_OVERHEAD_GB = 0.6
COMPUTE_BUFFER_PER_512_GB = 0.9


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
    # than total blocks. Look for an explicit attention_layer_count first.
    kv_layers = meta.get(
        f"{arch}.attention_layer_count",
        meta.get(f"{arch}.attention.layer_count", block_count)
    )

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

    # 1. llama.cpp -hf cache: flattened filenames containing repo + quant.
    #    The repo-less `*{quant}*.gguf` fallback is a last resort for caches
    #    whose filenames omit the repo; in a shared cache it can match a
    #    different model that uses the same quant.
    llama_cache = global_cfg.llama_cache
    for pattern in (f"*{repo}*{quant}*.gguf", f"*{quant}*.gguf"):
        matches = sorted(glob.glob(str(llama_cache / pattern)))
        if matches:
            return Path(matches[-1])

    # 2. HF hub snapshot layout (huggingface-cli downloads). With multiple
    #    snapshots we take the lexicographically last; single-snapshot repos
    #    (the common case) are unambiguous.
    hf_cache = global_cfg.hf_cache
    hub = hf_cache / "hub" if (hf_cache / "hub").is_dir() else hf_cache
    pattern = str(hub / f"models--{org}--{repo}" / "snapshots" / "*" / f"*{quant}*.gguf")
    matches = sorted(glob.glob(pattern))
    if matches:
        return Path(matches[-1])

    return None


def estimate_vram(
    model: ModelConfig,
    resolved_settings: dict[str, Any],
    global_cfg: GlobalConfig,
) -> Estimate | None:
    """Estimate VRAM (GiB) for a model + resolved settings, or None if the GGUF
    is not locally available / not readable. Never raises."""
    path = resolve_gguf_path(model.hf, global_cfg)
    if path is None:
        return None
    try:
        params = model_params_from_gguf(str(path))
    except Exception as exc:  # corrupt/unreadable header
        _log.warning("estimate: cannot read GGUF %s: %s", path, exc)
        return None
    if not params.get("kv_heads") or not params.get("head_dim"):
        _log.warning(
            "estimate: GGUF %s lacks KV-cache metadata (kv_heads=%s, head_dim=%s); "
            "estimate unavailable rather than under-counting KV", path,
            params.get("kv_heads"), params.get("head_dim"),
        )
        return None
    return compute_estimate(
        params=params,
        ctx_size=int(resolved_settings.get("ctx_size", _DEFAULT_CTX)),
        cache_type_k=resolved_settings.get("cache_type_k"),
        cache_type_v=resolved_settings.get("cache_type_v"),
        batch_size=int(resolved_settings.get("batch_size", _DEFAULT_BATCH)),
    )
