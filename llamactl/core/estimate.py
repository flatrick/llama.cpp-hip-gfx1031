"""Pre-launch VRAM estimate.

Ports the GGUF-metadata VRAM math from vram_calc.py (calibrated for this repo's
gfx1031 setup) and resolves the model's `org/repo:QUANT` -hf spec against the
local llama.cpp cache. Local-only: no network access. Returns None ("estimate
unavailable") when the GGUF is not on disk; never blocks launch.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llamactl.core.config import GlobalConfig

# Calibrated constants (ported from vram_calc.py).
CACHE_TYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "f32": 4.0, "q8_0": 1.0, "q5_1": 0.6875,
    "q5_0": 0.625, "q4_1": 0.5625, "q4_0": 0.5,
}
VRAM_OVERHEAD_GB = 0.6
COMPUTE_BUFFER_PER_512_GB = 0.9


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
    if ":" not in hf_spec or "/" not in hf_spec.split(":", 1)[0]:
        return None
    repo_part, quant = hf_spec.split(":", 1)
    org, repo = repo_part.split("/", 1)

    # 1. llama.cpp -hf cache: flattened filenames containing repo + quant.
    llama_cache = Path(global_cfg.llama_cache).expanduser()
    for pattern in (f"*{repo}*{quant}*.gguf", f"*{quant}*.gguf"):
        matches = sorted(glob.glob(str(llama_cache / pattern)))
        if matches:
            return Path(matches[-1])

    # 2. HF hub snapshot layout (huggingface-cli downloads).
    hf_cache = Path(global_cfg.hf_cache).expanduser()
    hub = hf_cache / "hub" if (hf_cache / "hub").is_dir() else hf_cache
    pattern = str(hub / f"models--{org}--{repo}" / "snapshots" / "*" / f"*{quant}*.gguf")
    matches = sorted(glob.glob(pattern))
    if matches:
        return Path(matches[-1])

    return None
