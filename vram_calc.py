#!/usr/bin/env python3
"""
VRAM budget calculator for llama.cpp inference.
Can read architecture parameters directly from a local GGUF file.
"""

import os
import glob
import struct

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_TYPE_BYTES = {
    "f16":  2.0,
    "f32":  4.0,
    "q8_0": 1.0,
    "q5_1": 0.6875,
    "q5_0": 0.625,
    "q4_1": 0.5625,
    "q4_0": 0.5,
}

VRAM_OVERHEAD_GB = 0.6   # ROCm/CUDA runtime + misc fixed cost
# Empirical compute-buffer baseline for a ~9B model at batch-size 512.
# Scales roughly linearly with batch size.
COMPUTE_BUFFER_PER_512_GB = 0.9

# ---------------------------------------------------------------------------
# GGUF metadata reader (self-contained, no extra dependencies)
# Reads only the header — never loads tensor data.
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"

# GGUF value-type IDs
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12


def _read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f, vtype):
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


def read_gguf_metadata(path):
    """Return dict of all metadata key→value from a GGUF file header."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path}")
        version    = struct.unpack("<I", f.read(4))[0]
        _tc        = struct.unpack("<Q", f.read(8))[0]   # tensor count (unused)
        kv_count   = struct.unpack("<Q", f.read(8))[0]
        meta = {}
        for _ in range(kv_count):
            key   = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            meta[key] = _read_value(f, vtype)
    return meta


def model_params_from_gguf(path):
    """
    Extract the parameters we need for VRAM calculation from a GGUF file.

    Returns dict with keys:
      arch, block_count, kv_layer_count, kv_heads, head_dim, weight_gb
    kv_layer_count may differ from block_count for hybrid SSM/attention models.
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
# HuggingFace cache helpers
# ---------------------------------------------------------------------------

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def hf_url_to_cache_path(url):
    """
    Convert a HuggingFace resolve URL to its local cache path (if downloaded).
    URL format: https://huggingface.co/{org}/{repo}/resolve/{ref}/{filename}
    Cache path: ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/*/{filename}
    """
    try:
        # strip scheme + host
        parts = url.split("huggingface.co/", 1)[1].split("/")
        org, repo = parts[0], parts[1]
        # parts[2] == "resolve", parts[3] == ref, rest is filename
        filename = "/".join(parts[4:])
        pattern = os.path.join(
            HF_CACHE, f"models--{org}--{repo}", "snapshots", "*", filename
        )
        matches = sorted(glob.glob(pattern))
        return matches[-1] if matches else None
    except Exception:
        return None


def find_gguf_in_cache():
    """List all .gguf files found in the HuggingFace cache."""
    return sorted(glob.glob(os.path.join(HF_CACHE, "**", "*.gguf"), recursive=True))


# ---------------------------------------------------------------------------
# Calculation helpers
# ---------------------------------------------------------------------------

def kv_cache_gb(kv_layers, kv_heads, head_dim, ctx_size, cache_bytes):
    """KV cache = 2 (K+V) × layers × heads × head_dim × ctx × bytes/element."""
    return 2 * kv_layers * kv_heads * head_dim * ctx_size * cache_bytes / 1024**3


def compute_buffer_gb(batch_size):
    """Rough estimate; scales linearly with batch size from empirical baseline."""
    return COMPUTE_BUFFER_PER_512_GB * (batch_size / 512)


def print_summary(label, model_gb, kv_gb, compute_gb, vram_total_gb):
    total    = model_gb + kv_gb + compute_gb + VRAM_OVERHEAD_GB
    headroom = vram_total_gb - total
    fit = "OK" if headroom > 0.5 else ("TIGHT" if headroom > 0 else "OOM")
    print(f"  {label}")
    print(f"    model weights : {model_gb:.2f} GB")
    print(f"    KV cache      : {kv_gb:.2f} GB")
    print(f"    compute buffer: {compute_gb:.2f} GB")
    print(f"    ROCm overhead : {VRAM_OVERHEAD_GB:.2f} GB")
    print(f"    {'─'*35}")
    print(f"    TOTAL         : {total:.2f} GB  (headroom: {headroom:+.2f} GB)  [{fit}]")
    print()
    return total, headroom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  llama.cpp VRAM budget calculator")
    print("=" * 60)
    print()

    # --- GPU ---
    vram_gb = float(input("GPU VRAM in GB [12]: ").strip() or "12")
    print()

    # --- Model source ---
    print("Model source:")
    print("  1. Enter a HuggingFace model URL (reads from local cache)")
    print("  2. Enter a local GGUF file path")
    print("  3. Choose from cached GGUF files")
    print("  4. Enter parameters manually")
    src = input("Choose [1]: ").strip() or "1"
    print()

    gguf_path = None

    if src == "1":
        url = input("HuggingFace URL: ").strip()
        gguf_path = hf_url_to_cache_path(url)
        if gguf_path:
            print(f"  Found: {gguf_path}")
        else:
            print("  Not found in local cache — falling back to manual input.")

    elif src == "2":
        p = input("GGUF file path: ").strip()
        if os.path.isfile(p):
            gguf_path = p
        else:
            print(f"  File not found: {p}")

    elif src == "3":
        cached = find_gguf_in_cache()
        if cached:
            for i, p in enumerate(cached, 1):
                size_gb = os.path.getsize(p) / 1024**3
                print(f"  {i}. {os.path.basename(p)}  ({size_gb:.1f} GB)")
            idx = input("Choose: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(cached):
                gguf_path = cached[int(idx) - 1]
        else:
            print("  No GGUF files found in HuggingFace cache.")

    # Try to read from GGUF
    kv_layers = kv_heads = head_dim = None
    model_gb = None

    if gguf_path:
        try:
            print(f"\n  Reading GGUF metadata from {os.path.basename(gguf_path)} ...")
            p = model_params_from_gguf(gguf_path)
            kv_layers = p["kv_layers"]
            kv_heads  = p["kv_heads"]
            head_dim  = p["head_dim"]
            model_gb  = p["weight_gb"]
            print(f"  Architecture   : {p['arch']}")
            print(f"  Total blocks   : {p['block_count']}")
            print(f"  KV layers      : {kv_layers}  {'(hybrid model)' if kv_layers < p['block_count'] else ''}")
            print(f"  KV heads       : {kv_heads}")
            print(f"  Head dim       : {head_dim}")
            print(f"  Weight size    : {model_gb:.2f} GB")
        except Exception as e:
            print(f"  Could not read GGUF metadata: {e}")

    # Fall back to manual input for any missing values
    if model_gb is None:
        model_gb = float(input("\nModel weight size in GB: ").strip())
    if kv_layers is None:
        kv_layers = int(input("KV layers (layers with attention KV cache): ").strip())
    if kv_heads is None:
        kv_heads = int(input("KV heads per layer: ").strip())
    if head_dim is None:
        head_dim = int(input("Head dimension: ").strip())
    print()

    # --- Inference settings ---
    print(f"KV cache types: {', '.join(CACHE_TYPE_BYTES)}")
    cache_k    = input("  cache-type-k [q8_0]: ").strip() or "q8_0"
    cache_v    = input("  cache-type-v [q8_0]: ").strip() or "q8_0"
    cache_bytes = (CACHE_TYPE_BYTES[cache_k] + CACHE_TYPE_BYTES[cache_v]) / 2

    ctx_size   = int(input("ctx-size [65536]: ").strip() or "65536")
    batch_size = int(input("batch-size [512]: ").strip() or "512")
    print()

    # --- Results ---
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print()

    kv_gb      = kv_cache_gb(kv_layers, kv_heads, head_dim, ctx_size, cache_bytes)
    compute_gb = compute_buffer_gb(batch_size)
    print_summary(
        f"ctx={ctx_size}  batch={batch_size}  cache={cache_k}/{cache_v}",
        model_gb, kv_gb, compute_gb, vram_gb,
    )

    # --- What-if: ctx-size ---
    print("What-if: vary ctx-size (same batch/cache)")
    print(f"  {'ctx':>8}  {'KV GB':>6}  {'total GB':>9}  {'headroom':>9}  status")
    print(f"  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*9}  ──────")
    for ctx in [8192, 16384, 32768, 65536, 131072]:
        kv  = kv_cache_gb(kv_layers, kv_heads, head_dim, ctx, cache_bytes)
        tot = model_gb + kv + compute_gb + VRAM_OVERHEAD_GB
        hr  = vram_gb - tot
        st  = "OK" if hr > 0.5 else ("TIGHT" if hr > 0 else "OOM")
        mk  = " ←" if ctx == ctx_size else ""
        print(f"  {ctx:>8}  {kv:>6.2f}  {tot:>9.2f}  {hr:>+9.2f}  {st}{mk}")
    print()

    # --- What-if: cache type ---
    print("What-if: vary cache type (same ctx/batch)")
    print(f"  {'cache':>12}  {'KV GB':>6}  {'total GB':>9}  {'headroom':>9}  status")
    print(f"  {'─'*12}  {'─'*6}  {'─'*9}  {'─'*9}  ──────")
    for ct in ["f16", "q8_0", "q5_0", "q4_0"]:
        cb  = CACHE_TYPE_BYTES[ct]
        kv  = kv_cache_gb(kv_layers, kv_heads, head_dim, ctx_size, cb)
        tot = model_gb + kv + compute_gb + VRAM_OVERHEAD_GB
        hr  = vram_gb - tot
        st  = "OK" if hr > 0.5 else ("TIGHT" if hr > 0 else "OOM")
        mk  = " ←" if ct == cache_k else ""
        print(f"  {ct+'/'+ct:>12}  {kv:>6.2f}  {tot:>9.2f}  {hr:>+9.2f}  {st}{mk}")
    print()

    print("Notes:")
    print("  - Compute buffer estimate is empirical (~9B model, batch-512 baseline).")
    print("  - Flash attention adds dynamic working memory during long prompt processing.")
    print("  - Keep at least 0.5 GB headroom; 1 GB+ is comfortable.")
    print("  - For hybrid SSM/attention models, kv_layers < block_count reduces KV")
    print("    cache dramatically — check GGUF metadata rather than assuming all layers.")


if __name__ == "__main__":
    main()
