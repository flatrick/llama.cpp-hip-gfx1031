from __future__ import annotations

from llamactl.core.estimate import Estimate, compute_estimate


def test_compute_estimate_breakdown():
    params = {
        "kv_layers": 32, "kv_heads": 8, "head_dim": 128, "weight_gb": 7.0,
    }
    est = compute_estimate(
        params=params,
        ctx_size=4096,
        cache_type_k="q8_0",   # 1.0 byte
        cache_type_v="q8_0",   # 1.0 byte
        batch_size=512,
    )
    # KV = (1.0 + 1.0) * 32 * 8 * 128 * 4096 / 1024**3
    expected_kv = (1.0 + 1.0) * 32 * 8 * 128 * 4096 / 1024**3
    assert est.model_gb == 7.0
    assert abs(est.kv_gb - expected_kv) < 1e-9
    assert abs(est.compute_gb - 0.9) < 1e-9          # 0.9 * (512/512)
    assert abs(est.overhead_gb - 0.6) < 1e-9
    assert abs(est.total_gb - (7.0 + expected_kv + 0.9 + 0.6)) < 1e-9


def test_compute_estimate_defaults_to_f16_cache():
    params = {"kv_layers": 1, "kv_heads": 1, "head_dim": 1, "weight_gb": 1.0}
    est = compute_estimate(params=params, ctx_size=1, cache_type_k=None,
                           cache_type_v=None, batch_size=512)
    # f16 = 2.0 bytes each → (2+2) * 1 * 1 * 1 * 1 / 1024**3
    assert abs(est.kv_gb - (4.0 / 1024**3)) < 1e-15


from pathlib import Path

from llamactl.core.estimate import resolve_gguf_path


def _global_cfg(tmp_path: Path):
    from llamactl.core.config import GlobalConfig
    return GlobalConfig(
        llama_cache=tmp_path / "llama",
        hf_cache=tmp_path / "hf",
    )


def test_resolve_finds_flattened_gguf_in_llama_cache(tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    target = cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-UD-Q5_K_XL.gguf"
    target.write_bytes(b"GGUF")
    got = resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg)
    assert got == target


def test_resolve_falls_back_to_hf_hub_layout(tmp_path):
    cfg = _global_cfg(tmp_path)
    snap = cfg.hf_cache / "models--unsloth--Qwen3.5-9B-GGUF" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    target = snap / "Qwen3.5-9B-UD-Q5_K_XL.gguf"
    target.write_bytes(b"GGUF")
    got = resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg)
    assert got == target


def test_resolve_returns_none_when_absent(tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    assert resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg) is None


def test_resolve_returns_none_on_malformed_spec(tmp_path):
    cfg = _global_cfg(tmp_path)
    assert resolve_gguf_path("no-colon-here", cfg) is None
