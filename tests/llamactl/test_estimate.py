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
