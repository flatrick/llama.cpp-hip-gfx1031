from __future__ import annotations

from pathlib import Path

from llamactl.core.config import GlobalConfig, ModelConfig
from llamactl.core.estimate import Estimate, compute_estimate, estimate_vram, resolve_gguf_path


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


def _global_cfg(tmp_path: Path) -> GlobalConfig:
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


def test_resolve_returns_none_on_empty_quant(tmp_path):
    # `org/repo:` (empty quant) must NOT degrade to a catch-all `*.gguf` glob.
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "some_other_model-Q4_K_M.gguf").write_bytes(b"GGUF")
    assert resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:", cfg) is None


def test_resolve_uses_quant_only_fallback_when_repo_absent(tmp_path):
    # Cache filename omits the repo name → only the `*{quant}*.gguf` fallback matches.
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    target = cfg.llama_cache / "UD-Q5_K_XL.gguf"
    target.write_bytes(b"GGUF")
    got = resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg)
    assert got == target


def test_resolve_handles_hf_hub_subdir_layout(tmp_path):
    # Real HF cache nests models under a `hub/` subdirectory.
    cfg = _global_cfg(tmp_path)
    snap = (cfg.hf_cache / "hub" / "models--unsloth--Qwen3.5-9B-GGUF"
            / "snapshots" / "abc")
    snap.mkdir(parents=True)
    target = snap / "Qwen3.5-9B-UD-Q5_K_XL.gguf"
    target.write_bytes(b"GGUF")
    got = resolve_gguf_path("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL", cfg)
    assert got == target


def _model(hf: str) -> ModelConfig:
    return ModelConfig(
        id="m", name="m", hf=hf, settings={}, backends={}, presets={}, images={},
        path=Path("x.toml"),
    )


def test_estimate_vram_returns_none_when_gguf_absent(tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    assert estimate_vram(model, {"ctx_size": 4096}, cfg) is None


def test_estimate_vram_uses_resolved_settings(tmp_path, monkeypatch):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    gguf = cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_x-UD-Q5_K_XL.gguf"
    gguf.write_bytes(b"GGUF")

    fake_params = {"kv_layers": 4, "kv_heads": 2, "head_dim": 64, "weight_gb": 3.0}
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: fake_params,
    )
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    settings = {"ctx_size": 2048, "cache_type_k": "q8_0",
                "cache_type_v": "q8_0", "batch_size": 1024}
    est = estimate_vram(model, settings, cfg)
    assert est is not None
    assert est.model_gb == 3.0
    assert abs(est.compute_gb - 0.9 * (1024 / 512)) < 1e-9


def test_estimate_vram_returns_none_on_unreadable_gguf(tmp_path, monkeypatch):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_x-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: (_ for _ in ()).throw(ValueError("bad header")),
    )
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    assert estimate_vram(model, {"ctx_size": 2048}, cfg) is None


class _DummyCfg:
    """Stand-in GlobalConfig: estimate_vram only forwards it to resolve_gguf_path,
    which is monkeypatched in these tests, so no real fields are read."""


def test_estimate_vram_none_when_kv_heads_missing(monkeypatch):
    """A GGUF whose header lacks KV-head metadata must yield None, not a
    falsely-low estimate that omits the KV-cache term."""
    from llamactl.core import estimate as est_mod

    monkeypatch.setattr(
        est_mod, "resolve_gguf_path", lambda hf, cfg: __import__("pathlib").Path("/x/m.gguf")
    )
    monkeypatch.setattr(
        est_mod, "model_params_from_gguf",
        lambda path: {"arch": "llm", "block_count": 32, "kv_layers": 32,
                      "kv_heads": 0, "head_dim": 128, "weight_gb": 7.0},
    )
    model = ModelConfig(id="m", name="M", hf="org/repo:Q5_K_M", settings={},
                        backends={}, presets={}, images={}, path=Path("m.toml"))
    result = est_mod.estimate_vram(model, {"ctx_size": 4096}, _DummyCfg())
    assert result is None


def test_estimate_vram_none_when_head_dim_missing(monkeypatch):
    """A GGUF whose header lacks head_dim metadata must yield None, not a
    falsely-low estimate that omits the KV-cache term."""
    from llamactl.core import estimate as est_mod

    monkeypatch.setattr(
        est_mod, "resolve_gguf_path", lambda hf, cfg: __import__("pathlib").Path("/x/m.gguf")
    )
    monkeypatch.setattr(
        est_mod, "model_params_from_gguf",
        lambda path: {"arch": "llm", "block_count": 32, "kv_layers": 32,
                      "kv_heads": 8, "head_dim": 0, "weight_gb": 7.0},
    )
    model = ModelConfig(id="m", name="M", hf="org/repo:Q5_K_M", settings={},
                        backends={}, presets={}, images={}, path=Path("m.toml"))
    assert est_mod.estimate_vram(model, {"ctx_size": 4096}, _DummyCfg()) is None
