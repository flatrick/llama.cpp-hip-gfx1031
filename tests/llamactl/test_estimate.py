from __future__ import annotations

from pathlib import Path

import pytest

from llamactl.core.config import GlobalConfig, ModelConfig
from llamactl.core.estimate import (
    Estimate,
    SweepCell,
    _resolve_kv_layers,
    classify_headroom,
    compute_estimate,
    estimate_sweep,
    estimate_vram,
    resolve_gguf_path,
    sweep_vram,
    SWEEP_CACHE_TYPES,
    SWEEP_CTX_SIZES,
)


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


@pytest.mark.parametrize("spec", [
    "../../etc/passwd:Q5_K_M",       # traversal in repo part
    "org/repo:../../../q",           # traversal in quant
    "org/re*po:Q5_K_M",              # glob metachar in repo
])
def test_resolve_gguf_path_rejects_unsafe_spec(spec, tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True, exist_ok=True)
    cfg.hf_cache.mkdir(parents=True, exist_ok=True)
    assert resolve_gguf_path(spec, cfg) is None


def test_resolve_gguf_path_glob_metachar_does_not_match_planted_file(tmp_path):
    # Non-vacuous: without validation, `*{repo}*{quant}*.gguf` with repo="re*po"
    # expands and matches this planted file; the guard must reject it → None.
    # (This test fails if the spec validation is removed.)
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True, exist_ok=True)
    (cfg.llama_cache / "xx_reXXpo_Q5_K_M.gguf").write_bytes(b"GGUF")
    assert resolve_gguf_path("org/re*po:Q5_K_M", cfg) is None


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


def test_estimate_vram_compute_buffer_uses_ubatch_not_batch(tmp_path, monkeypatch):
    """The graph compute buffer scales with the physical micro-batch (ubatch_size),
    not the logical batch_size. With ubatch 256 the buffer must be 0.9*(256/512),
    NOT 0.9*(1024/512) — the latter overcounted VRAM by ~1.35 GB (qwen3.5-4b:
    estimated 13.4 vs measured 10.94)."""
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_x-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: {"kv_layers": 4, "kv_heads": 2, "head_dim": 64, "weight_gb": 3.0},
    )
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    settings = {"ctx_size": 2048, "batch_size": 1024, "ubatch_size": 256}
    est = estimate_vram(model, settings, cfg)
    assert est is not None
    assert abs(est.compute_gb - 0.9 * (256 / 512)) < 1e-9


def test_estimate_vram_compute_buffer_falls_back_to_batch_without_ubatch(tmp_path, monkeypatch):
    """When ubatch_size is absent, the compute buffer falls back to batch_size."""
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_x-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: {"kv_layers": 4, "kv_heads": 2, "head_dim": 64, "weight_gb": 3.0},
    )
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    est = estimate_vram(model, {"ctx_size": 2048, "batch_size": 768}, cfg)
    assert est is not None
    assert abs(est.compute_gb - 0.9 * (768 / 512)) < 1e-9


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
    monkeypatch.setattr(
        "llamactl.core.estimate.resolve_gguf_path",
        lambda hf, cfg: Path("/x/m.gguf"),
    )
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: {"arch": "llm", "block_count": 32, "kv_layers": 32,
                      "kv_heads": 0, "head_dim": 128, "weight_gb": 7.0},
    )
    model = _model("org/repo:Q5_K_M")
    assert estimate_vram(model, {"ctx_size": 4096}, _DummyCfg()) is None


def test_estimate_vram_none_when_kv_metadata_is_array(monkeypatch):
    """Some GGUFs store head_count_kv / key_length as per-layer ARRAYS. We don't
    model that, so the estimate must be 'unavailable' (None) rather than crash
    trying to multiply a list — the value must never be silently wrong."""
    monkeypatch.setattr(
        "llamactl.core.estimate.resolve_gguf_path", lambda hf, cfg: Path("/x/m.gguf"),
    )
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: {"arch": "llm", "block_count": 32, "kv_layers": 32,
                      "kv_heads": [8, 8, 8], "head_dim": [128, 128], "weight_gb": 7.0},
    )
    assert estimate_vram(_model("org/repo:Q5_K_M"), {"ctx_size": 4096}, _DummyCfg()) is None
    assert sweep_vram(_model("org/repo:Q5_K_M"), {}, _DummyCfg()) is None


def test_estimate_vram_none_when_head_dim_missing(monkeypatch):
    """A GGUF whose header lacks head_dim metadata must yield None, not a
    falsely-low estimate that omits the KV-cache term."""
    monkeypatch.setattr(
        "llamactl.core.estimate.resolve_gguf_path",
        lambda hf, cfg: Path("/x/m.gguf"),
    )
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: {"arch": "llm", "block_count": 32, "kv_layers": 32,
                      "kv_heads": 8, "head_dim": 0, "weight_gb": 7.0},
    )
    model = _model("org/repo:Q5_K_M")
    assert estimate_vram(model, {"ctx_size": 4096}, _DummyCfg()) is None


# ── resolve_gguf_path: don't guess across models sharing a quant ──────────────

def test_resolve_ambiguous_quant_only_returns_none(tmp_path):
    # Two different models share the quant and neither matches the repo. The
    # repo-less fallback must NOT guess one — return None (estimate unavailable)
    # rather than estimate against the wrong (possibly much larger) model.
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "unsloth_Qwen3.5-4B-GGUF_Qwen3.5-4B-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    (cfg.llama_cache / "unsloth_Qwen3.6-35B-A3B-GGUF_Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    assert resolve_gguf_path("unsloth/Qwen3.5-0.8B-GGUF:UD-Q5_K_XL", cfg) is None


def test_resolve_prefers_hf_hub_repo_match_over_quant_only(tmp_path):
    # The real model is in the HF-hub cache (repo-specific); a DIFFERENT model
    # with the same quant sits in llama_cache. The repo-specific HF-hub match
    # must win over the repo-less fallback (the bug: it picked the wrong file).
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "unsloth_Qwen3.6-35B-A3B-GGUF_Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    snap = cfg.hf_cache / "models--unsloth--Qwen3.5-0.8B-GGUF" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    correct = snap / "Qwen3.5-0.8B-UD-Q5_K_XL.gguf"
    correct.write_bytes(b"GGUF")
    got = resolve_gguf_path("unsloth/Qwen3.5-0.8B-GGUF:UD-Q5_K_XL", cfg)
    assert got == correct


# ── _resolve_kv_layers: hybrid (SSM/attention) models ─────────────────────────

def test_kv_layers_uses_full_attention_interval():
    # Hybrid model: only every 4th of 24 layers is full-attention → 6 KV layers.
    meta = {"qwen35.full_attention_interval": 4}
    assert _resolve_kv_layers(meta, "qwen35", 24) == 6


def test_kv_layers_prefers_explicit_attention_layer_count():
    meta = {"qwen35.attention_layer_count": 8, "qwen35.full_attention_interval": 4}
    assert _resolve_kv_layers(meta, "qwen35", 24) == 8


def test_kv_layers_falls_back_to_block_count():
    assert _resolve_kv_layers({}, "llm", 32) == 32


# ── estimate_sweep: VRAM what-if grid (ports vram_calc.py's tables) ───────────

_SWEEP_PARAMS = {"kv_layers": 4, "kv_heads": 2, "head_dim": 64, "weight_gb": 3.0}


def test_classify_headroom_ok_tight_oom():
    # budget 11.0; thresholds: <0 → oom, <0.5 headroom → tight, else ok
    assert classify_headroom(5.0, 11.0) == "ok"        # 6.0 headroom
    assert classify_headroom(10.7, 11.0) == "tight"    # 0.3 headroom
    assert classify_headroom(11.0, 11.0) == "tight"    # 0.0 headroom (boundary)
    assert classify_headroom(12.0, 11.0) == "oom"      # negative headroom


def test_estimate_sweep_grid_shape_and_axes():
    cells = estimate_sweep(
        params=_SWEEP_PARAMS, base_settings={}, budget_gb=11.0,
    )
    # Full Cartesian grid of the default ctx × cache axes.
    assert len(cells) == len(SWEEP_CTX_SIZES) * len(SWEEP_CACHE_TYPES)
    assert all(isinstance(c, SweepCell) for c in cells)
    assert {c.ctx_size for c in cells} == set(SWEEP_CTX_SIZES)
    assert {c.cache_type for c in cells} == set(SWEEP_CACHE_TYPES)


def test_estimate_sweep_cell_matches_compute_estimate():
    cells = estimate_sweep(
        params=_SWEEP_PARAMS, base_settings={"batch_size": 1024}, budget_gb=11.0,
        ctx_sizes=(8192,), cache_types=("q8_0",),
    )
    assert len(cells) == 1
    cell = cells[0]
    direct = compute_estimate(
        params=_SWEEP_PARAMS, ctx_size=8192, cache_type_k="q8_0",
        cache_type_v="q8_0", batch_size=1024,
    )
    assert cell.estimate == direct
    assert cell.status == classify_headroom(direct.total_gb, 11.0)


def test_estimate_sweep_status_flips_with_budget():
    # A tiny budget makes every cell OOM; a huge budget makes every cell OK.
    oom = estimate_sweep(params=_SWEEP_PARAMS, base_settings={}, budget_gb=0.1)
    assert all(c.status == "oom" for c in oom)
    ok = estimate_sweep(params=_SWEEP_PARAMS, base_settings={}, budget_gb=10_000.0)
    assert all(c.status == "ok" for c in ok)


def test_sweep_vram_returns_none_when_gguf_absent(tmp_path):
    cfg = _global_cfg(tmp_path)
    cfg.llama_cache.mkdir(parents=True)
    model = _model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL")
    assert sweep_vram(model, {}, cfg) is None


def test_sweep_vram_none_when_kv_metadata_missing(monkeypatch):
    monkeypatch.setattr(
        "llamactl.core.estimate.resolve_gguf_path", lambda hf, cfg: Path("/x/m.gguf"),
    )
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: {"kv_layers": 4, "kv_heads": 0, "head_dim": 64, "weight_gb": 3.0},
    )
    assert sweep_vram(_model("org/repo:Q5_K_M"), {}, _DummyCfg()) is None


def test_sweep_vram_grid_uses_global_budget(monkeypatch, tmp_path):
    cfg = _global_cfg(tmp_path)  # vram_budget_gb defaults to 11.0
    cfg.llama_cache.mkdir(parents=True)
    (cfg.llama_cache / "unsloth_Qwen3.5-9B-GGUF_x-UD-Q5_K_XL.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(
        "llamactl.core.estimate.model_params_from_gguf",
        lambda path: _SWEEP_PARAMS,
    )
    cells = sweep_vram(_model("unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL"), {}, cfg)
    assert cells is not None
    assert len(cells) == len(SWEEP_CTX_SIZES) * len(SWEEP_CACHE_TYPES)
