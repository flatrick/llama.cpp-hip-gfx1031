from __future__ import annotations

from pathlib import Path

import pytest

from llamactl.core.config import (
    ConfigError,
    GlobalConfig,
    ModelConfig,
    _build_model_config,
    load_all,
    load_global,
    load_model,
    resolve_settings,
)

VALID_TOML = """\
name = "Test Model"
hf = "org/test-GGUF:Q5_K_M"

[settings]
ctx_size = 4096
flash_attn = "on"

[backends.rocm]
cache_type_k = "q8_0"

[presets.fast]
temp = 0.3

[images]
rocm = "llama-cpp-gfx1031:b8586"
"""


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_model_parses_all_sections(tmp_path: Path) -> None:
    path = write(tmp_path, "test-model.toml", VALID_TOML)
    model = load_model(path)
    assert model.id == "test-model"
    assert model.name == "Test Model"
    assert model.hf == "org/test-GGUF:Q5_K_M"
    assert model.settings == {"ctx_size": 4096, "flash_attn": "on"}
    assert model.backends == {"rocm": {"cache_type_k": "q8_0"}}
    assert model.presets == {"fast": {"temp": 0.3}}
    assert model.images == {"rocm": "llama-cpp-gfx1031:b8586"}
    assert model.path == path


def test_load_model_missing_hf_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "broken.toml", 'name = "No HF"\n[settings]\nctx_size = 1\n')
    with pytest.raises(ConfigError, match="hf"):
        load_model(path)


def test_load_model_bad_toml_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "syntax.toml", "name = [unclosed\n")
    with pytest.raises(ConfigError):
        load_model(path)


def test_load_all_isolates_bad_files(tmp_path: Path) -> None:
    write(tmp_path, "good.toml", VALID_TOML)
    bad = write(tmp_path, "bad.toml", "definitely not toml ===\n")
    configs, errors = load_all(tmp_path)
    assert [m.id for m in configs] == ["good"]
    assert list(errors) == [bad]


def test_load_all_empty_dir(tmp_path: Path) -> None:
    configs, errors = load_all(tmp_path)
    assert configs == []
    assert errors == {}


def test_load_model_non_table_section_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "bad-shape.toml", 'hf = "org/x:Q5"\nbackends = 3\n')
    with pytest.raises(ConfigError, match="backends"):
        load_model(path)


def test_load_model_non_table_backend_entry_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "bad-entry.toml", 'hf = "org/x:Q5"\n[backends]\nrocm = 3\n')
    with pytest.raises(ConfigError, match="backends.rocm"):
        load_model(path)


def test_load_model_non_string_hf_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "bad-hf.toml", "hf = 42\n")
    with pytest.raises(ConfigError, match="hf"):
        load_model(path)


def test_load_all_isolates_wrong_shaped_section(tmp_path: Path) -> None:
    write(tmp_path, "good.toml", VALID_TOML)
    bad = write(tmp_path, "shape.toml", 'hf = "org/x:Q5"\nimages = [1, 2]\n')
    configs, errors = load_all(tmp_path)
    assert [m.id for m in configs] == ["good"]
    assert list(errors) == [bad]


def make_model(tmp_path: Path) -> ModelConfig:
    content = """\
name = "Layered"
hf = "org/layered:Q5"

[settings]
ctx_size = 262144
temp = 0.7
cache_type_k = "f16"

[backends.rocm]
cache_type_k = "q8_0"
no_mmap = true

[presets.cool]
temp = 0.3
cache_type_k = "q4_0"
"""
    return load_model(write(tmp_path, "layered.toml", content))


def test_resolve_defaults_only(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, None, "vulkan", {})
    assert settings == {"ctx_size": 262144, "temp": 0.7, "cache_type_k": "f16"}


def test_backend_overrides_preset(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, "cool", "rocm", {})
    # preset set temp and cache_type_k; backend wins on cache_type_k
    assert settings["temp"] == 0.3
    assert settings["cache_type_k"] == "q8_0"
    assert settings["no_mmap"] is True


def test_overrides_win_over_everything(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, "cool", "rocm", {"cache_type_k": "f16"})
    assert settings["cache_type_k"] == "f16"


def test_none_overrides_are_ignored(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, None, "rocm", {"temp": None})
    assert settings["temp"] == 0.7


def test_unknown_preset_raises_with_available_list(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    with pytest.raises(ConfigError, match="cool"):
        resolve_settings(model, "c00l", "rocm", {})


def test_resolution_does_not_mutate_model(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    before = dict(model.settings)
    resolve_settings(model, "cool", "rocm", {"extra": 1})
    assert model.settings == before


def test_load_global_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_global(tmp_path / "nope.toml")
    assert cfg == GlobalConfig()
    assert cfg.default_backend == "rocm"
    assert cfg.port == 8080
    assert cfg.vram_budget_gb == 11.0
    assert cfg.name_prefix == "llamactl"


def test_load_global_partial_file_fills_defaults(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", 'default_backend = "vulkan"\nport = 8081\n')
    cfg = load_global(path)
    assert cfg.default_backend == "vulkan"
    assert cfg.port == 8081
    assert cfg.default_mode == "container"  # untouched default


def test_load_global_expands_user_paths(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", 'hf_cache = "~/somewhere"\n')
    cfg = load_global(path)
    assert "~" not in str(cfg.hf_cache)


def test_load_global_bad_toml_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", "port = [broken\n")
    with pytest.raises(ConfigError):
        load_global(path)


def test_load_global_wrong_typed_port_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", 'port = "8080"\n')
    with pytest.raises(ConfigError, match="port"):
        load_global(path)


def test_load_global_non_numeric_vram_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", 'vram_budget_gb = "x"\n')
    with pytest.raises(ConfigError, match="vram_budget_gb"):
        load_global(path)


def test_load_global_int_vram_accepted_as_float(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", "vram_budget_gb = 11\n")
    cfg = load_global(path)
    assert cfg.vram_budget_gb == 11.0
    assert isinstance(cfg.vram_budget_gb, float)


def test_checked_in_global_config_equals_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_global(repo_root / "configs" / "llamactl.toml")
    assert cfg == GlobalConfig()


def test_build_model_config_from_plain_dict(tmp_path):
    data = {"hf": "org/m:f", "name": "M", "settings": {"ctx_size": 4096}}
    mc = _build_model_config(data, tmp_path / "m.toml")
    assert isinstance(mc, ModelConfig)
    assert mc.id == "m"
    assert mc.hf == "org/m:f"
    assert mc.settings == {"ctx_size": 4096}


def test_resolve_settings_returns_deep_copy():
    mc = ModelConfig(
        id="m", name="M", hf="org/m:f",
        settings={"stop": ["</s>"], "ctx_size": 4096},
        backends={}, presets={}, images={}, path=__import__("pathlib").Path("/x/m.toml"),
    )
    resolved = resolve_settings(mc, None, "rocm", {})
    resolved["stop"].append("INJECTED")
    resolved["ctx_size"] = 999
    assert mc.settings["stop"] == ["</s>"]
    assert mc.settings["ctx_size"] == 4096
