from __future__ import annotations

from pathlib import Path

import pytest

from llamactl.core.config import ConfigError, ModelConfig, load_all, load_model

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
