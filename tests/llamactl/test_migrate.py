from __future__ import annotations

import json
from pathlib import Path

import tomllib
import tomlkit

from llamactl.core.migrate import convert_model, migrate

SAMPLE_JSON = {
    "name": "Sample Model",
    "hf": "org/sample-GGUF:Q5",
    "defaults": {
        "ctx_size": 262144,
        "flash_attn": True,        # legacy bool -> "on"
        "no_warmup": False,        # false bool -> dropped
        "cram": 2048,
        "prefill_assistant": False,  # -> no_prefill_assistant = true
        "model_draft": None,       # null -> dropped
        # note: min_p, repeat_penalty, jinja, top_k... deliberately absent —
        # run.py emitted hardcoded fallbacks for these, so migration must
        # materialize them (otherwise llama-server's own defaults kick in,
        # e.g. min_p 0.05 instead of run.py's forced 0.0).
    },
    "backends": {
        "rocm": {"cache_k": "q8_0", "cache_v": "q8_0", "no_mmap": True},
        "vulkan": {"cache_k": "q8_0"},
    },
    "presets": {
        "fast": {"temp": 0.3, "cache_k": "f16"},
    },
    "images": {"rocm": "llama-cpp-gfx1031:b8586"},
}


def convert_to_plain(raw: dict, model_id: str) -> tuple[dict, list[str]]:
    """Round-trip through TOML text so assertions see plain Python types."""
    doc, warnings = convert_model(raw, model_id)
    return tomllib.loads(tomlkit.dumps(doc)), warnings


def test_convert_settings_rules() -> None:
    data, warnings = convert_to_plain(SAMPLE_JSON, "sample")
    settings = data["settings"]
    assert settings["ctx_size"] == 262144
    assert settings["flash_attn"] == "on"
    assert "no_warmup" not in settings
    assert "model_draft" not in settings
    assert settings["no_prefill_assistant"] is True
    assert "prefill_assistant" not in settings
    assert warnings == []


def test_convert_materializes_run_py_implicit_defaults() -> None:
    data, _ = convert_to_plain(SAMPLE_JSON, "sample")
    settings = data["settings"]
    # absent from SAMPLE_JSON defaults; run.py always emitted these flags
    assert settings["min_p"] == 0.0
    assert settings["repeat_penalty"] == 1.0
    assert settings["jinja"] is True
    assert settings["top_k"] == 20
    # explicit values are never clobbered by the injection
    assert settings["ctx_size"] == 262144
    # injection applies to [settings] only, not backends/presets
    assert "min_p" not in data["backends"]["rocm"]
    assert "min_p" not in data["presets"]["fast"]


def test_convert_renames_legacy_cache_keys_everywhere() -> None:
    data, _ = convert_to_plain(SAMPLE_JSON, "sample")
    assert data["backends"]["rocm"]["cache_type_k"] == "q8_0"
    assert data["backends"]["rocm"]["cache_type_v"] == "q8_0"
    assert data["presets"]["fast"]["cache_type_k"] == "f16"
    assert "cache_k" not in data["backends"]["rocm"]


def test_convert_carries_name_hf_images() -> None:
    data, _ = convert_to_plain(SAMPLE_JSON, "sample")
    assert data["name"] == "Sample Model"
    assert data["hf"] == "org/sample-GGUF:Q5"
    assert data["images"]["rocm"] == "llama-cpp-gfx1031:b8586"


def test_convert_warns_on_unknown_top_level_key() -> None:
    raw = dict(SAMPLE_JSON)
    raw["mystery_section"] = {"a": 1}
    _, warnings = convert_model(raw, "sample")
    assert any("mystery_section" in w for w in warnings)


def test_migrate_writes_skips_and_forces(tmp_path: Path) -> None:
    src = tmp_path / "models"
    dest = tmp_path / "configs" / "models"
    src.mkdir()
    (src / "sample.json").write_text(json.dumps(SAMPLE_JSON), encoding="utf-8")

    first = migrate(src, dest)
    assert [(p.name, status) for p, status, _ in first] == [("sample.toml", "written")]
    with (dest / "sample.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["hf"] == "org/sample-GGUF:Q5"

    second = migrate(src, dest)
    assert [(p.name, status) for p, status, _ in second] == [("sample.toml", "skipped")]

    third = migrate(src, dest, force=True)
    assert [(p.name, status) for p, status, _ in third] == [("sample.toml", "written")]


def test_migrate_reports_invalid_json_as_failed(tmp_path: Path) -> None:
    src = tmp_path / "models"
    dest = tmp_path / "out"
    src.mkdir()
    (src / "broken.json").write_text("{not json", encoding="utf-8")
    results = migrate(src, dest)
    assert [(p.name, status) for p, status, _ in results] == [("broken.toml", "failed")]
    assert not (dest / "broken.toml").exists()


def test_migrate_isolates_malformed_section_and_continues(tmp_path: Path) -> None:
    src = tmp_path / "models"
    dest = tmp_path / "out"
    src.mkdir()
    (src / "a-bad.json").write_text(
        json.dumps({"hf": "org/x:Q5", "defaults": []}), encoding="utf-8"
    )
    (src / "b-good.json").write_text(json.dumps(SAMPLE_JSON), encoding="utf-8")
    results = migrate(src, dest)
    assert [(p.name, status) for p, status, _ in results] == [
        ("a-bad.toml", "failed"),
        ("b-good.toml", "written"),
    ]
    assert not (dest / "a-bad.toml").exists()
    assert (dest / "b-good.toml").exists()
