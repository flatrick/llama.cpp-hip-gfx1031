from __future__ import annotations

from pathlib import Path

import pytest

import tomlkit

from llamactl.core.config_edit import delete_key, duplicate_doc, ensure_section, is_valid_key, load_doc, model_id_from_name, new_model_doc, parse_value, save_doc, set_value, value_to_literal


def test_parse_value_types():
    assert parse_value("true") is True
    assert parse_value("false") is False
    assert parse_value("16384") == 16384 and isinstance(parse_value("16384"), int)
    assert parse_value("0.7") == 0.7 and isinstance(parse_value("0.7"), float)
    assert parse_value('["a", "b"]') == ["a", "b"]
    assert parse_value('"on"') == "on" and isinstance(parse_value('"on"'), str)


def test_parse_value_bare_word_is_string():
    assert parse_value("on") == "on"
    assert parse_value("q8_0") == "q8_0"
    assert parse_value("auto") == "auto"


def test_value_to_literal_round_trips_through_parse_value():
    for raw_value in (True, False, 16384, 0.7, "on", ["a", "b"]):
        literal = value_to_literal(raw_value)
        assert parse_value(literal) == raw_value
    assert isinstance(parse_value(value_to_literal(True)), bool)
    assert value_to_literal("on") == '"on"'


def test_is_valid_key():
    assert is_valid_key("ctx_size")
    assert is_valid_key("cache-type-k")
    assert not is_valid_key("bad key")
    assert not is_valid_key("has.dot")
    assert not is_valid_key("")


def test_load_and_save_round_trip_preserves_comments(tmp_path):
    src = tmp_path / "m.toml"
    src.write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\n'
        '# tuning comment\nctx_size = 4096\n',
        encoding="utf-8",
    )
    doc = load_doc(src)
    out = tmp_path / "out.toml"
    save_doc(out, doc)
    text = out.read_text(encoding="utf-8")
    assert "# tuning comment" in text
    assert 'ctx_size = 4096' in text


def test_save_doc_is_atomic_leaves_no_tmp(tmp_path):
    doc = load_doc_from_text('hf = "x"\n', tmp_path)
    target = tmp_path / "m.toml"
    save_doc(target, doc)
    assert target.exists()
    assert not (tmp_path / "m.toml.tmp").exists()


def load_doc_from_text(text: str, tmp_path) -> object:
    p = tmp_path / "seed.toml"
    p.write_text(text, encoding="utf-8")
    return load_doc(p)


def _doc(text: str):
    return tomlkit.parse(text)


def test_set_value_updates_existing_and_preserves_comment():
    doc = _doc('name = "M"\nhf = "x"\n\n[settings]\n# keep me\nctx_size = 4096\n')
    set_value(doc, "settings", "ctx_size", "8192")
    text = tomlkit.dumps(doc)
    assert "# keep me" in text
    assert doc["settings"]["ctx_size"] == 8192
    assert isinstance(doc["settings"]["ctx_size"], int)


def test_set_value_creates_nested_section():
    doc = _doc('hf = "x"\n')
    set_value(doc, "backends.rocm", "cache_type_k", "q8_0")
    assert doc["backends"]["rocm"]["cache_type_k"] == "q8_0"


def test_set_value_top_level():
    doc = _doc('hf = "x"\n')
    set_value(doc, "", "name", '"New Name"')
    assert doc["name"] == "New Name"


def test_set_value_bool_stays_bare_flag():
    doc = _doc('hf = "x"\n[settings]\n')
    set_value(doc, "settings", "jinja", "true")
    assert doc["settings"]["jinja"] is True


def test_delete_key_removes_only_that_key():
    doc = _doc('hf = "x"\n[settings]\na = 1\nb = 2\n')
    delete_key(doc, "settings", "a")
    assert "a" not in doc["settings"]
    assert doc["settings"]["b"] == 2


def test_ensure_section_creates_empty_preset_table():
    doc = _doc('hf = "x"\n')
    ensure_section(doc, "presets.thinking")
    assert "thinking" in doc["presets"]


def test_section_table_returns_none_on_scalar_collision():
    from llamactl.core.config_edit import _section_table
    # Integer scalars raise TypeError on `in` — the guard must catch this.
    doc = _doc('backends = 42\n')   # "backends" holds an integer, not a table
    # Descending into a scalar must not crash; create=False yields None.
    assert _section_table(doc, "backends.rocm", create=False) is None


def test_delete_key_no_crash_on_scalar_collision():
    doc = _doc('backends = 42\n')
    delete_key(doc, "backends.rocm", "x")   # must be a silent no-op, not a TypeError
    assert doc["backends"] == 42


def test_new_model_doc_minimal_shape():
    doc = new_model_doc("My Model", "org/m:f")
    assert doc["name"] == "My Model"
    assert doc["hf"] == "org/m:f"
    assert "settings" in doc and len(doc["settings"]) == 0


def test_duplicate_doc_is_independent_and_keeps_comments():
    src = tomlkit.parse('name = "M"\nhf = "x"\n[settings]\n# c\na = 1\n')
    dup = duplicate_doc(src)
    dup["settings"]["a"] = 999
    assert src["settings"]["a"] == 1
    assert "# c" in tomlkit.dumps(dup)


def test_model_id_from_name_slug():
    assert model_id_from_name("Qwen3 8B Instruct") == "Qwen3-8B-Instruct"
    assert model_id_from_name("a/b:c") == "a-b-c"
    assert model_id_from_name("  ") == "model"


def test_section_table_terminal_scalar_returns_none():
    doc = _doc('backends = 42\n')   # single-segment section occupied by a scalar
    from llamactl.core.config_edit import _section_table
    assert _section_table(doc, "backends", create=False) is None


def test_delete_key_returns_bool():
    doc = _doc('hf = "x"\n[settings]\na = 1\n')
    assert delete_key(doc, "settings", "a") is True
    assert delete_key(doc, "settings", "a") is False        # already gone
    assert delete_key(doc, "backends", "missing") is False  # absent section
