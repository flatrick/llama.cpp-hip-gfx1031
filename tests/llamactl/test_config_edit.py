from __future__ import annotations

from pathlib import Path

import pytest

from llamactl.core.config_edit import is_valid_key, parse_value, value_to_literal


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
