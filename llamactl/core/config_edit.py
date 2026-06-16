"""Comment-preserving edit/save layer for per-model TOML configs.

Operates on a live tomlkit.TOMLDocument so comments and formatting survive
edits. core never imports UI. Value typing follows the TOML-literal rule:
parse `key = <input>`; if it does not parse, the input is a plain string.
"""

from __future__ import annotations

import re
from typing import Any

import tomlkit
from tomlkit import TOMLDocument

_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_value(raw: str) -> Any:
    """TOML-literal typing with string fallback. Never raises.

    `true`/`false`->bool, `16384`->int, `0.7`->float, `["a","b"]`->list,
    `"on"`->str. A bare word that is not valid TOML (e.g. `q8_0`, `on`, `auto`)
    becomes a plain string.
    """
    try:
        return tomlkit.parse(f"_x_ = {raw}").unwrap()["_x_"]
    except Exception:
        return raw


def value_to_literal(value: Any) -> str:
    """Render a Python value as the TOML value literal that parse_value inverts.

    bool -> true/false, str -> quoted, int/float -> bare, list -> inline array.
    """
    doc = tomlkit.document()
    doc["_x_"] = value
    return tomlkit.dumps(doc).split("=", 1)[1].strip()


def is_valid_key(key: str) -> bool:
    """True if `key` is a valid single TOML bare key (A-Za-z0-9_-, non-empty)."""
    return bool(_BARE_KEY_RE.match(key))
