"""Comment-preserving edit/save layer for per-model TOML configs.

Operates on a live tomlkit.TOMLDocument so comments and formatting survive
edits. core never imports UI. Value typing follows the TOML-literal rule:
parse `key = <input>`; if it does not parse, the input is a plain string.
"""

from __future__ import annotations

import re
from pathlib import Path
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


def load_doc(path: Path) -> TOMLDocument:
    return tomlkit.parse(path.read_text(encoding="utf-8"))


def save_doc(path: Path, doc: TOMLDocument) -> None:
    """Atomic write (temp file + replace), mirroring registry.save_registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
    tmp.replace(path)


def _section_table(doc: TOMLDocument, section: str, create: bool) -> Any:
    """Return the table for `section` ("" = doc root). create=True makes any
    missing intermediate tables; create=False returns None if absent. If a path
    segment is occupied by a non-table value, returns None (create=False) or
    raises ValueError (create=True) instead of crashing."""
    if section == "":
        return doc
    node: Any = doc
    for part in section.split("."):
        if not isinstance(node, dict):  # tomlkit tables/documents are dicts; scalars are not
            if not create:
                return None
            raise ValueError(f"cannot descend into non-table section: {section!r}")
        if part not in node:
            if not create:
                return None
            node[part] = tomlkit.table()
        node = node[part]
    return node


def ensure_section(doc: TOMLDocument, section: str) -> None:
    """Create an empty table for `section` (e.g. 'presets.thinking') if absent."""
    _section_table(doc, section, create=True)


def set_value(doc: TOMLDocument, section: str, key: str, raw: str) -> None:
    """Set section.key = parse_value(raw), creating the section if needed."""
    table = _section_table(doc, section, create=True)
    table[key] = parse_value(raw)


def delete_key(doc: TOMLDocument, section: str, key: str) -> None:
    """Remove section.key if both exist; a no-op otherwise."""
    table = _section_table(doc, section, create=False)
    if table is not None and key in table:
        del table[key]


def new_model_doc(name: str, hf: str) -> TOMLDocument:
    doc = tomlkit.document()
    doc["name"] = name
    doc["hf"] = hf
    doc["settings"] = tomlkit.table()
    return doc


def duplicate_doc(src: TOMLDocument) -> TOMLDocument:
    """Independent copy with comments intact (re-parse a dump of src)."""
    return tomlkit.parse(tomlkit.dumps(src))


def model_id_from_name(name: str) -> str:
    """Slugify a display name into a filename stem (A-Za-z0-9._- ; '-' joins)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return slug or "model"
