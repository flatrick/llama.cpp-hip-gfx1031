"""One-shot migration of legacy models/*.json into configs/models/*.toml.

Conversion rules (see design doc):
  cache_k / cache_v          -> cache_type_k / cache_type_v (actual flag names)
  flash_attn = true          -> flash_attn = "on" (flag takes a value)
  prefill_assistant = false  -> no_prefill_assistant = true (bool convention)
  null values / false bools  -> dropped (absence is the only "off")

Additionally, run.py emitted hardcoded fallback flags for keys missing from
the JSON (e.g. --min-p 0.0 even when min_p was absent). To preserve observed
launch behavior under the new "absent = omitted" semantics, migration
materializes those implicit defaults into [settings] (setdefault — explicit
values always win). This happens once, at migration time only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tomlkit

LEGACY_KEY_RENAMES: dict[str, str] = {
    "cache_k": "cache_type_k",
    "cache_v": "cache_type_v",
}

KNOWN_TOP_LEVEL: frozenset[str] = frozenset(
    {"name", "hf", "defaults", "backends", "presets", "images"}
)

# Fallback values run.py's build_server_args() hardcoded for absent keys.
# Keys run.py only emitted when present (reasoning, cram, n_cpu_moe,
# model_draft, no_mmap, no_warmup, no_mmproj, prefill_assistant) are NOT here.
RUN_PY_IMPLICIT_DEFAULTS: dict[str, Any] = {
    "n_gpu_layers": -1,
    "batch_size": 1024,
    "ubatch_size": 256,
    "parallel": 1,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "top_k": 20,
    "top_p": 0.8,
    "temp": 0.7,
    "presence_penalty": 1.5,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "flash_attn": "on",
    "jinja": True,
}


def _convert_settings(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        key = LEGACY_KEY_RENAMES.get(key, key)
        if value is None:
            continue
        if key == "flash_attn" and isinstance(value, bool):
            if value:
                out["flash_attn"] = "on"
            continue
        if key == "prefill_assistant" and isinstance(value, bool):
            if value:
                out["prefill_assistant"] = True
            else:
                out["no_prefill_assistant"] = True
            continue
        if isinstance(value, bool) and not value:
            continue
        out[key] = value
    return out


def convert_model(
    raw: dict[str, Any], model_id: str
) -> tuple[tomlkit.TOMLDocument, list[str]]:
    warnings = [
        f"{model_id}: unknown top-level key '{k}' not migrated"
        for k in raw
        if k not in KNOWN_TOP_LEVEL
    ]
    doc = tomlkit.document()
    doc["name"] = raw.get("name", model_id)
    doc["hf"] = raw["hf"]
    settings = _convert_settings(raw.get("defaults", {}))
    for key, value in RUN_PY_IMPLICIT_DEFAULTS.items():
        settings.setdefault(key, value)
    doc["settings"] = settings
    if raw.get("backends"):
        backends = tomlkit.table()
        for backend, vals in raw["backends"].items():
            backends[backend] = _convert_settings(vals)
        doc["backends"] = backends
    if raw.get("presets"):
        presets = tomlkit.table()
        for preset, vals in raw["presets"].items():
            presets[preset] = _convert_settings(vals)
        doc["presets"] = presets
    if raw.get("images"):
        doc["images"] = dict(raw["images"])
    return doc, warnings


def migrate(
    src_dir: Path, dest_dir: Path, force: bool = False
) -> list[tuple[Path, str, list[str]]]:
    """Convert every src_dir/*.json. Returns (dest_path, status, warnings)
    per file, where status is one of: written, skipped, failed."""
    results: list[tuple[Path, str, list[str]]] = []
    for json_path in sorted(src_dir.glob("*.json")):
        dest = dest_dir / f"{json_path.stem}.toml"
        if dest.exists() and not force:
            results.append((dest, "skipped", []))
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            doc, warnings = convert_model(raw, json_path.stem)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            results.append((dest, "failed", [f"{json_path}: {exc}"]))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(tomlkit.dumps(doc), encoding="utf-8")
        results.append((dest, "written", warnings))
    return results
