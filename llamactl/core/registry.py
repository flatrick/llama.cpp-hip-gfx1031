"""Artifact registry: records of successful llama.cpp builds.

Stored as [[artifact]] tables in state/registry.toml. All list operations
return new lists (no mutation).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomlkit


@dataclass(frozen=True)
class Artifact:
    target: str            # rocm-image | vulkan-image | rocm-native | vulkan-native
    requested_ref: str     # submodule | latest-tag | tag:<n> | branch:<n> | commit:<sha>
    sha: str               # resolved commit sha
    build_number: str      # llama.cpp build tag, e.g. "b8586" ("" if unknown)
    built_at: str          # ISO-8601 timestamp
    image_tag: str | None = None    # set for *-image targets
    binary_path: str | None = None  # set for *-native targets


def load_registry(path: Path) -> list[Artifact]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    return [Artifact(**entry) for entry in data.get("artifact", [])]


def save_registry(path: Path, artifacts: list[Artifact]) -> None:
    doc = tomlkit.document()
    aot = tomlkit.aot()
    for a in artifacts:
        table = tomlkit.table()
        table["target"] = a.target
        table["requested_ref"] = a.requested_ref
        table["sha"] = a.sha
        table["build_number"] = a.build_number
        table["built_at"] = a.built_at
        if a.image_tag is not None:
            table["image_tag"] = a.image_tag
        if a.binary_path is not None:
            table["binary_path"] = a.binary_path
        aot.append(table)
    doc["artifact"] = aot
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def add_artifact(artifacts: list[Artifact], new: Artifact) -> list[Artifact]:
    """Append; an existing entry with the same (target, sha) is replaced."""
    kept = [a for a in artifacts if (a.target, a.sha) != (new.target, new.sha)]
    return [*kept, new]


def remove_artifact(artifacts: list[Artifact], target: str, sha: str) -> list[Artifact]:
    return [a for a in artifacts if (a.target, a.sha) != (target, sha)]
