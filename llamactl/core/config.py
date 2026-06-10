"""TOML config store: per-model configs and the global llamactl config."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """A config file is missing, unparseable, or structurally invalid."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    hf: str
    settings: dict[str, Any]
    backends: dict[str, dict[str, Any]]
    presets: dict[str, dict[str, Any]]
    images: dict[str, str]
    path: Path


def load_model(path: Path) -> ModelConfig:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if "hf" not in data:
        raise ConfigError(f"{path}: missing required key 'hf'")
    return ModelConfig(
        id=path.stem,
        name=data.get("name", path.stem),
        hf=data["hf"],
        settings=dict(data.get("settings", {})),
        backends={k: dict(v) for k, v in data.get("backends", {}).items()},
        presets={k: dict(v) for k, v in data.get("presets", {}).items()},
        images=dict(data.get("images", {})),
        path=path,
    )


def load_all(models_dir: Path) -> tuple[list[ModelConfig], dict[Path, str]]:
    """Load every *.toml in models_dir; bad files become error entries."""
    configs: list[ModelConfig] = []
    errors: dict[Path, str] = {}
    for path in sorted(models_dir.glob("*.toml")):
        try:
            configs.append(load_model(path))
        except ConfigError as exc:
            errors[path] = str(exc)
    return configs, errors
