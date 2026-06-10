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


def _require_table(path: Path, key: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(
            f"{path}: '{key}' must be a table, got {type(value).__name__}"
        )
    return value


def _require_str(path: Path, key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            f"{path}: '{key}' must be a string, got {type(value).__name__}"
        )
    return value


def load_model(path: Path) -> ModelConfig:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if "hf" not in data:
        raise ConfigError(f"{path}: missing required key 'hf'")
    hf = _require_str(path, "hf", data["hf"])
    name = _require_str(path, "name", data.get("name", path.stem))
    settings = dict(_require_table(path, "settings", data.get("settings", {})))
    backends = {
        k: dict(_require_table(path, f"backends.{k}", v))
        for k, v in _require_table(
            path, "backends", data.get("backends", {})
        ).items()
    }
    presets = {
        k: dict(_require_table(path, f"presets.{k}", v))
        for k, v in _require_table(
            path, "presets", data.get("presets", {})
        ).items()
    }
    images = dict(_require_table(path, "images", data.get("images", {})))
    return ModelConfig(
        id=path.stem,
        name=name,
        hf=hf,
        settings=settings,
        backends=backends,
        presets=presets,
        images=images,
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


def resolve_settings(
    model: ModelConfig,
    preset: str | None,
    backend: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge layers: settings -> preset -> backend -> overrides (later wins)."""
    settings = dict(model.settings)
    if preset is not None:
        if preset not in model.presets:
            available = ", ".join(model.presets) or "(none)"
            raise ConfigError(
                f"preset '{preset}' not found for model '{model.id}'; "
                f"available: {available}"
            )
        settings.update(model.presets[preset])
    settings.update(model.backends.get(backend, {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings
