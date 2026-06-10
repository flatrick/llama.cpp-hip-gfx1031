# llamactl Core Foundation (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the UI-agnostic core of `llamactl` — TOML config store, convention-based settings→argv mapper, one-shot JSON migration, and artifact-registry persistence — verified by dry-run parity with the frozen `run.py`.

**Architecture:** New `llamactl/core/` package (no UI in this phase) per the approved design at `docs/superpowers/specs/2026-06-10-llamactl-tui-design.md`. Pure functions + frozen dataclasses; `tomllib` for reads, `tomlkit` for writes (comment-preserving). The only CLI surface is `python -m llamactl migrate`. Phases 2–5 (Textual UI, lifecycle, builds, monitors) come in later plans.

**Tech Stack:** Python 3.14 (system), `tomllib` (stdlib), `tomlkit` 0.15 (already installed via pacman), pytest. No new dependencies.

**Conventions for every task:**
- Run tests from the repo root with `python -m pytest <path> -v`.
- Commit messages follow `<type>: <description>` (repo convention, no attribution footer).
- All new code uses `from __future__ import annotations` and type annotations on every function signature.

---

### Task 1: OpenSpec change skeleton

Per the repo's spec-driven workflow, implementation lands under an OpenSpec change with a delta spec for the new `llamactl` capability.

**Files:**
- Create: `openspec/changes/add-llamactl-core/proposal.md`
- Create: `openspec/changes/add-llamactl-core/tasks.md`
- Create: `openspec/changes/add-llamactl-core/specs/llamactl/spec.md`

- [ ] **Step 1: Write proposal.md**

```markdown
# Add llamactl core foundation

## Why

Launching llama-server today means hand-typing long `run.py` invocations, and
every new llama.cpp flag requires editing `build_server_args()` in `run.py`.
The approved design (`docs/superpowers/specs/2026-06-10-llamactl-tui-design.md`)
introduces `llamactl`, a self-contained TUI dashboard. This change delivers its
Phase 1: the UI-agnostic core.

## What Changes

- New `llamactl/core/` package: settings→argv mapper, TOML model/global config
  store, JSON→TOML migration, artifact-registry persistence.
- New `configs/` directory: `llamactl.toml` (global) and `models/*.toml`
  (migrated once from `models/*.json`).
- New CLI entry: `python -m llamactl migrate` (the only CLI surface in Phase 1).
- No existing script changes: `run.py`, `models/*.json`, `build.*.sh`, and the
  stress harness are untouched (frozen legacy path).

## Impact

- Affected scripts: none modified; new package + configs added alongside.
- Host/container: host-only Python tooling; no container or Dockerfile changes.
- New capability spec: `llamactl`.
```

- [ ] **Step 2: Write the delta spec** at `openspec/changes/add-llamactl-core/specs/llamactl/spec.md`

Note: per repo OpenSpec rules, every requirement body MUST have SHALL/MUST on the **first line**.

```markdown
## ADDED Requirements

### Requirement: Convention-based flag mapping
`llamactl.core.mapper.to_argv` SHALL convert settings keys to llama-server flags mechanically: snake_case keys become `--kebab-case` flags, `True` becomes a bare flag, `False`/`None` omit the flag, lists repeat the flag per element, and a small exceptions table maps single-dash flags (`cram` → `-cram`, `hf` → `-hf`); launcher-owned keys (`host`, `port`) are skipped.

#### Scenario: New upstream flag requires no code change
- WHEN a model config contains `some_new_flag = 5` and settings are mapped
- THEN the argv contains `--some-new-flag 5` with no change to llamactl source

#### Scenario: Boolean convention
- WHEN settings contain `no_mmap = true` and `no_warmup = false`
- THEN argv contains `--no-mmap` and contains no `--no-warmup`

### Requirement: Layered settings resolution
`llamactl.core.config.resolve_settings` SHALL merge `[settings]`, then the named preset, then the backend section, then explicit overrides (later layers win), and SHALL raise `ConfigError` naming the available presets when the requested preset does not exist.

#### Scenario: Backend overrides preset
- WHEN a preset sets `cache_type_k = "f16"` and `[backends.rocm]` sets `cache_type_k = "q8_0"`
- THEN resolving with that preset and backend `rocm` yields `cache_type_k = "q8_0"`

#### Scenario: Unknown preset fails fast
- WHEN resolving with preset name `thinking-budgted` (typo)
- THEN a `ConfigError` is raised listing the model's available presets

### Requirement: TOML model config store
`llamactl.core.config.load_all` SHALL load every `configs/models/*.toml`, and SHALL isolate per-file failures: an unparseable or invalid file is reported as an error entry while remaining models load normally.

#### Scenario: One bad file does not break the store
- WHEN `configs/models/` contains one valid model file and one file with a TOML syntax error
- THEN `load_all` returns the valid model plus an error entry naming the bad path

### Requirement: One-shot JSON migration
`python -m llamactl migrate` SHALL convert each `models/*.json` to `configs/models/<id>.toml`, renaming legacy keys (`cache_k` → `cache_type_k`, `cache_v` → `cache_type_v`), converting `flash_attn = true` to `"on"`, converting `prefill_assistant = false` to `no_prefill_assistant = true`, dropping `null`/`false` boolean values, materializing run.py's implicit flag defaults (e.g. `min_p = 0.0`, `repeat_penalty = 1.0`, `jinja = true`) into `[settings]` when absent so observed launch behavior is preserved, and SHALL skip existing destination files unless `--force` is given.

#### Scenario: Re-run is a no-op
- WHEN migrate runs a second time without `--force`
- THEN every existing destination file is reported as `skipped` and left unmodified

### Requirement: Migration argv parity with run.py
The argv produced for a migrated model (settings resolved for backend `rocm`, no preset, mapped via `to_argv`) SHALL contain exactly the same flag tokens as the server-argument portion of `python run.py --model <id> --backend rocm --container --dry-run` for the same model, compared as unordered multisets.

#### Scenario: qwen3.6-35b-a3b parity
- WHEN `models/qwen3.6-35b-a3b.json` is migrated and resolved for rocm with port 8080 and host 0.0.0.0
- THEN the sorted argv tokens equal the sorted server-argument tokens from run.py's dry-run output

### Requirement: Artifact registry persistence
`llamactl.core.registry` SHALL persist build artifacts (target, requested ref, resolved sha, build number, timestamp, image tag or binary path) to `state/registry.toml`, SHALL load an empty list when the file is absent, and SHALL replace an existing entry on `(target, sha)` collision rather than duplicating it.

#### Scenario: Round-trip
- WHEN two artifacts are added and saved, and the file is loaded again
- THEN the loaded artifacts equal the saved ones field-for-field
```

- [ ] **Step 3: Write tasks.md** (mirrors this plan)

```markdown
# Tasks — add-llamactl-core

- [ ] 1. OpenSpec change skeleton (this file) — validated with `openspec validate`
- [ ] 2. Package scaffold: `llamactl/`, root `conftest.py`, `.gitignore` `state/`
- [ ] 3. `core/mapper.py` with golden tests
- [ ] 4. `core/config.py` model loading with error isolation + tests
- [ ] 5. `core/config.py` layered `resolve_settings` + tests
- [ ] 6. `core/config.py` global config + checked-in `configs/llamactl.toml`
- [ ] 7. `core/registry.py` round-trip persistence + tests
- [ ] 8. `core/migrate.py` JSON→TOML conversion + tests
- [ ] 9. `python -m llamactl migrate` CLI; run real migration; commit `configs/models/*.toml`
- [ ] 10. run.py dry-run parity test passes; full suite green

Verification: `python -m pytest tests/llamactl -v` green, plus parity scenario
from the spec checked against `python run.py --model qwen3.6-35b-a3b --backend
rocm --container --dry-run`.
```

- [ ] **Step 4: Validate**

Run: `openspec validate --type change add-llamactl-core`
Expected: validation passes. If it complains about requirement format, check that each `### Requirement:` body has SHALL on its first line.

- [ ] **Step 5: Commit**

```bash
git add openspec/changes/add-llamactl-core/
git commit -m "docs: add llamactl-core OpenSpec change with delta spec"
```

---

### Task 2: Package scaffold

**Files:**
- Create: `llamactl/__init__.py`, `llamactl/core/__init__.py`
- Create: `conftest.py` (repo root)
- Create: `tests/llamactl/` (directory; first test file arrives in Task 3)
- Modify: `.gitignore`

- [ ] **Step 1: Create package directories**

```bash
mkdir -p llamactl/core tests/llamactl
```

`llamactl/__init__.py`:

```python
"""llamactl — TUI dashboard for llama.cpp on ROCm/Vulkan (core in Phase 1)."""
```

`llamactl/core/__init__.py`:

```python
"""UI-agnostic core: config store, mapper, migration, registry."""
```

- [ ] **Step 2: Create root conftest.py**

The existing tests import `stress_harness` relying on the invocation dir being on `sys.path`. A root `conftest.py` makes pytest put the repo root on `sys.path` regardless of invocation style.

```python
"""Repo-root conftest: ensures the repo root is importable in tests."""
```

- [ ] **Step 3: Ignore runtime state**

Append to `.gitignore`:

```
state/
```

- [ ] **Step 4: Verify pytest still collects existing tests**

Run: `python -m pytest tests/ test_run.py --collect-only -q | tail -3`
Expected: same count of collected tests as before adding conftest (no errors).

- [ ] **Step 5: Commit**

```bash
git add llamactl/ conftest.py .gitignore
git commit -m "chore: scaffold llamactl package and test plumbing"
```

---

### Task 3: Settings mapper (`core/mapper.py`)

**Files:**
- Create: `llamactl/core/mapper.py`
- Test: `tests/llamactl/test_mapper.py`

- [ ] **Step 1: Write the failing tests**

`tests/llamactl/test_mapper.py`:

```python
from __future__ import annotations

from llamactl.core.mapper import build_server_argv, to_argv


def test_int_value_maps_to_kebab_flag() -> None:
    assert to_argv({"ctx_size": 262144}) == ["--ctx-size", "262144"]


def test_string_value_passes_through() -> None:
    assert to_argv({"flash_attn": "on"}) == ["--flash-attn", "on"]


def test_float_value() -> None:
    assert to_argv({"temp": 0.6, "min_p": 0.0}) == ["--temp", "0.6", "--min-p", "0.0"]


def test_bool_true_emits_bare_flag() -> None:
    assert to_argv({"no_mmap": True}) == ["--no-mmap"]


def test_bool_false_omits_flag() -> None:
    assert to_argv({"no_warmup": False}) == []


def test_none_omits_flag() -> None:
    assert to_argv({"model_draft": None}) == []


def test_list_repeats_flag() -> None:
    assert to_argv({"lora": ["a.gguf", "b.gguf"]}) == [
        "--lora", "a.gguf", "--lora", "b.gguf",
    ]


def test_exceptions_table_single_dash() -> None:
    assert to_argv({"cram": 2048}) == ["-cram", "2048"]


def test_reserved_keys_skipped() -> None:
    assert to_argv({"host": "0.0.0.0", "port": 8080}) == []


def test_unknown_key_flows_through() -> None:
    # The whole point: new upstream flags need zero code changes.
    assert to_argv({"some_new_flag": 5}) == ["--some-new-flag", "5"]


def test_insertion_order_preserved() -> None:
    assert to_argv({"b_key": 1, "a_key": 2}) == ["--b-key", "1", "--a-key", "2"]


def test_build_server_argv_wraps_hf_host_port() -> None:
    argv = build_server_argv("org/model:Q5", {"ctx_size": 4096}, "0.0.0.0", 8080)
    assert argv == [
        "-hf", "org/model:Q5",
        "--ctx-size", "4096",
        "--host", "0.0.0.0",
        "--port", "8080",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_mapper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llamactl.core.mapper'`

- [ ] **Step 3: Implement the mapper**

`llamactl/core/mapper.py`:

```python
"""Convert resolved settings dicts into llama-server argv.

Convention (no allowlist — unknown keys flow through on purpose):
  snake_case key   -> --kebab-case flag
  True             -> bare flag present
  False / None     -> flag omitted
  list             -> flag repeated per element
  anything else    -> str(value) as the flag's argument
"""

from __future__ import annotations

from typing import Any

# Flags that do not follow the double-dash kebab-case convention.
EXCEPTIONS: dict[str, str] = {
    "cram": "-cram",
    "hf": "-hf",
}

# Keys owned by the launcher (injected by lifecycle), never read from settings.
RESERVED: frozenset[str] = frozenset({"host", "port"})


def _flag_name(key: str) -> str:
    return EXCEPTIONS.get(key, "--" + key.replace("_", "-"))


def to_argv(settings: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in settings.items():
        if key in RESERVED or value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(value)]
    return argv


def build_server_argv(
    hf: str, settings: dict[str, Any], host: str, port: int
) -> list[str]:
    """Full llama-server argument list: model ref, settings, then host/port."""
    return ["-hf", hf, *to_argv(settings), "--host", host, "--port", str(port)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_mapper.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/mapper.py tests/llamactl/test_mapper.py
git commit -m "feat: convention-based settings-to-argv mapper"
```

---

### Task 4: Model config loading (`core/config.py`, part 1)

**Files:**
- Create: `llamactl/core/config.py`
- Test: `tests/llamactl/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/llamactl/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llamactl.core.config'`

- [ ] **Step 3: Implement model loading**

`llamactl/core/config.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/config.py tests/llamactl/test_config.py
git commit -m "feat: TOML model config store with per-file error isolation"
```

---

### Task 5: Layered settings resolution (`core/config.py`, part 2)

**Files:**
- Modify: `llamactl/core/config.py` (append function)
- Test: `tests/llamactl/test_config.py` (append tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/llamactl/test_config.py`)

```python
from llamactl.core.config import resolve_settings


def make_model(tmp_path: Path) -> ModelConfig:
    content = """\
name = "Layered"
hf = "org/layered:Q5"

[settings]
ctx_size = 262144
temp = 0.7
cache_type_k = "f16"

[backends.rocm]
cache_type_k = "q8_0"
no_mmap = true

[presets.cool]
temp = 0.3
cache_type_k = "q4_0"
"""
    return load_model(write(tmp_path, "layered.toml", content))


def test_resolve_defaults_only(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, None, "vulkan", {})
    assert settings == {"ctx_size": 262144, "temp": 0.7, "cache_type_k": "f16"}


def test_backend_overrides_preset(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, "cool", "rocm", {})
    # preset set temp and cache_type_k; backend wins on cache_type_k
    assert settings["temp"] == 0.3
    assert settings["cache_type_k"] == "q8_0"
    assert settings["no_mmap"] is True


def test_overrides_win_over_everything(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, "cool", "rocm", {"cache_type_k": "f16"})
    assert settings["cache_type_k"] == "f16"


def test_none_overrides_are_ignored(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    settings = resolve_settings(model, None, "rocm", {"temp": None})
    assert settings["temp"] == 0.7


def test_unknown_preset_raises_with_available_list(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    with pytest.raises(ConfigError, match="cool"):
        resolve_settings(model, "c00l", "rocm", {})


def test_resolution_does_not_mutate_model(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    before = dict(model.settings)
    resolve_settings(model, "cool", "rocm", {"extra": 1})
    assert model.settings == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_config.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'resolve_settings'`

- [ ] **Step 3: Implement** (append to `llamactl/core/config.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_config.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/config.py tests/llamactl/test_config.py
git commit -m "feat: layered settings resolution with fail-fast preset check"
```

---

### Task 6: Global config (`core/config.py`, part 3) + checked-in `configs/llamactl.toml`

**Files:**
- Modify: `llamactl/core/config.py` (append)
- Create: `configs/llamactl.toml`
- Test: `tests/llamactl/test_config.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/llamactl/test_config.py`)

```python
from llamactl.core.config import GlobalConfig, load_global


def test_load_global_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_global(tmp_path / "nope.toml")
    assert cfg == GlobalConfig()
    assert cfg.default_backend == "rocm"
    assert cfg.port == 8080
    assert cfg.vram_budget_gb == 11.0
    assert cfg.name_prefix == "llamactl"


def test_load_global_partial_file_fills_defaults(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", 'default_backend = "vulkan"\nport = 8081\n')
    cfg = load_global(path)
    assert cfg.default_backend == "vulkan"
    assert cfg.port == 8081
    assert cfg.default_mode == "container"  # untouched default


def test_load_global_expands_user_paths(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", 'hf_cache = "~/somewhere"\n')
    cfg = load_global(path)
    assert "~" not in str(cfg.hf_cache)


def test_load_global_bad_toml_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "g.toml", "port = [broken\n")
    with pytest.raises(ConfigError):
        load_global(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_config.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'GlobalConfig'`

- [ ] **Step 3: Implement** (append to `llamactl/core/config.py`)

```python
@dataclass(frozen=True)
class GlobalConfig:
    default_backend: str = "rocm"
    default_mode: str = "container"
    port: int = 8080
    vram_budget_gb: float = 11.0
    hf_cache: Path = Path("~/.cache/huggingface").expanduser()
    llama_cache: Path = Path("~/.cache/llama.cpp").expanduser()
    name_prefix: str = "llamactl"


def load_global(path: Path) -> GlobalConfig:
    if not path.exists():
        return GlobalConfig()
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    defaults = GlobalConfig()
    return GlobalConfig(
        default_backend=data.get("default_backend", defaults.default_backend),
        default_mode=data.get("default_mode", defaults.default_mode),
        port=data.get("port", defaults.port),
        vram_budget_gb=data.get("vram_budget_gb", defaults.vram_budget_gb),
        hf_cache=Path(data.get("hf_cache", defaults.hf_cache)).expanduser(),
        llama_cache=Path(data.get("llama_cache", defaults.llama_cache)).expanduser(),
        name_prefix=data.get("name_prefix", defaults.name_prefix),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_config.py -v`
Expected: 15 passed

- [ ] **Step 5: Create the checked-in global config**

`configs/llamactl.toml`:

```toml
# llamactl global configuration — values shown are also the built-in defaults.

default_backend = "rocm"      # rocm | vulkan
default_mode = "container"    # container | native
port = 8080
vram_budget_gb = 11.0         # warn threshold on the 12 GB gfx1031 card
name_prefix = "llamactl"      # container-name / label prefix

# Cache mounts (defaults shown; uncomment to override)
# hf_cache = "~/.cache/huggingface"
# llama_cache = "~/.cache/llama.cpp"
```

- [ ] **Step 6: Commit**

```bash
git add llamactl/core/config.py tests/llamactl/test_config.py configs/llamactl.toml
git commit -m "feat: global llamactl config with checked-in defaults file"
```

---

### Task 7: Artifact registry (`core/registry.py`)

**Files:**
- Create: `llamactl/core/registry.py`
- Test: `tests/llamactl/test_registry.py`

- [ ] **Step 1: Write the failing tests**

`tests/llamactl/test_registry.py`:

```python
from __future__ import annotations

from pathlib import Path

from llamactl.core.registry import (
    Artifact,
    add_artifact,
    load_registry,
    remove_artifact,
    save_registry,
)

IMAGE = Artifact(
    target="rocm-image",
    requested_ref="tag:b8586",
    sha="7cadbfce10fc16032cfb576ca4607cd2dd183bf1",
    build_number="b8586",
    built_at="2026-06-10T12:00:00",
    image_tag="llama-cpp-gfx1031:b8586",
)
NATIVE = Artifact(
    target="vulkan-native",
    requested_ref="branch:master",
    sha="abc123",
    build_number="",
    built_at="2026-06-10T13:00:00",
    binary_path="state/builds/abc123/vulkan-native/llama-server",
)


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "registry.toml") == []


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "registry.toml"
    save_registry(path, [IMAGE, NATIVE])
    assert load_registry(path) == [IMAGE, NATIVE]


def test_add_returns_new_list() -> None:
    artifacts: list[Artifact] = []
    result = add_artifact(artifacts, IMAGE)
    assert result == [IMAGE]
    assert artifacts == []  # input not mutated


def test_add_replaces_same_target_and_sha() -> None:
    rebuilt = Artifact(
        target=IMAGE.target,
        requested_ref=IMAGE.requested_ref,
        sha=IMAGE.sha,
        build_number=IMAGE.build_number,
        built_at="2026-06-11T09:00:00",
        image_tag=IMAGE.image_tag,
    )
    result = add_artifact([IMAGE, NATIVE], rebuilt)
    assert result == [NATIVE, rebuilt]


def test_remove_by_target_and_sha() -> None:
    result = remove_artifact([IMAGE, NATIVE], "rocm-image", IMAGE.sha)
    assert result == [NATIVE]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llamactl.core.registry'`

- [ ] **Step 3: Implement the registry**

`llamactl/core/registry.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/registry.py tests/llamactl/test_registry.py
git commit -m "feat: artifact registry with TOML persistence"
```

---

### Task 8: JSON→TOML migration (`core/migrate.py`)

**Files:**
- Create: `llamactl/core/migrate.py`
- Test: `tests/llamactl/test_migrate.py`

- [ ] **Step 1: Write the failing tests**

`tests/llamactl/test_migrate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llamactl/test_migrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llamactl.core.migrate'`

- [ ] **Step 3: Implement migration**

`llamactl/core/migrate.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llamactl/test_migrate.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/migrate.py tests/llamactl/test_migrate.py
git commit -m "feat: one-shot JSON-to-TOML model config migration"
```

---

### Task 9: `python -m llamactl migrate` CLI + real migration

**Files:**
- Create: `llamactl/__main__.py`
- Create (generated): `configs/models/*.toml`

- [ ] **Step 1: Implement the CLI entry point**

`llamactl/__main__.py`:

```python
"""CLI entry point. Phase 1 exposes only `migrate`; the dashboard is Phase 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llamactl.core.migrate import migrate

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llamactl")
    sub = parser.add_subparsers(dest="command")
    mig = sub.add_parser(
        "migrate", help="One-shot import of models/*.json into configs/models/*.toml"
    )
    mig.add_argument("--src", type=Path, default=REPO_ROOT / "models")
    mig.add_argument("--dest", type=Path, default=REPO_ROOT / "configs" / "models")
    mig.add_argument(
        "--force", action="store_true", help="Overwrite existing .toml files"
    )
    args = parser.parse_args(argv)

    if args.command != "migrate":
        parser.print_help()
        print("\nThe dashboard arrives in Phase 2; only 'migrate' exists today.")
        return 0

    failed = False
    for dest, status, warnings in migrate(args.src, args.dest, force=args.force):
        print(f"{status:>8}  {dest}")
        for warning in warnings:
            print(f"          WARNING: {warning}")
        failed = failed or status == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test the help and a dry run against a temp dest**

Run: `python -m llamactl migrate --dest /tmp/llamactl-migrate-test && ls /tmp/llamactl-migrate-test | head -5`
Expected: one `written  …` line per file in `models/` (20 files), then five `.toml` filenames. No `failed` lines, no warnings.

- [ ] **Step 3: Run the real migration**

Run: `python -m llamactl migrate`
Expected: 20 `written` lines into `configs/models/`.

- [ ] **Step 4: Spot-check one migrated file**

Run: `cat configs/models/qwen3.6-35b-a3b.toml`
Expected: `hf = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q5_K_XL"`, a `[settings]` table with `flash_attn = "on"`, `no_prefill_assistant = true`, the materialized implicit defaults `min_p = 0.0` and `repeat_penalty = 1.0` and `jinja = true`, no `no_warmup`, `[backends.rocm]` containing `cache_type_k = "q8_0"` and `no_mmap = true`, and the five `[presets.*]` tables.

- [ ] **Step 5: Verify rerun skips**

Run: `python -m llamactl migrate`
Expected: 20 `skipped` lines.

- [ ] **Step 6: Commit CLI + generated configs**

```bash
git add llamactl/__main__.py configs/models/
git commit -m "feat: llamactl migrate CLI; import model configs from models/*.json"
```

---

### Task 10: run.py dry-run parity test + wrap-up

**Files:**
- Test: `tests/llamactl/test_parity.py`
- Modify: `openspec/changes/add-llamactl-core/tasks.md` (check off completed tasks)

- [ ] **Step 1: Write the parity test**

`tests/llamactl/test_parity.py`:

```python
"""Phase-1 exit criterion: migrated config + mapper reproduces run.py's argv.

Compares the server-argument portion of `run.py --dry-run` (everything after
the image name) against build_server_argv() for the same model, as unordered
multisets — run.py emits flags in hardcoded order, the mapper in TOML order.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from llamactl.core.config import load_model, resolve_settings
from llamactl.core.mapper import build_server_argv
from llamactl.core.migrate import migrate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "qwen3.6-35b-a3b"
IMAGE = "llama-cpp-gfx1031:latest"


def _image_present() -> bool:
    runtime = shutil.which("podman") or shutil.which("docker")
    if not runtime:
        return False
    result = subprocess.run(
        [runtime, "image", "inspect", IMAGE], capture_output=True
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not _image_present(), reason="requires container runtime + rocm image"
)
def test_rocm_container_argv_parity(tmp_path: Path) -> None:
    dry_run = subprocess.run(
        [
            sys.executable, "run.py",
            "--model", MODEL_ID,
            "--backend", "rocm",
            "--container",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    tokens = shlex.split(dry_run.stdout.strip())
    legacy_args = tokens[tokens.index(IMAGE) + 1 :]

    migrate(REPO_ROOT / "models", tmp_path)
    model = load_model(tmp_path / f"{MODEL_ID}.toml")
    settings = resolve_settings(model, None, "rocm", {})
    new_args = build_server_argv(model.hf, settings, "0.0.0.0", 8080)

    assert Counter(new_args) == Counter(legacy_args)
```

- [ ] **Step 2: Run the parity test**

Run: `python -m pytest tests/llamactl/test_parity.py -v`
Expected: 1 passed (or 1 skipped on a machine without the rocm image — on this machine it must PASS).

If it fails, diff the two Counters: every difference must trace to a migration rule (rename/bool conversion) that mishandled a key — fix `migrate.py` or `mapper.py`, never the test, then re-run `python -m llamactl migrate --force` and re-commit `configs/models/`.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest tests/ test_run.py -v`
Expected: all llamactl tests + all pre-existing stress_harness tests + test_run.py PASS, zero failures.

- [ ] **Step 4: Check off OpenSpec tasks**

Edit `openspec/changes/add-llamactl-core/tasks.md`: mark items 1–10 as `- [x]`.

Run: `openspec validate --all`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add tests/llamactl/test_parity.py openspec/changes/add-llamactl-core/tasks.md
git commit -m "test: run.py dry-run parity check closes llamactl phase 1"
```

---

## Out of scope for this plan (later phases / plans)

- Phase 2: lifecycle + monitor + Serve tab (Textual app starts here)
- Phase 3: build subsystem + Builds tab
- Phase 4: Models settings-editor tab
- Phase 5: OOM test tab + pre-launch estimates
- Archiving the OpenSpec change (happens after all phases, or per-phase changes — decide at Phase 2 planning)
