## Why

The launcher and model config system has grown organically and now has two structural problems:

1. **Model config proliferation**: one JSON file per model×thinking-variant combination has produced
   21 files for ~10 underlying models. Adding a new model means creating 5-7 near-identical files
   that differ only in a handful of sampling params. This is hard to maintain and impossible to
   browse quickly.

2. **Backend conflation**: the current `--backend` choices (`rocm-docker`, `vulkan-docker`, `vulkan`,
   `rocm`) conflate two orthogonal concerns — GPU type (ROCm vs Vulkan) and execution mode (container
   vs native binary). Model configs duplicate settings across `rocm-docker` and `rocm` keys that are
   always identical. ROCm native is currently non-functional due to host dependency issues; only
   `rocm-docker` is used in practice.

## What Changes

### Model JSON: add `presets`, collapse `backends` to 2 keys

- Add a `presets` section: named overlay dicts (thinking-unrestricted, thinking-budgeted,
  thinking-disabled, etc.) that override model defaults with model-specific values
- Collapse `backends` from 4 keys (`rocm-docker`, `vulkan-docker`, `vulkan`, `rocm`) to 2
  (`rocm`, `vulkan`) — execution mode is no longer a config concern
- Reduce ~21 model JSON files to ~10 (one per underlying model)

### run.py: split backend into GPU type + `--container` flag

- `--backend` choices: `rocm`, `vulkan` (GPU type only)
- New `--container` flag: wraps invocation in podman/docker; omitting it runs the binary natively
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` set only for native ROCm (already baked into Docker image)
- New `--preset` flag: applies named preset from model JSON before backend and CLI overrides

### Settings resolution order

```
model defaults → preset → backend → CLI flags
```

### `--list-models` → `-l` / `--list` with smart filtering

| flags | output |
|-------|--------|
| `-l` | all models with their preset names |
| `-l --model <name>` | that model's presets with key values shown |
| `-l --preset <name>` | all models that have that preset |
| `-l --model <name> --preset <name>` | fully resolved settings for that combination |

### Error handling

- Unknown `--preset` value: hard error listing available presets for the selected model
- Unknown `--model` value: existing behaviour (already lists available models)

## Capabilities

### Modified Capabilities

- `launcher`: `run.py` — unified launcher for llama-server
- `model-configs`: `models/*.json` — per-model configuration and preset definitions

### Removed Capabilities

- Native `rocm` backend was dead code; removed cleanly (re-enabling later = drop `--container`)

## Impact

- `run.py`: backend logic, settings resolution, CLI argument parser, list/error output
- `models/*.json`: all model configs restructured (~21 → ~10 files)
- `test_run.py`: tests updated to match new signatures and new model JSON shape
