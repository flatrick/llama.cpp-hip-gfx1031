## Context

`run.py` is the unified launcher for llama-server. It loads a model JSON config, merges
settings through a layered resolution chain, then either wraps the invocation in a container
or runs a local binary directly. Model configs currently encode execution mode (docker vs native)
as part of the backend key, which has caused config duplication and a proliferating file count.

## Goals / Non-Goals

**Goals:**
- Separate GPU type from execution mode in both the CLI and model configs
- Add a named-preset system so thinking/sampling variants live inside the model JSON
- Provide useful listing commands for discoverability during active model testing
- Keep the settings resolution chain simple and explicit
- Maintain test coverage through the restructure

**Non-Goals:**
- Supporting GPU types other than ROCm and Vulkan
- Auto-detecting GPU type (explicit `--backend` remains required)
- Changing the llama-server arguments themselves
- Modifying Docker build scripts

## Decisions

### Decision 1: `--container` as an additive flag, not a backend variant

`--backend rocm` describes the GPU. `--container` describes how to invoke. These are
orthogonal and should be expressed orthogonally. This also makes re-enabling native ROCm
trivial in future: just stop passing `--container`.

### Decision 2: Presets embedded in model JSON, not a shared file

A shared presets file would require cross-model value standardisation. In practice,
`thinking-unrestricted` for Qwen3.5-9B uses different sampling params than for phi-4.
Embedding presets in each model's JSON keeps values model-specific while using a
consistent naming convention across models.

### Decision 3: Settings resolution order — preset before backend

```
model defaults → preset → backend → CLI flags
```

Preset overrides behavior (reasoning mode, sampling). Backend overrides hardware constraints
(ctx_size, cache types). Hardware constraints should win over behavior preferences, so backend
sits above preset. CLI flags always win.

### Decision 4: Hard error on unknown preset, not silent no-op

During active testing, a typo in `--preset` would silently run with wrong settings if
`.get()` just returns `{}`. A hard error with available-preset list is the right default
for a tool used interactively.

### Decision 5: `-l` / `--list` replaces `--list-models`, gains model/preset filtering

The new listing command reuses the existing `--model` and `--preset` flags to filter output:
- No model, no preset → all models + preset names (overview)
- Model only → that model's presets with values (drill-down)
- Preset only → all models that have it (cross-model search)
- Both → fully resolved settings (verification before running)

### Decision 6: `HSA_OVERRIDE_GFX_VERSION` for native ROCm only

The Docker image bakes in `HSA_OVERRIDE_GFX_VERSION=10.3.0` at build time. Setting it again
at container runtime is harmless but unnecessary. For native ROCm it must be set at
invocation time. Condition: `if backend == rocm AND NOT --container`.

## Model JSON Structure

```json
{
  "name": "Qwen3.5-9B UD-Q5_K_XL",
  "hf": "unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL",
  "defaults": {
    "ctx_size": 262144,
    "cache_k": "q8_0",
    "cache_v": "q8_0",
    "batch_size": 1024,
    "ubatch_size": 256,
    "n_gpu_layers": -1,
    "parallel": 1,
    "flash_attn": true,
    "no_warmup": false,
    "no_mmproj": true,
    "jinja": true,
    "cram": 2048,
    "top_k": 20,
    "top_p": 0.8,
    "temp": 0.7,
    "presence_penalty": 1.5,
    "reasoning": "auto",
    "prefill_assistant": false
  },
  "presets": {
    "thinking-unrestricted": {
      "reasoning": "on",
      "reasoning_budget": -1,
      "temp": 0.6,
      "top_p": 0.95,
      "presence_penalty": 0.0,
      "repeat_penalty": 1.0
    },
    "thinking-budgeted": {
      "reasoning": "on",
      "reasoning_budget": 8192,
      "temp": 0.6,
      "top_p": 0.95,
      "presence_penalty": 0.0,
      "repeat_penalty": 1.0
    },
    "thinking-disabled": {
      "reasoning": "off",
      "reasoning_budget": 0
    }
  },
  "backends": {
    "rocm":   { "ctx_size": 131072, "cache_k": "f16", "cache_v": "f16" },
    "vulkan": {}
  }
}
```

## `resolve_settings` Signature Change

```python
# Before
def resolve_settings(cfg: dict, backend: str, overrides: dict) -> dict:
    settings = dict(cfg.get("defaults", {}))
    settings.update(cfg.get("backends", {}).get(backend, {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings

# After
def resolve_settings(cfg: dict, preset: str | None, backend: str, overrides: dict) -> dict:
    settings = dict(cfg.get("defaults", {}))
    if preset is not None:
        presets = cfg.get("presets", {})
        if preset not in presets:
            available = ", ".join(presets.keys()) or "(none)"
            print(f"ERROR: preset '{preset}' not found for model '{cfg.get('name', '?')}'")
            print(f"  Available presets: {available}")
            print(f"  (was it misspelled?)")
            sys.exit(1)
        settings.update(presets[preset])
    settings.update(cfg.get("backends", {}).get(backend, {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings
```

## CLI Shape

```
python run.py --model <name> --backend rocm|vulkan [--container] [--preset <name>]
              [--ctx-size N] [--cache-k TYPE] [--cache-v TYPE]
              [--temp F] [--top-p F] [--top-k N]
              [--reasoning on|off|auto] [--reasoning-budget N]
              [--min-p F] [--repeat-penalty F] [--prefill-assistant | --no-prefill-assistant]
              [--port N] [--host HOST] [--image TAG] [--binary PATH]
              [--dry-run]

python run.py -l [--model <name>] [--preset <name>]
```

## Risks / Trade-offs

- **Renaming backends breaks existing scripts/aliases** → Any shell alias or script using
  `--backend rocm-docker` will break. Mitigation: document clearly in commit message; the
  old names were confusing enough that the breakage is worth it.
- **Collapsing 21 → 10 files loses git history per-variant** → The per-variant files had
  independent git history. Mitigation: history for the base model file is preserved; variant
  history is minor tuning and not worth preserving separately.
- **Tests need updating** → `test_run.py` references old model filenames and
  `resolve_settings` signature. Updating tests is part of the task list.
