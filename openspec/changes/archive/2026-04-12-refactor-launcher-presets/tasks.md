## 1. Restructure model JSON files

- [x] 1.1 Rewrite `models/qwen3.5-9b.json` — add `presets` section with thinking-unrestricted,
      thinking-budgeted, thinking-disabled; collapse `backends` to `rocm`/`vulkan` keys only
- [x] 1.2 Rewrite `models/qwen3.5-0.8b.json` — same structure
- [x] 1.3 Rewrite `models/qwen3.5-2b.json` — same structure
- [x] 1.4 Rewrite `models/qwen3.5-4b.json` — same structure
- [x] 1.5 Rewrite `models/phi-4.json` — add presets where applicable; collapse backends
- [x] 1.6 Rewrite `models/phi-4-reasoning.json` — add presets; collapse backends
- [x] 1.7 Rewrite `models/phi-4-reasoning-plus.json` — add presets; collapse backends
- [x] 1.8 Rewrite `models/gemma-4-e4b-it.json` — collapse backends (no thinking presets needed)
- [x] 1.9 Rewrite `models/ministral-3-14b-instruct.json` — add presets; collapse backends
- [x] 1.10 Rewrite `models/ministral-3-14b-reasoning.json` — add presets; collapse backends
- [x] 1.11 Rewrite `models/deepseek-coder-6.7B-kexer.json` — collapse backends
- [x] 1.12 Rewrite `models/deepseek-coder-7B-instruct.json` — collapse backends
- [x] 1.13 Rewrite `models/deepseek-coder-v2-lite-instruct.json` — collapse backends
- [x] 1.14 Delete all `qwen3.5-9b.thinking-*.json` variant files (7 files)
- [x] 1.15 Delete all `gemma-4-e4b-it.thinking-*.json` variant files (2 files)

## 2. Refactor `run.py`

- [x] 2.1 Update `resolve_settings(cfg, backend, overrides)` →
      `resolve_settings(cfg, preset, backend, overrides)` with hard error on unknown preset
- [x] 2.2 Change `--backend` choices from 4-way string to `rocm` / `vulkan`
- [x] 2.3 Add `--container` flag
- [x] 2.4 Add `--preset` / `-p` argument
- [x] 2.5 Add `--temp`, `--top-p`, `--top-k` CLI overrides (add to `cli_overrides` dict and
      `build_server_args`)
- [x] 2.6 Replace `--list-models` with `-l` / `--list`; implement 4-case filtering logic
      (no args, model only, preset only, both)
- [x] 2.7 Rewrite `run_rocm()` and `run_vulkan_docker()` into a single `run_container()`
      that accepts GPU type and builds appropriate device flags
- [x] 2.8 Rewrite `run_rocm_local()` and `run_vulkan_local()` into a single `run_native()`
      that sets `HSA_OVERRIDE_GFX_VERSION` only when backend is `rocm`
- [x] 2.9 Remove dead `run_local()` function
- [x] 2.10 Update `main()` dispatch: replace 4-way backend if/elif with 2-way
      (container vs native) × GPU type
- [x] 2.11 Update `check_image()` builder hint to use new flag names
- [x] 2.12 Update module docstring and examples

## 3. Update tests

- [x] 3.1 Update `resolve_settings` call signatures throughout `test_run.py`
- [x] 3.2 Add tests for preset resolution (known preset, unknown preset error)
- [x] 3.3 Add tests for new `--temp`, `--top-p`, `--top-k` CLI overrides
- [x] 3.4 Update model name references to use consolidated JSON files
- [x] 3.5 Add tests for `-l` listing logic (mock model configs)
- [x] 3.6 Verify all existing tests still pass

## 4. Verify

- [x] 4.1 Run `pytest test_run.py -v` — all tests pass (53/53)
- [x] 4.2 Dry-run spot check: `python run.py --model qwen3.5-9b --backend rocm --container
      --preset thinking-unrestricted --dry-run` — verify correct args emitted
- [x] 4.3 Dry-run spot check: `python run.py --model qwen3.5-9b --backend vulkan
      --preset thinking-budgeted --dry-run` — native Vulkan path
- [x] 4.4 List check: `python run.py -l` — shows all models with preset names
- [x] 4.5 List check: `python run.py -l --model qwen3.5-9b` — shows qwen presets with values
- [x] 4.6 List check: `python run.py -l --preset thinking-unrestricted` — models that have it
- [x] 4.7 Error check: `python run.py --model qwen3.5-9b --backend rocm --container
      --preset nonexistent` — hard error with suggestions
