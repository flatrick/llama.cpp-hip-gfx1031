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
