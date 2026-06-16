# Tasks — add-llamactl-core

- [x] 1. OpenSpec change skeleton (this file) — validated with `openspec validate`
- [x] 2. Package scaffold: `llamactl/`, root `conftest.py`, `.gitignore` `state/`
- [x] 3. `core/mapper.py` with golden tests
- [x] 4. `core/config.py` model loading with error isolation + tests
- [x] 5. `core/config.py` layered `resolve_settings` + tests
- [x] 6. `core/config.py` global config + checked-in `configs/llamactl.toml`
- [x] 7. `core/registry.py` round-trip persistence + tests
- [x] 8. `core/migrate.py` JSON→TOML conversion + tests
- [x] 9. `python -m llamactl migrate` CLI; run real migration; commit `configs/models/*.toml`
- [x] 10. run.py dry-run parity test passes; full suite green

Verification: `python -m pytest tests/llamactl -v` green, plus parity scenario
from the spec checked against `python run.py --model qwen3.6-35b-a3b --backend
rocm --container --dry-run`.
