from __future__ import annotations

from pathlib import Path

import pytest


def _repo_with_model(tmp_path: Path) -> Path:
    models = tmp_path / "configs" / "models"
    models.mkdir(parents=True)
    (tmp_path / "state").mkdir()
    # Point cache dirs at empty tmp paths so resolve_gguf_path always returns None.
    hf_cache = tmp_path / "hf_cache"
    llama_cache = tmp_path / "llama_cache"
    hf_cache.mkdir()
    llama_cache.mkdir()
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        f'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
        f'hf_cache = "{hf_cache}"\n'
        f'llama_cache = "{llama_cache}"\n'
    )
    (models / "m.toml").write_text(
        'name = "M"\nhf = "unsloth/Foo-GGUF:Q5_K_M"\n\n[settings]\nctx_size = 4096\n'
    )
    return tmp_path


@pytest.mark.asyncio
async def test_serve_shows_estimate_unavailable_when_no_gguf(tmp_path):
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.serve import _LaunchForm
    from textual.widgets import Static, Select

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        form = app.query_one(_LaunchForm)
        # app.on_mount has already populated app._models; refresh the Select so
        # the model option ("m") is available to pick.
        form.refresh_model_options()
        await pilot.pause()
        model_select = form.query_one("#model-select", Select)
        # select the only model (id == path stem == "m")
        model_select.value = "m"
        await pilot.pause()
        line = form.query_one("#estimate-line", Static)
        assert "unavailable" in str(line.render()).lower()
