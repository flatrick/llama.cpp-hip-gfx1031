from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Minimal repo layout with empty models dir and defaults global config."""
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "state").mkdir()
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    return tmp_path


@pytest.mark.asyncio
async def test_app_boots_without_error(repo_root: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    app = LlamaCtlApp(repo_root=repo_root)
    async with app.run_test(headless=True) as pilot:
        assert pilot.app.is_running


@pytest.mark.asyncio
async def test_serve_tab_is_active_on_startup(repo_root: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import TabbedContent
    app = LlamaCtlApp(repo_root=repo_root)
    async with app.run_test(headless=True) as pilot:
        tabs = app.query_one(TabbedContent)
        assert tabs.active == "serve"


@pytest.mark.asyncio
async def test_tab_switching_works(repo_root: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    app = LlamaCtlApp(repo_root=repo_root)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        assert pilot.app.is_running


@pytest.mark.asyncio
async def test_model_selector_populated_from_models(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Select

    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "test-model.toml").write_text(
        'name = "Test Model"\nhf = "org/model:file"\n\n[settings]\nctx_size = 4096\n'
    )
    (tmp_path / "state").mkdir(exist_ok=True)

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        model_select = app.query_one("#model-select", Select)
        # The select must have at least one option (our test model)
        assert len(model_select._options) > 0


@pytest.mark.asyncio
async def test_argv_preview_widget_is_present(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Static

    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir(exist_ok=True)

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        # The argv preview Static must be present in the DOM
        preview = app.query_one("#argv-preview", Static)
        assert preview is not None


@pytest.mark.asyncio
async def test_serve_screen_reattaches_on_mount(tmp_path: Path, monkeypatch) -> None:
    """find_running is called on mount; if a server is already up, status updates."""
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.core.lifecycle as lc_mod

    fake_info = ServerInfo(
        model_id="qwen3-9b", backend="rocm", preset="",
        mode="container", host="0.0.0.0", port=8080,
        started_at="2026-06-16T14:00:00",
        container_name="llamactl-qwen3-9b",
    )
    monkeypatch.setattr(lc_mod, "find_running", lambda *_a, **_kw: fake_info)

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\n'
        'port = 8080\nvram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        from llamactl.ui.screens.serve import ServeScreen
        serve = app.query_one(ServeScreen)
        assert serve.has_running_server
        assert serve._server_info == fake_info
