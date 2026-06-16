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
