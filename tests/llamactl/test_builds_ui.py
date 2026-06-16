from __future__ import annotations

from pathlib import Path

import pytest


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()
    return tmp_path


@pytest.mark.asyncio
async def test_builds_tab_has_form_and_table(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Button, DataTable, Input, RichLog, Select

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "builds"
        await pilot.pause()
        assert app.query_one("#ref-input", Input) is not None
        assert app.query_one("#target-select", Select) is not None
        assert app.query_one("#btn-build", Button) is not None
        assert app.query_one("#build-log", RichLog) is not None
        assert app.query_one("#artifact-table", DataTable) is not None
        table = app.query_one("#artifact-table", DataTable)
        assert len(table.columns) == 5
