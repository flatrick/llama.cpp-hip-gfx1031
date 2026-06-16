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
        assert len(table.columns) == 6


@pytest.mark.asyncio
async def test_build_streams_lines_and_reenables_button(tmp_path: Path, monkeypatch) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Button
    import llamactl.ui.screens.builds as builds_mod

    def fake_run_build(req, *a, **k):
        yield "line one"
        yield "line two"

    monkeypatch.setattr(builds_mod, "run_build", fake_run_build)

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "builds"
        await pilot.pause()
        btn = app.query_one("#btn-build", Button)
        screen = app.query_one(builds_mod.BuildsScreen)
        screen.on_button_pressed(Button.Pressed(btn))
        assert btn.disabled  # synchronous guard
        # let the thread worker finish and post back
        for _ in range(40):
            if not btn.disabled:
                break
            await pilot.pause(0.05)
        assert btn.disabled is False
