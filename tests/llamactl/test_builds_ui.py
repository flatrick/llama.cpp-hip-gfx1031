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


@pytest.mark.asyncio
async def test_builds_tab_shows_existing_artifacts_on_startup(tmp_path: Path) -> None:
    """Regression: artifacts from a prior session must appear in the Builds table
    on startup. The table is built in BuildsScreen.on_mount, which runs before
    LlamaCtlApp.on_mount loads the registry, so it needs a post-load refresh."""
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.core.registry import Artifact, save_registry
    from textual.widgets import DataTable

    repo = _repo(tmp_path)
    save_registry(
        repo / "state" / "registry.toml",
        [Artifact(target="rocm-image", requested_ref="latest-tag", sha="s1deadbeef",
                  build_number="b10", built_at="2026-06-16T00:00:00",
                  image_tag="llama-cpp-gfx1031:b10")],
    )
    app = LlamaCtlApp(repo_root=repo)
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "builds"
        await pilot.pause()
        table = app.query_one("#artifact-table", DataTable)
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_builds_tab_has_no_cache_toggle_unchecked_by_default(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Checkbox

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "builds"
        await pilot.pause()
        toggle = app.query_one("#no-cache-toggle", Checkbox)
        assert toggle.value is False  # cache enabled by default


@pytest.mark.asyncio
async def test_build_passes_no_cache_from_toggle(tmp_path: Path, monkeypatch) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Button, Checkbox
    import llamactl.ui.screens.builds as builds_mod

    captured = {}

    def fake_run_build(req, *a, **k):
        captured["no_cache"] = req.no_cache
        yield "done"

    monkeypatch.setattr(builds_mod, "run_build", fake_run_build)

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "builds"
        await pilot.pause()
        app.query_one("#no-cache-toggle", Checkbox).value = True
        btn = app.query_one("#btn-build", Button)
        screen = app.query_one(builds_mod.BuildsScreen)
        screen.on_button_pressed(Button.Pressed(btn))
        for _ in range(40):
            if "no_cache" in captured:
                break
            await pilot.pause(0.05)
        assert captured.get("no_cache") is True
