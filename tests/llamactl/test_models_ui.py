from __future__ import annotations

from pathlib import Path

import pytest


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "m.toml").write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\nctx_size = 4096\n'
    )
    (tmp_path / "state").mkdir()
    return tmp_path


def _tree_leaf_labels(tree) -> list[str]:
    out: list[str] = []

    def walk(node):
        out.append(str(node.label))
        for child in node.children:
            walk(child)

    walk(tree.root)
    return out


@pytest.mark.asyncio
async def test_models_tab_lists_and_loads_model(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen
    from textual.widgets import Button, Tree

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        assert app.query_one("#model-list") is not None
        assert app.query_one("#editor-tree", Tree) is not None
        assert app.query_one("#btn-save", Button) is not None
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        tree = app.query_one("#editor-tree", Tree)
        labels = _tree_leaf_labels(tree)
        assert any("ctx_size = 4096" in lbl for lbl in labels)


@pytest.mark.asyncio
async def test_edit_value_sets_dirty_and_save_writes(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        ms.apply_set("settings", "ctx_size", "8192")
        assert ms.is_dirty
        ms.action_save()
        await pilot.pause()
        assert not ms.is_dirty
        text = (tmp_path / "configs" / "models" / "m.toml").read_text()
        assert "ctx_size = 8192" in text


@pytest.mark.asyncio
async def test_set_button_with_blank_section_targets_top_level(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen
    from textual.widgets import Button, Input

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        # Leave the section dropdown blank; set a top-level key via the BUTTON path.
        ms.query_one("#key-input", Input).value = "name"
        ms.query_one("#value-input", Input).value = '"Renamed"'
        ms.on_button_pressed(Button.Pressed(ms.query_one("#btn-set", Button)))
        await pilot.pause()
        assert ms._doc["name"] == "Renamed"
        assert "Select" not in ms._doc   # no spurious doc["Select"]["NULL"] table
