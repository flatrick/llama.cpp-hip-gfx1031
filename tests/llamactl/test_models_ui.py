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


@pytest.mark.asyncio
async def test_resolved_view_renders_merged_settings(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "m.toml").write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\nctx_size = 4096\n'
        '\n[backends.rocm]\ncache_type_k = "f16"\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        text = ms.resolved_text("rocm", None)
        assert "ctx_size" in text and "4096" in text
        assert "cache_type_k" in text and "f16" in text


@pytest.mark.asyncio
async def test_resolved_view_selectors_drive_backend_and_preset(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen
    from textual.widgets import Button, Select, Static

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "m.toml").write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\ntemp = 0.7\n'
        '\n[backends.vulkan]\ncache_type_k = "q8_0"\n'
        '\n[presets.creative]\ntemp = 1.2\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        # Enter resolved mode
        ms.on_button_pressed(Button.Pressed(ms.query_one("#btn-resolved", Button)))
        await pilot.pause()
        # The selectors exist and a preset option for 'creative' is available
        rv_backend = ms.query_one("#rv-backend", Select)
        rv_preset = ms.query_one("#rv-preset", Select)
        preset_values = [v for _label, v in rv_preset._options]
        assert "creative" in preset_values
        # Drive vulkan backend + creative preset, then re-render
        rv_backend.value = "vulkan"
        rv_preset.value = "creative"
        ms.render_resolved()
        await pilot.pause()
        text = ms.query_one("#resolved-view", Static).content
        assert "cache_type_k" in text and "q8_0" in text   # vulkan override merged
        assert "temp = 1.2" in text                          # creative preset wins over base 0.7


@pytest.mark.asyncio
async def test_create_duplicate_delete_model(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        models_dir = tmp_path / "configs" / "models"

        ms.create_model("New One", "org/new:f")
        await pilot.pause()
        assert (models_dir / "New-One.toml").exists()
        assert any(m.id == "New-One" for m in app._models)

        # collision: must not overwrite
        ms.create_model("New One", "org/other:f")
        assert (models_dir / "New-One.toml").read_text().count("org/new:f") == 1

        ms.select_model("m")
        ms.duplicate_model("m copy")
        await pilot.pause()
        assert (models_dir / "m-copy.toml").exists()

        ms.delete_model("New-One")
        await pilot.pause()
        assert not (models_dir / "New-One.toml").exists()
        assert not any(m.id == "New-One" for m in app._models)


@pytest.mark.asyncio
async def test_import_json_creates_toml(tmp_path: Path) -> None:
    import json
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    src = tmp_path / "legacy.json"
    src.write_text(json.dumps({"hf": "org/legacy:f", "name": "Legacy",
                               "defaults": {"ctx_size": 2048}}))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.import_json(src)
        await pilot.pause()
        out = tmp_path / "configs" / "models" / "legacy.toml"
        assert out.exists()
        assert 'hf = "org/legacy:f"' in out.read_text()
        assert any(m.id == "legacy" for m in app._models)


@pytest.mark.asyncio
async def test_new_model_appears_in_serve_picker(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen
    from llamactl.ui.screens.serve import _LaunchForm
    from textual.widgets import Select

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.create_model("Fresh Model", "org/fresh:f")
        await pilot.pause()
        form = app.query_one(_LaunchForm)
        values = [v for _label, v in form.query_one("#model-select", Select)._options]
        assert "Fresh-Model" in values
