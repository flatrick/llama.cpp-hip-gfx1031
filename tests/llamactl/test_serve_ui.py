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
async def test_serve_model_select_lists_all_models_on_startup(tmp_path: Path) -> None:
    """Regression: the Serve model picker is built in _LaunchForm.compose (before the
    app loads models), so the app must refresh it on_mount. Two real models => >= 2
    options; the bug left only the blank placeholder (1)."""
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Select

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    for mid in ("alpha", "beta"):
        (tmp_path / "configs" / "models" / f"{mid}.toml").write_text(
            f'name = "{mid}"\nhf = "org/{mid}:f"\n\n[settings]\nctx_size = 4096\n'
        )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
        sel = app.query_one("#model-select", Select)
        assert len(sel._options) >= 2


@pytest.mark.asyncio
async def test_serve_dropdowns_are_labeled(repo_root: Path) -> None:
    """Each launch-form dropdown carries a visible Label so the controls aren't
    unmarked; the Mode label spells out the container/native choice."""
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Label

    app = LlamaCtlApp(repo_root=repo_root)
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
        for lbl_id in ("#lbl-model", "#lbl-backend", "#lbl-mode",
                       "#lbl-preset", "#lbl-image"):
            assert app.query_one(lbl_id, Label) is not None
        mode_label = str(app.query_one("#lbl-mode", Label).render()).lower()
        assert "container" in mode_label and "native" in mode_label


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


@pytest.mark.asyncio
async def test_launch_button_disabled_during_worker(tmp_path: Path, monkeypatch) -> None:
    """Launch button is disabled immediately when clicked, before worker completes."""
    import asyncio
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Button, Select

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\n'
        'port = 8080\nvram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "m.toml").write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\nctx_size = 2048\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        from llamactl.ui.screens.serve import ServeScreen
        serve = app.query_one(ServeScreen)
        launch_btn = app.query_one("#btn-launch", Button)
        assert not launch_btn.disabled

        # on_button_pressed must disable the button synchronously, before the
        # async worker it schedules gets a chance to run (and possibly re-enable
        # it on an early-return failure path).
        event = Button.Pressed(launch_btn)
        serve.on_button_pressed(event)
        # Button is disabled immediately (synchronous guard against double-launch)
        assert launch_btn.disabled


@pytest.mark.asyncio
async def test_poll_loop_logs_exception_instead_of_crashing(tmp_path: Path, monkeypatch) -> None:
    """An exception in the poll loop is logged to the RichLog, not propagated."""
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.core.lifecycle as lc_mod
    import llamactl.ui.screens.serve as serve_mod

    fake_info = ServerInfo(
        model_id="m", backend="rocm", preset="",
        mode="native", host="0.0.0.0", port=8080,
        started_at="2026-06-16T00:00:00", pid=1234,
    )
    monkeypatch.setattr(lc_mod, "find_running", lambda *_a, **_kw: fake_info)
    monkeypatch.setattr(serve_mod, "_native_pid_alive", lambda pid: True)

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

        # Make check_health blow up
        monkeypatch.setattr(
            "llamactl.core.monitor.check_health",
            lambda *_: (_ for _ in ()).throw(RuntimeError("simulated crash"))
        )

        # Should not raise — exception is caught and logged
        await serve._poll_vram_and_health()
        await pilot.pause(0)
        # Server info must still be intact (poll failure ≠ server gone)
        assert serve._server_info is not None


@pytest.mark.asyncio
async def test_stop_failure_preserves_server_state(tmp_path: Path, monkeypatch) -> None:
    """If stop_server raises, server info is preserved and the UI stays in running state."""
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.core.lifecycle as lc_mod

    fake_info = ServerInfo(
        model_id="m", backend="rocm", preset="",
        mode="native", host="0.0.0.0", port=8080,
        started_at="2026-06-16T00:00:00", pid=1234,
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

        # Make stop_server fail
        monkeypatch.setattr(lc_mod, "stop_server", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("stop failed")))

        await serve._action_stop()
        await pilot.pause(0)

        # Server info must be preserved — stop failed
        assert serve._server_info is not None


@pytest.mark.asyncio
async def test_stopping_server_resets_vram_gauge(tmp_path: Path, monkeypatch) -> None:
    """Stopping a server clears the VRAM gauge instead of leaving a stale reading."""
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.core.lifecycle as lc_mod

    fake_info = ServerInfo(
        model_id="m", backend="rocm", preset="",
        mode="native", host="0.0.0.0", port=8080,
        started_at="2026-06-16T00:00:00", pid=1234,
    )
    monkeypatch.setattr(lc_mod, "find_running", lambda *_a, **_kw: fake_info)
    monkeypatch.setattr(lc_mod, "stop_server", lambda *_a, **_kw: None)

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\n'
        'port = 8080\nvram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        from llamactl.ui.screens.serve import ServeScreen, _VramGauge
        serve = app.query_one(ServeScreen)
        assert serve.has_running_server

        # Simulate a prior non-zero VRAM reading on the gauge.
        gauge = serve.query_one(_VramGauge)
        gauge.vram_kib = 4_000_000
        await pilot.pause(0)

        await serve._action_stop()
        await pilot.pause(0)

        assert serve._server_info is None
        assert gauge.vram_kib == 0


@pytest.mark.asyncio
async def test_dead_native_server_clears_state(tmp_path: Path, monkeypatch) -> None:
    """When liveness probe returns False for a native PID, state clears to EXITED."""
    from llamactl.core.lifecycle import ServerInfo, ServerState
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.core.lifecycle as lc_mod
    import llamactl.ui.screens.serve as serve_mod

    fake_info = ServerInfo(
        model_id="qwen3-9b", backend="rocm", preset="",
        mode="native", host="0.0.0.0", port=8080,
        started_at="2026-06-16T14:00:00",
        pid=9999,
    )
    # Simulate server already running on mount
    monkeypatch.setattr(lc_mod, "find_running", lambda *_a, **_kw: fake_info)
    # Simulate the native process being dead
    monkeypatch.setattr(serve_mod, "_native_pid_alive", lambda pid: False)

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\n'
        'port = 8080\nvram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        from llamactl.ui.screens.serve import ServeScreen, _StatusHeader
        serve = app.query_one(ServeScreen)
        # Confirm server is attached on mount
        assert serve.has_running_server

        # Manually invoke one poll cycle to trigger the liveness probe
        await serve._poll_vram_and_health()
        await pilot.pause(0)

        # Server info should be cleared and status set to EXITED
        assert serve._server_info is None
        header = serve.query_one(_StatusHeader)
        assert header.state == ServerState.EXITED


@pytest.mark.asyncio
async def test_serve_form_has_artifact_select_from_registry(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Select
    from llamactl.core.registry import Artifact, save_registry

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()
    save_registry(
        tmp_path / "state" / "registry.toml",
        [Artifact(target="rocm-image", requested_ref="latest-tag", sha="s1",
                  build_number="b10", built_at="2026-06-16T00:00:00",
                  image_tag="llama-cpp-gfx1031:b10")],
    )

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        artifact_select = app.query_one("#artifact-select", Select)
        values = [v for _label, v in artifact_select._options]
        assert "llama-cpp-gfx1031:b10" in values


@pytest.mark.asyncio
async def test_blank_selects_return_none_not_sentinel(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.serve import _LaunchForm
    from llamactl.core.registry import Artifact, save_registry

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "m.toml").write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\nctx_size = 2048\n'
    )
    (tmp_path / "state").mkdir()
    save_registry(
        tmp_path / "state" / "registry.toml",
        [Artifact(target="rocm-image", requested_ref="latest-tag", sha="s1",
                  build_number="b10", built_at="2026-06-16T00:00:00",
                  image_tag="llama-cpp-gfx1031:b10")],
    )

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        form = app.query_one(_LaunchForm)
        # Default state: no artifact and no preset selected → must be None, not "Select.NULL"
        assert form._get_selected_artifact() is None
        assert form._get_selected_preset() is None
