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


@pytest.mark.asyncio
async def test_test_tab_disabled_without_server(tmp_path):
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import Button, TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        btn = screen.query_one("#btn-run-test", Button)
        assert btn.disabled is True


@pytest.mark.asyncio
async def test_test_tab_notices_server_started_after_mount(tmp_path, monkeypatch):
    """Switching to the Test tab re-checks for a running server. A container
    started (from the Serve tab) after the Test pane mounted must be noticed."""
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.ui.screens.test as test_mod
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import Button, TabbedContent

    state: dict[str, ServerInfo | None] = {"server": None}
    monkeypatch.setattr(test_mod, "find_running", lambda *_a, **_kw: state["server"])

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        screen = app.query_one(TestScreen)
        btn = screen.query_one("#btn-run-test", Button)
        # No server at mount → run disabled.
        assert btn.disabled is True

        # A container comes up after mount.
        state["server"] = ServerInfo(
            model_id="m", backend="rocm", preset="", mode="container",
            host="0.0.0.0", port=8080, started_at="", container_name="llamactl-m",
        )
        # Switching to the Test tab must re-detect it and enable the button.
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()

        assert btn.disabled is False


@pytest.mark.asyncio
async def test_verdict_banner_renders(tmp_path):
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen, _Verdict
    from llamactl.core.oomtest import OomTestResult
    from textual.widgets import Static, TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        screen.post_message(_Verdict(
            OomTestResult("WARN", 11.3, 120000, None, "peak over budget")))
        await pilot.pause()
        verdict = screen.query_one("#verdict", Static)
        # Use .render() rather than .renderable — consistent with the Serve
        # tab smoke test (test_serve_shows_estimate_unavailable_when_no_gguf)
        # which found that this Textual version exposes updated content via
        # render() rather than .renderable in headless pilot mode.
        assert "WARN" in str(verdict.render())


@pytest.mark.asyncio
async def test_cancel_signal_is_threading_event(tmp_path):
    """Cancellation must be a threading.Event (explicit cross-thread signal),
    available immediately after construction — not a plain bool set in on_mount."""
    import threading
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        assert isinstance(screen._cancel_evt, threading.Event)
        assert not screen._cancel_evt.is_set()


@pytest.mark.asyncio
async def test_unmount_sets_cancel_signal(tmp_path):
    """Tearing down the screen mid-run must signal cancellation so the worker
    stops before the next phase (no thread leak / shutdown hang)."""
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        # Simulate an in-progress run.
        screen._test_active = True
        screen.on_unmount()
        assert screen._cancel_evt.is_set()


@pytest.mark.asyncio
async def test_quick_checkbox_default_checked_and_passed(tmp_path, monkeypatch):
    """The Quick-mode checkbox defaults to checked and its value is passed as quick."""
    from llamactl.ui.app import LlamaCtlApp
    import llamactl.ui.screens.test as test_mod
    from llamactl.core.lifecycle import ServerInfo
    from llamactl.ui.screens.test import TestScreen
    from textual.widgets import Checkbox, TabbedContent

    server = ServerInfo(
        model_id="m", backend="rocm", preset="", mode="native",
        host="0.0.0.0", port=8080, started_at="", pid=1234,
    )
    monkeypatch.setattr(test_mod, "find_running", lambda *_a, **_kw: server)

    captured = {}
    def fake_run_oom_check(srv, reporter, *, global_cfg, cancel, quick=True):
        captured["quick"] = quick
        from llamactl.core.oomtest import OomTestResult
        return OomTestResult("OK", 1.0, 100, None, "done")
    monkeypatch.setattr(test_mod, "run_oom_check", fake_run_oom_check)

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        cb = screen.query_one("#quick-mode", Checkbox)
        assert cb.value is True  # default checked = quick

        # Uncheck → full run.
        cb.value = False
        await pilot.pause()
        screen._start_run(screen.query_one("#btn-run-test"))
        # Worker runs on a thread; give it a moment to call run_oom_check.
        for _ in range(50):
            if "quick" in captured:
                break
            await pilot.pause(0.02)
        assert captured["quick"] is False


@pytest.mark.asyncio
async def test_verdict_detail_markup_is_escaped(tmp_path):
    """Log-excerpt detail containing Textual markup tags must render literally,
    not be parsed as markup (which would silently strip the tags from the
    visible message). Non-vacuous: fails if result.detail is not escaped."""
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.test import TestScreen, _Verdict
    from llamactl.core.oomtest import OomTestResult
    from textual.widgets import Static, TabbedContent

    app = LlamaCtlApp(repo_root=_repo_with_model(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one(TabbedContent).active = "test"
        await pilot.pause()
        screen = app.query_one(TestScreen)
        # A log excerpt that happens to contain valid Textual markup tokens.
        screen.post_message(_Verdict(
            OomTestResult("FAIL", None, None, "ramp",
                          "load error [bold]oom[/] at [red]layer 9")))
        await pilot.pause()
        plain = screen.query_one("#verdict", Static).render().plain
        # Without escaping, "[bold]"/"[/]"/"[red]" are consumed as markup and
        # vanish from the plain text; with escaping they survive literally.
        assert "[bold]" in plain
        assert "[red]" in plain
