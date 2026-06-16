"""Serve tab — status header, VRAM gauge, launch form, and log pane."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, ProgressBar, RichLog, Select, Static

from llamactl.core.config import ModelConfig, resolve_settings
from llamactl.core.lifecycle import ServerInfo, ServerState, resolve_image
from llamactl.core.mapper import build_server_argv

if TYPE_CHECKING:
    from llamactl.ui.app import LlamaCtlApp


# ── Status Header ─────────────────────────────────────────────────────────────

class _StatusHeader(Widget):
    DEFAULT_CSS = """
    _StatusHeader {
        height: auto;
        padding: 0 1;
        layout: horizontal;
    }
    _StatusHeader #state-badge {
        width: auto;
        padding: 0 1;
        margin-right: 1;
    }
    _StatusHeader #state-badge.stopped  { color: $text-muted; }
    _StatusHeader #state-badge.starting { color: $warning; }
    _StatusHeader #state-badge.loading  { color: $warning; }
    _StatusHeader #state-badge.ready    { color: $success; }
    _StatusHeader #state-badge.unhealthy { color: $error; }
    _StatusHeader #state-badge.exited   { color: $text-muted; }
    _StatusHeader #server-meta {
        width: 1fr;
        padding: 0 1;
    }
    """

    state: reactive[ServerState] = reactive(ServerState.STOPPED)
    info: reactive[ServerInfo | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Static("● STOPPED", id="state-badge", classes="stopped")
        yield Static("No server running", id="server-meta")

    def watch_state(self, new_state: ServerState) -> None:
        try:
            badge = self.query_one("#state-badge", Static)
        except NoMatches:
            return
        badge.update(f"● {new_state.value.upper()}")
        # Replace all state CSS classes with the current one
        for s in ServerState:
            badge.remove_class(s.name.lower())
        badge.add_class(new_state.name.lower())

    def watch_info(self, new_info: ServerInfo | None) -> None:
        try:
            meta = self.query_one("#server-meta", Static)
        except NoMatches:
            return
        if new_info is None:
            meta.update("No server running")
        else:
            text = (
                f"{new_info.model_id}  backend={new_info.backend}"
                f"  mode={new_info.mode}  port={new_info.port}"
            )
            if new_info.preset:
                text += f"  preset={new_info.preset}"
            meta.update(text)


# ── VRAM Gauge ────────────────────────────────────────────────────────────────

class _VramGauge(Widget):
    DEFAULT_CSS = """
    _VramGauge {
        height: auto;
        padding: 0 1;
        layout: horizontal;
    }
    _VramGauge Label {
        width: auto;
        padding: 0 1;
    }
    _VramGauge ProgressBar {
        width: 1fr;
    }
    _VramGauge #vram-label {
        width: auto;
        padding: 0 1;
    }
    _VramGauge._vram-over ProgressBar > .bar--bar {
        color: $error;
    }
    """

    vram_kib: reactive[int] = reactive(0)
    budget_kib: reactive[int] = reactive(11 * 1024 * 1024)

    def compose(self) -> ComposeResult:
        yield Label("VRAM")
        yield ProgressBar(id="vram-bar", total=100, show_eta=False)
        yield Static("0.0 / 11.0 GiB", id="vram-label")

    def watch_vram_kib(self, new_vram: int) -> None:
        try:
            bar = self.query_one("#vram-bar", ProgressBar)
            label = self.query_one("#vram-label", Static)
        except NoMatches:
            return

        budget = self.budget_kib
        used_gib = new_vram / (1024 * 1024)
        budget_gib = budget / (1024 * 1024)
        label.update(f"{used_gib:.1f} / {budget_gib:.1f} GiB")

        if budget > 0:
            progress = min(100, int(new_vram * 100 / budget))
        else:
            progress = 0
        bar.progress = progress

        if new_vram > budget:
            self.add_class("_vram-over")
        else:
            self.remove_class("_vram-over")


# ── Launch Form ───────────────────────────────────────────────────────────────

class _LaunchForm(Widget):
    DEFAULT_CSS = """
    _LaunchForm {
        height: auto;
        padding: 1;
        layout: vertical;
    }
    _LaunchForm Select {
        margin-bottom: 1;
    }
    _LaunchForm .button-row {
        height: auto;
        layout: horizontal;
    }
    _LaunchForm .button-row Button {
        margin-right: 1;
    }
    _LaunchForm #argv-preview {
        margin-top: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        models: list[ModelConfig],
        model_errors: dict[Path, str],
    ) -> None:
        super().__init__()
        self._models = models
        self._model_errors = model_errors
        # Build a lookup dict for quick access by id
        self._model_map: dict[str, ModelConfig] = {m.id: m for m in models}

    def compose(self) -> ComposeResult:
        model_options = [(m.name, m.id) for m in self._models]

        if model_options:
            yield Select(options=model_options, id="model-select")
        else:
            yield Select(options=[], id="model-select", allow_blank=True)

        yield Select(
            options=[("ROCm", "rocm"), ("Vulkan", "vulkan")],
            id="backend-select",
            value="rocm",
        )
        yield Select(
            options=[("Container", "container"), ("Native", "native")],
            id="mode-select",
            value="container",
        )
        yield Select(options=[], id="preset-select", allow_blank=True)

        with Widget(classes="button-row"):
            yield Button("Launch", id="btn-launch", variant="success")
            yield Button("Stop", id="btn-stop", variant="error", disabled=True)
            yield Button("Copy argv", id="btn-copy-argv")

        yield Static("(select a model to preview the launch command)", id="argv-preview")

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id
        if select_id in ("model-select", "backend-select", "preset-select"):
            if select_id == "model-select":
                self._refresh_preset_options()
            self._refresh_argv_preview()

    def _get_selected_model(self) -> ModelConfig | None:
        try:
            model_select = self.query_one("#model-select", Select)
        except NoMatches:
            return None
        value = model_select.value
        if value is Select.BLANK or value is None:
            return None
        return self._model_map.get(str(value))

    def _get_selected_backend(self) -> str:
        try:
            backend_select = self.query_one("#backend-select", Select)
            value = backend_select.value
            if value is Select.BLANK or value is None:
                return "rocm"
            return str(value)
        except NoMatches:
            return "rocm"

    def _get_selected_preset(self) -> str | None:
        try:
            preset_select = self.query_one("#preset-select", Select)
            value = preset_select.value
            if value is Select.BLANK or value is None:
                return None
            return str(value)
        except NoMatches:
            return None

    def _refresh_preset_options(self) -> None:
        try:
            preset_select = self.query_one("#preset-select", Select)
        except NoMatches:
            return
        model = self._get_selected_model()
        if model is None:
            preset_select.set_options([])
            return
        options = [(name, name) for name in model.presets]
        preset_select.set_options(options)

    def _refresh_argv_preview(self) -> None:
        try:
            preview = self.query_one("#argv-preview", Static)
        except NoMatches:
            return

        model = self._get_selected_model()
        if model is None:
            preview.update("(select a model to preview the launch command)")
            return

        backend = self._get_selected_backend()
        preset = self._get_selected_preset()

        try:
            app: LlamaCtlApp = self.app  # type: ignore[assignment]
            port = app._global_cfg.port
            settings = resolve_settings(model, preset, backend, {})
            image = resolve_image(model, backend)
            argv = build_server_argv(model.hf, settings, "0.0.0.0", port)
            argv_str = " ".join(argv)
            preview.update(f"[bold]Image:[/bold] {image}\n[bold]argv:[/bold] {argv_str}")
        except Exception as exc:
            preview.update(f"[red]Config error: {exc}[/red]")

    def get_launch_params(
        self,
    ) -> tuple[ModelConfig | None, str, str | None]:
        """Return (selected model, selected backend, selected preset or None)."""
        model = self._get_selected_model()
        backend = self._get_selected_backend()
        preset = self._get_selected_preset()
        return model, backend, preset


# ── Serve Screen ──────────────────────────────────────────────────────────────

class ServeScreen(Widget):
    DEFAULT_CSS = """
    ServeScreen {
        height: 1fr;
        layout: vertical;
    }
    ServeScreen #log-pane {
        height: 1fr;
        border: solid $panel;
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._server_info: ServerInfo | None = None

    @property
    def has_running_server(self) -> bool:
        return self._server_info is not None

    def compose(self) -> ComposeResult:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        models = getattr(app, "_models", [])
        model_errors = getattr(app, "_model_errors", {})

        yield _StatusHeader()
        yield _VramGauge()
        yield _LaunchForm(models, model_errors)
        yield RichLog(id="log-pane")

    def on_mount(self) -> None:
        """Re-attach to any already-running server, then start polling."""
        import llamactl.core.lifecycle as _lc
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        info = _lc.find_running(app._global_cfg, app._state_dir)
        if info is not None:
            self._set_server(info)
        self.set_interval(2.0, self._poll_vram_and_health)

    def _set_server(self, info: ServerInfo | None) -> None:
        """Update all reactive state when a server starts or stops."""
        self._server_info = info
        header = self.query_one(_StatusHeader)
        header.info = info
        form = self.query_one(_LaunchForm)
        stop_btn = form.query_one("#btn-stop", Button)
        launch_btn = form.query_one("#btn-launch", Button)
        if info is not None:
            stop_btn.disabled = False
            launch_btn.disabled = True
            header.state = ServerState.STARTING
        else:
            stop_btn.disabled = True
            launch_btn.disabled = False
            header.state = ServerState.STOPPED

    async def _poll_vram_and_health(self) -> None:
        """Called every 2 seconds by set_interval. Updates VRAM gauge and health state."""
        if self._server_info is None:
            return
        import asyncio
        from llamactl.core.monitor import check_health, read_vram_kib
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        port = self._server_info.port
        pid = str(self._server_info.pid) if self._server_info.pid is not None else None

        new_state = await asyncio.to_thread(check_health, port)

        header = self.query_one(_StatusHeader)
        header.state = new_state

        if new_state == ServerState.EXITED:
            self._set_server(None)
            return

        if pid is not None:
            vram_kib = await asyncio.to_thread(read_vram_kib, pid)
            gauge = self.query_one(_VramGauge)
            gauge.vram_kib = vram_kib
            gauge.budget_kib = int(app._global_cfg.vram_budget_gb * 1024 * 1024)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-launch":
            self._action_launch()
        elif event.button.id == "btn-stop":
            self._action_stop()
        elif event.button.id == "btn-copy-argv":
            self._action_copy_argv()

    def _action_launch(self) -> None:
        from llamactl.core.lifecycle import (
            launch_container,
            launch_native,
            resolve_image,
        )
        from llamactl.core.runtime import find_runtime
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        form = self.query_one(_LaunchForm)
        model, backend, preset = form.get_launch_params()
        if model is None:
            self.notify("Select a model first.", severity="warning")
            return
        mode_select = form.query_one("#mode-select", Select)
        mode = str(mode_select.value or "container")
        log = self.query_one("#log-pane", RichLog)
        try:
            settings = resolve_settings(model, preset, backend, {})
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        try:
            if mode == "container":
                rt = find_runtime()
                if rt is None:
                    self.notify("No container runtime found (podman/docker).", severity="error")
                    return
                image = resolve_image(model, backend)
                info = launch_container(
                    runtime=rt,
                    global_cfg=app._global_cfg,
                    model=model,
                    resolved_settings=settings,
                    backend=backend,
                    preset=preset or "",
                    image=image,
                    state_dir=app._state_dir,
                )
                log.write(f"[green]Container started:[/green] {info.container_name}")
            else:
                import shutil as _shutil
                binary = _shutil.which("llama-server") or ""
                if not binary:
                    self.notify("llama-server not found in PATH.", severity="error")
                    return
                info = launch_native(
                    global_cfg=app._global_cfg,
                    model=model,
                    resolved_settings=settings,
                    backend=backend,
                    preset=preset or "",
                    binary=binary,
                    state_dir=app._state_dir,
                )
                log.write(f"[green]Native server started:[/green] PID {info.pid}")
        except Exception as exc:
            self.notify(f"Launch failed: {exc}", severity="error")
            log.write(f"[red]Launch error:[/red] {exc}")
            return
        self._set_server(info)

    def _action_stop(self) -> None:
        if self._server_info is None:
            return
        from llamactl.core.lifecycle import stop_server
        from llamactl.core.runtime import find_runtime
        log = self.query_one("#log-pane", RichLog)
        try:
            rt = find_runtime() if self._server_info.mode == "container" else None
            stop_server(self._server_info, runtime=rt)
            log.write("[yellow]Server stopped.[/yellow]")
        except Exception as exc:
            self.notify(f"Stop failed: {exc}", severity="error")
            log.write(f"[red]Stop error:[/red] {exc}")
        self._set_server(None)

    def _action_copy_argv(self) -> None:
        form = self.query_one(_LaunchForm)
        preview = form.query_one("#argv-preview", Static)
        text = str(preview.renderable)
        try:
            import pyperclip  # optional dep
            pyperclip.copy(text)
            self.notify("argv copied to clipboard.")
        except Exception:
            self.notify("Install pyperclip to enable copy.", severity="warning")
