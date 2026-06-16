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
