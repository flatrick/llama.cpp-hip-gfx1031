"""LlamaCtlApp — Textual TUI entry point."""
from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from llamactl.core.config import GlobalConfig, ModelConfig, load_all, load_global
from llamactl.core.registry import Artifact, load_registry
from llamactl.ui.screens.builds import BuildsScreen
from llamactl.ui.screens.models import ModelsScreen
from llamactl.ui.screens.serve import ServeScreen


class LlamaCtlApp(App):
    TITLE = "llamactl"
    SUB_TITLE = "llama.cpp on ROCm / Vulkan"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("?", "help", "Help", show=False),
    ]

    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        self._repo_root = repo_root
        self._config_dir = repo_root / "configs"
        self._state_dir = repo_root / "state"
        # These are set in on_mount so the app can compose first
        self._global_cfg: GlobalConfig = GlobalConfig()
        self._models: list[ModelConfig] = []
        self._model_errors: dict[Path, str] = {}
        self._artifacts: list[Artifact] = []

    def on_mount(self) -> None:
        from llamactl.ui.screens.serve import ServeScreen, _LaunchForm

        self._global_cfg = load_global(self._config_dir / "llamactl.toml")
        self._models, self._model_errors = load_all(self._config_dir / "models")
        try:
            self._artifacts = load_registry(self._state_dir / "registry.toml")
        except FileNotFoundError:
            self._artifacts = []
        except Exception as exc:
            self._artifacts = []
            self.notify(f"Registry load failed: {exc}", severity="warning", timeout=5)
        try:
            self.query_one(_LaunchForm).refresh_artifact_options()
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="serve"):
            with TabPane("Serve", id="serve"):
                yield ServeScreen()
            with TabPane("Models", id="models"):
                yield ModelsScreen()
            with TabPane("Builds", id="builds"):
                yield BuildsScreen()
            with TabPane("Test", id="test"):
                yield Static("OOM boundary test — coming in Phase 5")
        yield Footer()

    def action_quit(self) -> None:
        serve = self.query_one(ServeScreen)
        if serve.has_running_server:
            self.notify(
                "Server is still running — stop it first or just close this window.",
                severity="warning",
                timeout=4,
            )
            return
        self.exit()
