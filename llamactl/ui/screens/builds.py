"""Builds tab — source-ref/target request form, streaming log, artifact table."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.coordinate import Coordinate
from textual.widget import Widget
from textual.widgets import Button, Checkbox, DataTable, Input, Label, RichLog, Select

from llamactl.core.builds import BuildRequest, delete_artifact, is_in_use, run_build

if TYPE_CHECKING:
    from llamactl.ui.app import LlamaCtlApp

_TARGETS = [
    ("ROCm image", "rocm-image"),
    ("Vulkan image", "vulkan-image"),
    ("ROCm native", "rocm-native"),
    ("Vulkan native", "vulkan-native"),
]


class BuildsScreen(Widget):
    DEFAULT_CSS = """
    BuildsScreen { height: 1fr; layout: vertical; }
    BuildsScreen Input, BuildsScreen Select, BuildsScreen Checkbox { margin-bottom: 1; }
    BuildsScreen #build-log { height: 1fr; border: solid $panel; margin: 1 0; }
    BuildsScreen #artifact-table { height: 10; border: solid $panel; }
    BuildsScreen .button-row { height: auto; layout: horizontal; }
    BuildsScreen .button-row Button { margin-right: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Source ref (submodule, latest-tag, tag:bNNNN, branch:NAME, commit:SHA):")
        yield Input(value="latest-tag", id="ref-input")
        yield Select(options=_TARGETS, value="rocm-image", id="target-select")
        yield Checkbox("Force rebuild (no cache)", id="no-cache-toggle")
        with Widget(classes="button-row"):
            yield Button("Build", id="btn-build", variant="success")
            yield Button("Delete selected", id="btn-delete", variant="error")
        yield RichLog(id="build-log")
        yield DataTable(id="artifact-table")

    def on_mount(self) -> None:
        table = self.query_one("#artifact-table", DataTable)
        table.add_columns("ref", "sha", "build", "date", "target", "in-use")
        self._refresh_table()

    def _refresh_table(self) -> None:
        from llamactl.core.lifecycle import find_running
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        table = self.query_one("#artifact-table", DataTable)
        table.clear()
        try:
            server = find_running(app._global_cfg, app._state_dir)
        except Exception:
            server = None
        # is_in_use may shell out (podman inspect) per artifact; acceptable for the
        # small single-user registry. Revisit with a thread worker if it grows large.
        for art in app._artifacts:
            used = "●" if is_in_use(art, server) else ""
            table.add_row(art.requested_ref, art.sha[:12], art.build_number or "—",
                          (art.built_at or "")[:10], art.target, used,
                          key=f"{art.target}:{art.sha}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-build":
            event.button.disabled = True
            self.query_one("#btn-delete", Button).disabled = True
            ref = self.query_one("#ref-input", Input).value.strip()
            target = str(self.query_one("#target-select", Select).value)
            no_cache = self.query_one("#no-cache-toggle", Checkbox).value
            log = self.query_one("#build-log", RichLog)
            self.run_worker(
                lambda: self._build_thread(ref, target, no_cache, log),
                thread=True, exclusive=True, group="build",
            )
        elif event.button.id == "btn-delete":
            self._delete_selected()

    def _build_thread(self, ref: str, target: str, no_cache: bool, log: RichLog) -> None:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        try:
            req = BuildRequest(ref=ref, target=target, no_cache=no_cache)
            for line in run_build(
                req, app._repo_root, app._state_dir,
                app._repo_root / "llama.cpp-src", app._state_dir / "registry.toml",
            ):
                app.call_from_thread(log.write, line)
        except Exception as exc:
            app.call_from_thread(log.write, f"[red]Build failed:[/red] {exc}")
        finally:
            app.call_from_thread(self._on_build_done)

    def _refresh_serve_picker(self) -> None:
        """Push the updated registry into the Serve tab's artifact picker."""
        from llamactl.ui.screens.serve import _LaunchForm
        try:
            self.app.query_one(_LaunchForm).refresh_artifact_options()
        except Exception:
            pass

    def _on_build_done(self) -> None:
        from llamactl.core.registry import load_registry
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        try:
            app._artifacts = load_registry(app._state_dir / "registry.toml")
        except Exception as exc:
            self.notify(f"Registry reload failed: {exc}", severity="warning")
        self.query_one("#btn-build", Button).disabled = False
        self.query_one("#btn-delete", Button).disabled = False
        self._refresh_table()
        self._refresh_serve_picker()

    def _delete_selected(self) -> None:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        table = self.query_one("#artifact-table", DataTable)
        if not app._artifacts:
            self.notify("Select an artifact row first.", severity="warning")
            return
        try:
            row_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key
            target, sha = str(row_key.value).split(":", 1)
        except Exception:
            self.notify("Could not identify the selected row.", severity="warning")
            return
        match = next((a for a in app._artifacts
                      if a.target == target and a.sha == sha), None)
        if match is None:
            return
        self.run_worker(lambda: self._delete_thread(match), thread=True, group="delete")

    def _delete_thread(self, artifact) -> None:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        remaining = delete_artifact(artifact, app._state_dir,
                                    app._state_dir / "registry.toml")

        def _done() -> None:
            app._artifacts = remaining
            self._refresh_table()
            self._refresh_serve_picker()

        app.call_from_thread(_done)
