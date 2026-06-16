"""Serve tab skeleton — full layout added in Task 5."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class ServeScreen(Widget):
    DEFAULT_CSS = "ServeScreen { height: 1fr; padding: 1 2; }"

    @property
    def has_running_server(self) -> bool:
        return False  # Task 6 sets this from lifecycle state

    def compose(self) -> ComposeResult:
        yield Static("Serve tab loading…", id="serve-placeholder")
