"""Test tab — quick OOM boundary check against the running server."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, DataTable, Label, Static

from llamactl.core.lifecycle import find_running
from llamactl.core.oomtest import OomTestResult, run_oom_check


# ---------------------------------------------------------------------------
# Thread-safe messages (posted from reporter worker thread; handled on UI loop)
# ---------------------------------------------------------------------------

class _PhaseRow(Message):
    """A single data-table row produced by the reporter worker."""

    def __init__(self, cells: tuple[str, ...]) -> None:
        self.cells = cells
        super().__init__()


class _Verdict(Message):
    """Final test verdict; posted after run_oom_check returns."""

    def __init__(self, result: OomTestResult) -> None:
        self.result = result
        super().__init__()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: float | None) -> str:
    """Format an optional float as a 2-decimal string."""
    return f"{value:.2f}" if value is not None else "n/a"


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class TextualReporter:
    """Drop-in for ConsoleReporter; posts messages to the screen instead of
    printing.

    Called from the worker + sampler threads — only posts messages,
    never touches widgets directly (thread-safety rule).
    """

    def __init__(self, screen: "TestScreen") -> None:
        self._screen = screen

    def start_run(self, result) -> None:
        self._screen.post_message(_PhaseRow((
            "",
            f"ctx={result.ctx_size:,}",
            "",
            "",
            _fmt(result.baseline_vram_gb),
            "",
            "baseline",
        )))

    def start_phase(self, phase) -> None:
        self._screen.post_message(_PhaseRow(("", f"── {phase.title} ──", "", "", "", "", "")))

    def record_sample(self, phase_key, sample, warn_at) -> None:
        req = sample.request
        prefill = req.prefill_display() if req else "—"
        gen = req.gen_toks_display() if req else "—"
        self._screen.post_message(_PhaseRow((
            phase_key,
            str(sample.label),
            prefill,
            gen,
            _fmt(sample.peak_vram_gb),
            _fmt(sample.post_vram_gb),
            sample.status,
        )))

    def finish_phase(self, phase) -> None:
        return None

    def finish_run(self, result) -> None:
        # Verdict banner is set from run_oom_check's return value in _run_worker.
        return None

    def error(self, message: str) -> None:
        # Dual-purpose (info + fatal): show as an informational status row.
        self._screen.post_message(_PhaseRow(("", "", "", "", "", "", message.strip())))


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class TestScreen(Widget):
    DEFAULT_CSS = """
    TestScreen { height: 1fr; layout: vertical; }
    TestScreen #test-table { height: 1fr; border: solid $panel; }
    TestScreen #verdict { padding: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("OOM boundary check", id="test-title")
        yield Static("", id="test-precondition")
        yield Button("Run check", id="btn-run-test", disabled=True)
        table = DataTable(id="test-table")
        table.add_columns(
            "phase", "step", "prefill", "gen tok/s", "peak", "post", "status"
        )
        yield table
        yield Static("", id="verdict")
        yield Static(
            "[dim]Full multi-phase stress suite: run `python stress_test.py` "
            "(set QUICK=1 for a faster pass). This tab runs a quick subset and "
            "can take a few minutes at large context.[/dim]",
            id="test-footer",
        )

    def on_mount(self) -> None:
        self._running = False  # tracks whether a test run is in progress
        self._cancelled = False
        self._refresh_precondition()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _server(self):
        app = self.app
        return find_running(app._global_cfg, app._state_dir)

    def _refresh_precondition(self) -> None:
        server = self._server()
        try:
            pre = self.query_one("#test-precondition", Static)
            btn = self.query_one("#btn-run-test", Button)
        except Exception:
            return
        if server is None:
            pre.update("[yellow]Start a server on the Serve tab first.[/yellow]")
            btn.disabled = True
        else:
            pre.update(
                f"Target: {server.model_id} / {server.backend} "
                f"/ {server.mode} / port {server.port}"
            )
            btn.disabled = self._running  # keep disabled while a run is active

    # ── Button handler ───────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Run / Stop toggle."""
        if event.button.id != "btn-run-test":
            return
        if self._running:
            # User pressed "Stop"
            self._cancelled = True
            event.button.label = "Stopping…"
            event.button.disabled = True
        else:
            self._start_run(event.button)

    def _start_run(self, button: Button) -> None:
        server = self._server()
        if server is None:
            self._refresh_precondition()
            return
        self._cancelled = False
        self._running = True
        try:
            self.query_one("#test-table", DataTable).clear()
            self.query_one("#verdict", Static).update("[dim]Running…[/dim]")
        except Exception:
            pass
        button.label = "Stop"
        self.run_worker(
            lambda: self._run_worker(server),
            thread=True,
            exclusive=True,
            group="oom-test",
        )

    # ── Worker (runs on a thread — must only post messages) ──────────────────

    def _run_worker(self, server) -> None:
        reporter = TextualReporter(self)
        try:
            result = run_oom_check(
                server,
                reporter,
                global_cfg=self.app._global_cfg,
                cancel=lambda: self._cancelled,
            )
        except Exception as exc:
            result = OomTestResult(
                "FAIL", None, None, None, f"Test crashed: {exc!r}"
            )
        self.post_message(_Verdict(result))

    # ── Message handlers (run on UI event loop) ───────────────────────────────

    @on(_PhaseRow)
    def _on_phase_row(self, message: _PhaseRow) -> None:
        try:
            self.query_one("#test-table", DataTable).add_row(*message.cells)
        except Exception:
            pass

    @on(_Verdict)
    def _on_verdict(self, message: _Verdict) -> None:
        result = message.result
        if result.verdict.startswith("OK"):
            colour = "green"
        elif result.verdict == "WARN":
            colour = "yellow"
        else:
            colour = "red"
        try:
            self.query_one("#verdict", Static).update(
                f"[{colour} bold]{result.verdict}[/{colour} bold]  {result.detail}"
            )
            btn = self.query_one("#btn-run-test", Button)
            btn.label = "Run check"
            btn.disabled = False
        except Exception:
            pass
        self._running = False
        self._refresh_precondition()
