"""Test tab — quick OOM boundary check against the running server."""
from __future__ import annotations

import threading

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, DataTable, Label, Static

from llamactl.core.config import GlobalConfig
from llamactl.core.lifecycle import ServerInfo, find_running
from llamactl.core.monitor import read_server_vram_kib
from llamactl.core.oomtest import OomTestResult, run_oom_check
from llamactl.core.runtime import find_runtime
from llamactl.ui.screens.serve import _VramGauge


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


class _Progress(Message):
    """Current phase/round line for the progress indicator."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class _PeakStat(Message):
    """A peak-VRAM observation (gb) with the budget (gb) for headroom display."""

    def __init__(self, peak_gb: float | None, budget_gb: float | None) -> None:
        self.peak_gb = peak_gb
        self.budget_gb = budget_gb
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
        self._screen.post_message(_Progress(phase.title))

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
        self._screen.post_message(_Progress(f"{phase_key} — {sample.label}"))
        self._screen.post_message(_PeakStat(sample.peak_vram_gb, warn_at))

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # NB: do NOT name this `_running` — that collides with Textual's
        # MessagePump._running (set True once the widget's loop starts), which
        # would silently force this flag True after mount.
        self._test_active = False      # whether a test run is in progress
        self._cancel_evt = threading.Event()  # cross-thread cancel signal
        self._vram_timer = None
        self._peak_gb: float | None = None

    def compose(self) -> ComposeResult:
        yield Label("OOM boundary check", id="test-title")
        yield Static("", id="test-precondition")
        yield _VramGauge(id="test-vram")
        yield Checkbox("Quick mode (faster, fewer rounds)", value=True, id="quick-mode")
        yield Button("Run check", id="btn-run-test", disabled=True)
        table = DataTable(id="test-table")
        table.add_columns(
            "phase", "step", "prefill", "gen tok/s", "peak", "post", "status"
        )
        yield table
        yield Static("", id="verdict")
        yield Static("", id="test-progress")
        yield Static("", id="test-peak")
        yield Static(
            "[dim]Quick mode runs a reduced subset; uncheck it for a fuller pass "
            "(more sustained/cold/defrag rounds — takes longer). For the complete "
            "multi-phase suite run `python stress_test.py`.[/dim]",
            id="test-footer",
        )

    def on_mount(self) -> None:
        self._refresh_precondition()

    def on_unmount(self) -> None:
        # Stop any in-flight run: set the cancel signal so the worker won't start
        # another phase, and cancel the worker group. (Textual cannot interrupt an
        # in-flight HTTP request mid-phase, but this prevents the thread leak /
        # delayed-exit when the app quits during a run.)
        # cancel_group is a no-op (returns []) when the group is empty, so no
        # guard is needed — and swallowing errors here would hide real bugs.
        self._cancel_evt.set()
        self.workers.cancel_group(self, "oom-test")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _server(self) -> ServerInfo | None:
        app = self.app
        return find_running(app._global_cfg, app._state_dir)

    def _refresh_precondition(self) -> None:
        server = self._server()
        try:
            pre = self.query_one("#test-precondition", Static)
            btn = self.query_one("#btn-run-test", Button)
        except NoMatches:
            return
        if server is None:
            pre.update("[yellow]Start a server on the Serve tab first.[/yellow]")
            btn.disabled = True
        else:
            pre.update(
                f"Target: {server.model_id} / {server.backend} "
                f"/ {server.mode} / port {server.port}"
            )
            btn.disabled = self._test_active  # keep disabled while a run is active
            try:
                self.query_one("#quick-mode", Checkbox).disabled = self._test_active
            except NoMatches:
                pass

    def _poll_test_vram(self) -> None:
        """Update the live VRAM gauge while a run is active. UI-thread only."""
        if not self._test_active:
            return
        server = self._server()
        if server is None:
            return
        rt = find_runtime() if server.mode == "container" else None
        kib = read_server_vram_kib(server, rt)
        try:
            gauge = self.query_one("#test-vram", _VramGauge)
            gauge.budget_kib = int(self.app._global_cfg.vram_budget_gb * 1024 * 1024)
            gauge.vram_kib = kib
        except NoMatches:
            pass

    # ── Button handler ───────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Run / Stop toggle."""
        if event.button.id != "btn-run-test":
            return
        if self._test_active:
            # User pressed "Stop"
            self._cancel_evt.set()
            event.button.label = "Stopping…"
            event.button.disabled = True
        else:
            self._start_run(event.button)

    def _start_run(self, button: Button) -> None:
        server = self._server()
        if server is None:
            self._refresh_precondition()
            return
        self._cancel_evt.clear()
        self._test_active = True
        if self._vram_timer is None:
            self._vram_timer = self.set_interval(2.0, self._poll_test_vram)
        self._poll_test_vram()
        # Capture app state on the UI thread; never read self.app from the worker.
        global_cfg = self.app._global_cfg
        self._peak_gb = None
        try:
            self.query_one("#test-table", DataTable).clear()
            self.query_one("#verdict", Static).update("[dim]Running…[/dim]")
        except NoMatches:
            pass
        try:
            self.query_one("#test-progress", Static).update("")
            self.query_one("#test-peak", Static).update("")
        except NoMatches:
            pass
        button.label = "Stop"
        quick = True
        try:
            quick = self.query_one("#quick-mode", Checkbox).value
        except NoMatches:
            pass
        self.query_one("#quick-mode", Checkbox).disabled = True
        self.run_worker(
            lambda: self._run_worker(server, global_cfg, quick),
            thread=True,
            exclusive=True,
            group="oom-test",
        )

    # ── Worker (runs on a thread — must only post messages) ──────────────────

    def _run_worker(self, server: ServerInfo, global_cfg: GlobalConfig, quick: bool) -> None:
        reporter = TextualReporter(self)
        try:
            result = run_oom_check(
                server,
                reporter,
                global_cfg=global_cfg,
                cancel=self._cancel_evt.is_set,
                quick=quick,
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
        except NoMatches:
            pass

    @on(_Progress)
    def _on_progress(self, message: _Progress) -> None:
        try:
            self.query_one("#test-progress", Static).update(message.text)
        except NoMatches:
            pass

    @on(_PeakStat)
    def _on_peak(self, message: _PeakStat) -> None:
        if message.peak_gb is not None:
            if self._peak_gb is None or message.peak_gb > self._peak_gb:
                self._peak_gb = message.peak_gb
        if self._peak_gb is None:
            return
        budget = message.budget_gb
        try:
            label = self.query_one("#test-peak", Static)
        except NoMatches:
            return
        if budget is not None:
            headroom = budget - self._peak_gb
            label.update(f"Peak {self._peak_gb:.1f} / budget {budget:.1f} GiB (headroom {headroom:.1f})")
        else:
            label.update(f"Peak {self._peak_gb:.1f} GiB")

    @on(_Verdict)
    def _on_verdict(self, message: _Verdict) -> None:
        result = message.result
        if result.verdict.startswith("OK"):
            colour = "green"
        elif result.verdict in ("WARN", "STOPPED"):
            colour = "yellow"
        else:
            colour = "red"
        try:
            self.query_one("#verdict", Static).update(
                f"[{colour} bold]{result.verdict}[/{colour} bold]  {escape(result.detail)}"
            )
        except NoMatches:
            pass
        try:
            btn = self.query_one("#btn-run-test", Button)
            btn.label = "Run check"
            btn.disabled = False
        except NoMatches:
            pass
        self._test_active = False
        if self._vram_timer is not None:
            self._vram_timer.stop()
            self._vram_timer = None
        try:
            self.query_one("#test-vram", _VramGauge).vram_kib = 0
        except NoMatches:
            pass
        try:
            self.query_one("#quick-mode", Checkbox).disabled = False
        except NoMatches:
            pass
        self._refresh_precondition()
