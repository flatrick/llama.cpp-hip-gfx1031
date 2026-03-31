from __future__ import annotations

import sys

from .models import PhaseResult, PhaseSample, StressRunResult


def fmt_vram(value: float | None, warn_at: float | None = None) -> str:
    if value is None:
        return "n/a"
    warn = " !" if warn_at is not None and value >= warn_at else ""
    return f"{value:.2f} GB{warn}"


class ConsoleReporter:
    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stdout
        self._current_phase: str | None = None

    def _write(self, text: str = "") -> None:
        print(text, file=self._stream)

    def error(self, message: str) -> None:
        self._write(message)

    def start_run(self, result: StressRunResult) -> None:
        config = result.config
        ctx_size_source = (
            "forced by CTX_SIZE=N"
            if config.ctx_size_override is not None
            else "auto-detected; override with CTX_SIZE=N"
        )
        self._write("=" * 70)
        self._write("  llama-server stress test")
        self._write("=" * 70)
        self._write(f"  API        : {config.api_url}")
        self._write(f"  CTX_SIZE   : {result.ctx_size:,}  ({ctx_size_source})")
        self._write(f"  Steps      : {result.steps}")
        self._write(f"  Max gen    : {config.max_tokens} tokens/request")
        self._write(f"  VRAM warn  : {config.vram_warn_gb} GB  (! marker)")
        self._write(f"  Hard timeout : {config.request_timeout}s/request")
        self._write(f"  Stall timeout: {config.stall_timeout}s without server progress")
        self._write(f"  Sustained  : {config.sustained_rounds} rounds")
        self._write(f"  Cold-start : {config.cold_rounds} rounds")
        self._write(f"  Defrag     : {config.defrag_cycles} fill→evict cycles")
        self._write()
        self._write(f"  {result.runtime.status_message}")
        self._write(f"  VRAM tracking: {result.runtime.vram_mode}")
        self._write()
        self._write(f"VRAM at baseline : {fmt_vram(result.baseline_vram_gb, config.vram_warn_gb)}")
        self._write()

    def start_phase(self, phase: PhaseResult) -> None:
        self._current_phase = phase.key
        if phase.key == "ramp":
            self._write(f"  {'~tokens':>8}  {'prompt len':>12}  {'prefill':>12}  "
                        f"{'gen tok/s':>10}  {'peak VRAM':>10}  {'post VRAM':>10}  status")
            self._write(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  ──────")
            return

        self._write()
        self._write("=" * 70)
        self._write(f"  {phase.title}")
        self._write("=" * 70)

        if phase.key in {"sustained", "cold-start"}:
            self._write(f"  {'round':>6}  {'prefill':>12}  {'gen tok/s':>10}  "
                        f"{'peak VRAM':>10}  {'post VRAM':>10}  status")
            self._write(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  ──────")
        elif phase.key == "defrag":
            self._write(f"  {'cycle':>6}  {'step':>6}  {'prefill':>12}  {'gen tok/s':>10}  "
                        f"{'peak VRAM':>10}  {'post VRAM':>10}  status")
            self._write(f"  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  ──────")

    def record_sample(self, phase_key: str, sample: PhaseSample, warn_at: float) -> None:
        request_display = sample.request.prefill_display() if sample.request else "—"
        gen_display = sample.request.gen_toks_display() if sample.request else "—"
        peak = fmt_vram(sample.peak_vram_gb, warn_at)
        post = fmt_vram(sample.post_vram_gb, warn_at)

        if phase_key == "ramp":
            prompt_len = f"{sample.prompt_length_chars:,}" if sample.prompt_length_chars is not None else "—"
            self._write(f"  {sample.label:>8}  {prompt_len:>12}  {request_display:>12}  {gen_display:>10}  "
                        f"{peak:>10}  {post:>10}  {sample.status}")
            return

        if phase_key in {"sustained", "cold-start"}:
            self._write(f"  {sample.label:>6}  {request_display:>12}  {gen_display:>10}  "
                        f"{peak:>10}  {post:>10}  {sample.status}")
            return

        if phase_key == "defrag":
            cycle, step = sample.label.split(":", 1)
            self._write(f"  {cycle:>6}  {step:>6}  {request_display:>12}  {gen_display:>10}  "
                        f"{peak:>10}  {post:>10}  {sample.status}")

    def finish_phase(self, phase: PhaseResult) -> None:
        if phase.log_excerpt:
            self._write()
            self._write(f"  --- last {len(phase.log_excerpt)} container log lines ---")
            for line in phase.log_excerpt:
                self._write(f"  {line}")

        for label, value in phase.details:
            self._write(f"  {label:<18}: {value}")

        if phase.summary:
            self._write()
            self._write(phase.summary)

    def finish_run(self, result: StressRunResult) -> None:
        if not result.success:
            return
        self._write()
        self._write("=" * 70)
        self._write("  ALL PHASES PASSED")
        self._write(f"  VRAM tracking : {result.runtime.vram_mode}")
        self._write(f"  Final VRAM    : {fmt_vram(result.final_vram_gb, result.config.vram_warn_gb)}")
        self._write("  Server is healthy and memory-safe at configured ctx-size.")
        self._write("=" * 70)
