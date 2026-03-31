from __future__ import annotations

import time
import urllib.error

from .config import StressConfig
from .models import PhaseResult, PhaseSample, RuntimeInfo
from .monitoring import PeakVramSampler, VramMonitor
from .prompting import PromptBuilder
from .reporting import ConsoleReporter, fmt_vram
from .runtime import ContainerLogReader, ContainerRuntimeInspector
from .server import LlamaServerClient


class BasePhase:
    key = "base"

    def __init__(
        self,
        config: StressConfig,
        client: LlamaServerClient,
        prompt_builder: PromptBuilder,
        vram_monitor: VramMonitor,
        runtime_inspector: ContainerRuntimeInspector,
        runtime_info: RuntimeInfo,
        reporter: ConsoleReporter,
    ) -> None:
        self.config = config
        self.client = client
        self.prompt_builder = prompt_builder
        self.vram_monitor = vram_monitor
        self.runtime_inspector = runtime_inspector
        self.runtime_info = runtime_info
        self.reporter = reporter

    def open_log_reader(self) -> ContainerLogReader | None:
        return self.runtime_inspector.start_log_reader(self.runtime_info)

    def _log_excerpt(self, log_reader: ContainerLogReader | None) -> list[str]:
        if log_reader is None:
            return []
        lines = log_reader.dump_lines(self.config.log_lines)
        log_reader.stop()
        return lines

    def _sample_request(
        self,
        log_reader: ContainerLogReader | None,
        label: str,
        prompt: str,
        prompt_length_chars: int | None = None,
        system: str | None = None,
    ) -> PhaseSample:
        sampler = PeakVramSampler(self.vram_monitor).start()
        try:
            request = self.client.send_request(
                prompt=prompt,
                runtime_inspector=self.runtime_inspector,
                runtime_info=self.runtime_info,
                log_reader=log_reader,
                system=system,
            )
        except urllib.error.HTTPError as error:
            peak = sampler.stop()
            post = self.vram_monitor.read()
            body = error.read().decode(errors="replace")
            return PhaseSample(
                label=label,
                prompt_length_chars=prompt_length_chars,
                peak_vram_gb=peak,
                post_vram_gb=post,
                ok=False,
                status=f"FAIL HTTP {error.code}: {body[:50]}",
            )
        except Exception as error:
            peak = sampler.stop()
            post = self.vram_monitor.read()
            return PhaseSample(
                label=label,
                prompt_length_chars=prompt_length_chars,
                peak_vram_gb=peak,
                post_vram_gb=post,
                ok=False,
                status=f"FAIL {error}",
            )

        peak = sampler.stop()
        post = self.vram_monitor.read()
        return PhaseSample(
            label=label,
            prompt_length_chars=prompt_length_chars,
            request=request,
            peak_vram_gb=peak,
            post_vram_gb=post,
            ok=True,
            status="OK",
        )


class RampPhase(BasePhase):
    key = "ramp"

    def run(self, steps: list[int]) -> PhaseResult:
        result = PhaseResult(key=self.key, title="Ramp")
        self.reporter.start_phase(result)
        log_reader = self.open_log_reader()
        last_ok_tokens = 0

        for target in steps:
            prompt = self.prompt_builder.build(target)
            sample = self._sample_request(
                log_reader=log_reader,
                label=str(target),
                prompt=prompt,
                prompt_length_chars=len(prompt),
            )
            result.samples.append(sample)
            self.reporter.record_sample(self.key, sample, self.config.vram_warn_gb)
            if not sample.ok:
                result.success = False
                result.last_ok_tokens = last_ok_tokens
                result.summary = f"Last successful size: ~{last_ok_tokens:,} tokens"
                result.log_excerpt = self._log_excerpt(log_reader)
                return result
            last_ok_tokens = target
            time.sleep(1)

        result.last_ok_tokens = last_ok_tokens
        result.summary = f"Ramp passed. Last tested: ~{last_ok_tokens:,} tokens."
        if log_reader:
            log_reader.stop()
        return result


class SustainedPhase(BasePhase):
    key = "sustained"

    def run(self, last_ok_tokens: int) -> PhaseResult:
        title = f"Phase 2: Sustained — {self.config.sustained_rounds} rounds at ~{last_ok_tokens:,} tokens"
        result = PhaseResult(key=self.key, title=title)
        self.reporter.start_phase(result)
        log_reader = self.open_log_reader()
        prompt = self.prompt_builder.build(last_ok_tokens)

        for index in range(1, self.config.sustained_rounds + 1):
            sample = self._sample_request(log_reader, str(index), prompt)
            result.samples.append(sample)
            self.reporter.record_sample(self.key, sample, self.config.vram_warn_gb)
            if not sample.ok:
                result.success = False
                result.log_excerpt = self._log_excerpt(log_reader)
                return result
            time.sleep(1)

        result.summary = "Sustained load passed."
        if log_reader:
            log_reader.stop()
        return result


class ColdStartPhase(BasePhase):
    key = "cold-start"

    def run(self, last_ok_tokens: int) -> PhaseResult:
        title = f"Phase 3: Cold-start — {self.config.cold_rounds} rounds, fresh KV each time"
        result = PhaseResult(key=self.key, title=title)
        self.reporter.start_phase(result)
        self.reporter.error(f"  (leak threshold: {self.config.leak_threshold_gb} GB growth across rounds)")
        log_reader = self.open_log_reader()
        first_cold_vram: float | None = None

        for index in range(1, self.config.cold_rounds + 1):
            prompt = self.prompt_builder.build(last_ok_tokens, prefix=f"[cold-start round {index}]")
            sample = self._sample_request(log_reader, str(index), prompt)
            if sample.ok and sample.post_vram_gb is not None:
                if first_cold_vram is None:
                    first_cold_vram = sample.post_vram_gb
                growth = sample.post_vram_gb - first_cold_vram
                if growth > self.config.leak_threshold_gb:
                    sample.status = f"OK  LEAK +{growth:.2f} GB"
                if growth > self.config.leak_threshold_gb:
                    result.success = False
            result.samples.append(sample)
            self.reporter.record_sample(self.key, sample, self.config.vram_warn_gb)
            if not sample.ok:
                result.success = False
                result.log_excerpt = self._log_excerpt(log_reader)
                return result
            if first_cold_vram is not None and sample.post_vram_gb is not None:
                growth = sample.post_vram_gb - first_cold_vram
                if growth > self.config.leak_threshold_gb:
                    result.summary = (
                        f"Cold-start FAILED: VRAM grew {growth:.2f} GB "
                        f"(threshold {self.config.leak_threshold_gb} GB)"
                    )
                    result.log_excerpt = self._log_excerpt(log_reader)
                    return result
            time.sleep(1)

        result.summary = "Cold-start passed — no KV cache leak detected."
        if log_reader:
            log_reader.stop()
        return result


class DefragPhase(BasePhase):
    key = "defrag"

    def run(self, last_ok_tokens: int) -> PhaseResult:
        title = f"Phase 4: Defrag stress — {self.config.defrag_cycles} fill→evict cycles"
        result = PhaseResult(key=self.key, title=title)
        self.reporter.start_phase(result)
        log_reader = self.open_log_reader()
        first_fill_peak: float | None = None
        evict_system = "You are a different assistant with no memory of previous conversations."

        for cycle in range(1, self.config.defrag_cycles + 1):
            fill_prompt = self.prompt_builder.build(last_ok_tokens, prefix=f"[defrag cycle {cycle} fill]")
            fill_sample = self._sample_request(log_reader, f"{cycle}:fill", fill_prompt)
            if fill_sample.ok and fill_sample.peak_vram_gb is not None:
                if first_fill_peak is None:
                    first_fill_peak = fill_sample.peak_vram_gb
                drift = fill_sample.peak_vram_gb - first_fill_peak
                if drift > self.config.leak_threshold_gb:
                    fill_sample.status = f"OK  drift +{drift:.2f} GB"
            result.samples.append(fill_sample)
            self.reporter.record_sample(self.key, fill_sample, self.config.vram_warn_gb)
            if not fill_sample.ok:
                result.success = False
                result.log_excerpt = self._log_excerpt(log_reader)
                return result
            if first_fill_peak is not None and fill_sample.peak_vram_gb is not None:
                drift = fill_sample.peak_vram_gb - first_fill_peak
                if drift > self.config.leak_threshold_gb:
                    result.success = False
                    result.summary = (
                        f"Defrag FAILED: fill-peak drifted {drift:.2f} GB "
                        f"(threshold {self.config.leak_threshold_gb} GB)"
                    )
                    result.log_excerpt = self._log_excerpt(log_reader)
                    return result

            evict_sample = self._sample_request(
                log_reader=log_reader,
                label=f"{cycle}:evict",
                prompt=f"Respond with one word: ready. [cycle {cycle}]",
                system=evict_system,
            )
            result.samples.append(evict_sample)
            self.reporter.record_sample(self.key, evict_sample, self.config.vram_warn_gb)
            if not evict_sample.ok:
                result.success = False
                result.log_excerpt = self._log_excerpt(log_reader)
                return result
            time.sleep(1)

        result.summary = "Defrag stress passed — no peak VRAM drift across fill→evict cycles."
        if log_reader:
            log_reader.stop()
        return result


class BoundaryPhase(BasePhase):
    key = "boundary"

    def run(self, ctx_size: int) -> PhaseResult:
        title = "Phase 5: Boundary — request at ctx_size+1 must get clean HTTP 400"
        result = PhaseResult(key=self.key, title=title)
        self.reporter.start_phase(result)

        over_prompt = self.prompt_builder.build(ctx_size + self.config.max_tokens * 2)
        vram_before = self.vram_monitor.read()
        log_reader = self.open_log_reader()
        sampler = PeakVramSampler(self.vram_monitor).start()

        try:
            self.client.send_request(
                prompt=over_prompt,
                runtime_inspector=self.runtime_inspector,
                runtime_info=self.runtime_info,
                log_reader=log_reader,
            )
            sampler.stop()
            result.success = False
            result.summary = "Boundary test FAILED."
            result.details.append((
                "Request status",
                "FAIL: expected HTTP 400 but request succeeded — ctx_size may be misconfigured",
            ))
            result.log_excerpt = self._log_excerpt(log_reader)
            return result
        except urllib.error.HTTPError as error:
            peak = sampler.stop()
            vram_after = self.vram_monitor.read()
            body = error.read().decode(errors="replace")
            healthy = self.client.server_healthy()

            status_400 = "OK" if error.code == 400 else f"FAIL (got HTTP {error.code})"
            status_msg = "OK" if "exceeds" in body.lower() else f"FAIL (body: {body[:60]})"
            if vram_before is None or vram_after is None or abs(vram_after - vram_before) < 0.05:
                status_vram = f"OK (peak in-flight: {fmt_vram(peak, self.config.vram_warn_gb)})"
            else:
                status_vram = (
                    f"WARN ({vram_after:.2f} vs {vram_before:.2f} GB before, "
                    f"peak in-flight: {fmt_vram(peak, self.config.vram_warn_gb)})"
                )
            status_health = "OK" if healthy else "FAIL (server unreachable after reject)"

            result.details.extend([
                ("HTTP 400 received", status_400),
                ("Error message", status_msg),
                ("VRAM unchanged", status_vram),
                ("Server still alive", status_health),
            ])
            result.success = all(not status.startswith("FAIL") for _, status in result.details)
            result.summary = "" if result.success else "Boundary test FAILED."
        except Exception as error:
            sampler.stop()
            result.success = False
            result.summary = "Boundary test FAILED."
            result.details.append(("Unexpected exception", f"FAIL: {error}"))
            result.log_excerpt = self._log_excerpt(log_reader)
            return result

        if log_reader:
            log_reader.stop()
        return result
