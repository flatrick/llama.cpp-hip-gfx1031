from __future__ import annotations

from .config import StressConfig
from .models import PhaseResult, RuntimeInfo, StressRunResult
from .monitoring import VramMonitor
from .phases import BoundaryPhase, ColdStartPhase, DefragPhase, RampPhase, SustainedPhase
from .prompting import PromptBuilder
from .reporting import ConsoleReporter
from .runtime import ContainerRuntimeInspector
from .server import LlamaServerClient


class StressTestRunner:
    def __init__(self, config: StressConfig, reporter: ConsoleReporter | None = None) -> None:
        self.config = config
        self.reporter = reporter or ConsoleReporter()
        self.client = LlamaServerClient(config)
        self.prompt_builder = PromptBuilder(config.filler_chunk, config.tokens_per_chunk)
        self.runtime_inspector = ContainerRuntimeInspector(config.api_url, log_lines=config.log_lines)

    def run(self) -> StressRunResult:
        if not self.client.server_healthy():
            self.reporter.error(f"ERROR: server not reachable at {self.config.api_url}")
            return StressRunResult(
                config=self.config,
                ctx_size=self.config.ctx_size_fallback,
                steps=[],
                runtime=RuntimeInfo(None, None, "Container logs: podman/docker not found in PATH"),
                baseline_vram_gb=None,
                success=False,
            )

        runtime_info = self.runtime_inspector.detect()
        vram_monitor = VramMonitor.create(self.runtime_inspector, runtime_info)
        ctx_size = (
            self.config.ctx_size_override
            if self.config.ctx_size_override is not None
            else self.client.server_ctx_size()
        )
        steps = self.config.build_steps(ctx_size)
        baseline_vram = vram_monitor.read()

        run_result = StressRunResult(
            config=self.config,
            ctx_size=ctx_size,
            steps=steps,
            runtime=runtime_info,
            baseline_vram_gb=baseline_vram,
        )
        self.reporter.start_run(run_result)

        phases: list[PhaseResult] = []

        ramp = RampPhase(
            config=self.config,
            client=self.client,
            prompt_builder=self.prompt_builder,
            vram_monitor=vram_monitor,
            runtime_inspector=self.runtime_inspector,
            runtime_info=runtime_info,
            reporter=self.reporter,
        ).run(steps)
        phases.append(ramp)
        self.reporter.finish_phase(ramp)
        if not ramp.success or not ramp.last_ok_tokens:
            run_result.phases = phases
            run_result.final_vram_gb = vram_monitor.read()
            return run_result

        sustained = SustainedPhase(
            config=self.config,
            client=self.client,
            prompt_builder=self.prompt_builder,
            vram_monitor=vram_monitor,
            runtime_inspector=self.runtime_inspector,
            runtime_info=runtime_info,
            reporter=self.reporter,
        ).run(ramp.last_ok_tokens)
        phases.append(sustained)
        self.reporter.finish_phase(sustained)
        if not sustained.success:
            run_result.phases = phases
            run_result.final_vram_gb = vram_monitor.read()
            return run_result

        cold_start = ColdStartPhase(
            config=self.config,
            client=self.client,
            prompt_builder=self.prompt_builder,
            vram_monitor=vram_monitor,
            runtime_inspector=self.runtime_inspector,
            runtime_info=runtime_info,
            reporter=self.reporter,
        ).run(ramp.last_ok_tokens)
        phases.append(cold_start)
        self.reporter.finish_phase(cold_start)
        if not cold_start.success:
            run_result.phases = phases
            run_result.final_vram_gb = vram_monitor.read()
            return run_result

        defrag = DefragPhase(
            config=self.config,
            client=self.client,
            prompt_builder=self.prompt_builder,
            vram_monitor=vram_monitor,
            runtime_inspector=self.runtime_inspector,
            runtime_info=runtime_info,
            reporter=self.reporter,
        ).run(ramp.last_ok_tokens)
        phases.append(defrag)
        self.reporter.finish_phase(defrag)
        if not defrag.success:
            run_result.phases = phases
            run_result.final_vram_gb = vram_monitor.read()
            return run_result

        boundary = BoundaryPhase(
            config=self.config,
            client=self.client,
            prompt_builder=self.prompt_builder,
            vram_monitor=vram_monitor,
            runtime_inspector=self.runtime_inspector,
            runtime_info=runtime_info,
            reporter=self.reporter,
        ).run(ctx_size)
        phases.append(boundary)
        self.reporter.finish_phase(boundary)

        run_result.phases = phases
        run_result.final_vram_gb = vram_monitor.read()
        run_result.success = all(phase.success for phase in phases)
        self.reporter.finish_run(run_result)
        return run_result
