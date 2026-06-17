from __future__ import annotations

from stress_harness.config import StressConfig
from stress_harness.models import PhaseSample, RuntimeInfo
from stress_harness.phases import SustainedPhase


class _Reporter:
    def start_phase(self, phase): pass
    def record_sample(self, phase_key, sample, warn_at): pass


class _Client:
    def __init__(self):
        self.calls = 0

    def send_request(self, **kwargs):
        self.calls += 1
        # Minimal request stand-in; phases only read it back via PhaseSample.
        return object()


class _Inspector:
    def start_log_reader(self, runtime_info):
        return None


class _PromptBuilder:
    def build(self, target, prefix=""):
        return "x"


class _Monitor:
    def read(self):
        return 1.0


def test_sustained_phase_stops_when_cancelled():
    """With cancel() True after 2 rounds, the phase must not run all 20 rounds."""
    config = StressConfig(sustained_rounds=20)
    client = _Client()
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] > 2   # allow 2 rounds, then cancel

    phase = SustainedPhase(
        config=config,
        client=client,
        prompt_builder=_PromptBuilder(),
        vram_monitor=_Monitor(),
        runtime_inspector=_Inspector(),
        runtime_info=RuntimeInfo(runtime=None, container_id=None, status_message="x"),
        reporter=_Reporter(),
        cancel=cancel,
    )
    phase.run(last_ok_tokens=1000)
    assert client.calls <= 2  # stopped early, not 20
