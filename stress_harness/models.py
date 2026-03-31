from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import StressConfig


@dataclass(slots=True)
class RuntimeInfo:
    runtime: str | None
    container_id: str | None
    status_message: str
    vram_mode: str = "n/a"


@dataclass(slots=True)
class RequestMetrics:
    elapsed_s: float
    prompt_n: int | None
    prompt_per_second: float | None
    predicted_per_second: float | None
    response: dict[str, Any]

    @classmethod
    def from_response(cls, response: Any, elapsed_s: float) -> "RequestMetrics":
        response_dict = _coerce_mapping(response)
        timings = _extract_timings(response_dict)
        prompt_n = timings.get("prompt_n")
        return cls(
            elapsed_s=elapsed_s,
            prompt_n=prompt_n if isinstance(prompt_n, int) else None,
            prompt_per_second=_maybe_float(timings.get("prompt_per_second")),
            predicted_per_second=_maybe_float(timings.get("predicted_per_second")),
            response=response_dict,
        )

    def prefill_display(self) -> str:
        if self.prompt_per_second:
            prompt_n = self.prompt_n if self.prompt_n is not None else "?"
            return f"{prompt_n}t/{self.elapsed_s:.1f}s"
        return f"{self.elapsed_s:.1f}s"

    def gen_toks_display(self) -> str:
        if self.predicted_per_second is None:
            return "—"
        return f"{self.predicted_per_second:.1f}"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                return dict(item)
    return {}


def _extract_timings(response: Mapping[str, Any]) -> dict[str, Any]:
    timings = response.get("timings", {})
    if isinstance(timings, Mapping):
        return dict(timings)
    if isinstance(timings, list):
        for item in timings:
            if isinstance(item, Mapping):
                return dict(item)
    return {}


@dataclass(slots=True)
class PhaseSample:
    label: str
    prompt_length_chars: int | None = None
    request: RequestMetrics | None = None
    peak_vram_gb: float | None = None
    post_vram_gb: float | None = None
    ok: bool = True
    status: str = "OK"


@dataclass(slots=True)
class PhaseResult:
    key: str
    title: str
    samples: list[PhaseSample] = field(default_factory=list)
    success: bool = True
    summary: str = ""
    log_excerpt: list[str] = field(default_factory=list)
    details: list[tuple[str, str]] = field(default_factory=list)
    last_ok_tokens: int | None = None


@dataclass(slots=True)
class StressRunResult:
    config: "StressConfig"
    ctx_size: int
    steps: list[int]
    runtime: RuntimeInfo
    baseline_vram_gb: float | None
    phases: list[PhaseResult] = field(default_factory=list)
    success: bool = False
    final_vram_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
