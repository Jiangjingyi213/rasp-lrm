from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RuntimeObservation:
    generated_tokens: int
    entropy: float
    confidence: float
    hidden_state: object | None = None


class RuntimeRatioController(Protocol):
    def choose_ratio(self, observation: RuntimeObservation) -> float:
        ...


def conservative_ratio_cap(
    observation: RuntimeObservation,
    *,
    default_max_ratio: float,
    early_tokens: int = 0,
    early_max_ratio: float | None = None,
    high_entropy_threshold: float | None = None,
    high_entropy_max_ratio: float | None = None,
    low_confidence_threshold: float | None = None,
    low_confidence_max_ratio: float | None = None,
) -> tuple[float, list[str]]:
    cap = float(default_max_ratio)
    reasons: list[str] = []
    if early_tokens > 0 and observation.generated_tokens < early_tokens and early_max_ratio is not None:
        cap = min(cap, float(early_max_ratio))
        reasons.append("early_tokens")
    if high_entropy_threshold is not None and high_entropy_max_ratio is not None:
        if float(observation.entropy) > float(high_entropy_threshold):
            cap = min(cap, float(high_entropy_max_ratio))
            reasons.append("high_entropy")
    if low_confidence_threshold is not None and low_confidence_max_ratio is not None:
        if float(observation.confidence) < float(low_confidence_threshold):
            cap = min(cap, float(low_confidence_max_ratio))
            reasons.append("low_confidence")
    return max(0.0, cap), reasons


@dataclass
class FixedRatioController:
    ratio: float = 0.0

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        del observation
        return float(self.ratio)


@dataclass
class ConfidenceThresholdController:
    """Simple history-only controller for plumbing checks, not the final router."""

    low_confidence: float = 0.50
    medium_confidence: float = 0.75
    low_ratio: float = 0.0
    medium_ratio: float = 0.05
    high_ratio: float = 0.10

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        if observation.confidence < self.low_confidence:
            return self.low_ratio
        if observation.confidence < self.medium_confidence:
            return self.medium_ratio
        return self.high_ratio
