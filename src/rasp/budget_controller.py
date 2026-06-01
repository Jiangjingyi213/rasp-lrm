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
