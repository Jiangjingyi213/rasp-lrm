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
class FixedSingleWindowController:
    """Apply one fixed ratio at one causal decode boundary, then return dense."""

    boundary_tokens: int
    ratio: float
    window_tokens: int = 16

    def __post_init__(self) -> None:
        if self.boundary_tokens < 0:
            raise ValueError("boundary_tokens must be non-negative")
        if self.window_tokens < 1:
            raise ValueError("window_tokens must be positive")
        if self.boundary_tokens % self.window_tokens != 0:
            raise ValueError("boundary_tokens must align with runtime window_tokens")
        if not 0.0 <= float(self.ratio) < 1.0:
            raise ValueError("ratio must be in [0, 1)")
        self.last_decision: dict | None = None

    def reset(self) -> None:
        self.last_decision = None

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        active = int(observation.generated_tokens) == int(self.boundary_tokens)
        selected = float(self.ratio) if active else 0.0
        self.last_decision = {
            "controller": "fixed_single_window",
            "configured_boundary_tokens": int(self.boundary_tokens),
            "window_tokens": int(self.window_tokens),
            "window_activated": active,
        }
        return selected


@dataclass
class FixedMultiWindowController:
    """Apply fixed-ratio windows on a causal cadence with dense cooldowns."""

    ratio: float
    cadence_tokens: int
    max_windows: int
    decision_start: int = 32
    window_tokens: int = 16
    cooldown_tokens: int = 16

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ratio) < 1.0:
            raise ValueError("Fixed multi-window ratio must be in (0, 1)")
        if int(self.window_tokens) < 1:
            raise ValueError("window_tokens must be positive")
        if int(self.decision_start) <= 0 or int(self.decision_start) % int(self.window_tokens):
            raise ValueError("decision_start must be positive and window-aligned")
        if int(self.cadence_tokens) < int(self.window_tokens) + int(self.cooldown_tokens):
            raise ValueError("cadence_tokens must leave the configured dense cooldown")
        if int(self.cadence_tokens) % int(self.window_tokens):
            raise ValueError("cadence_tokens must be window-aligned")
        if int(self.max_windows) < 1:
            raise ValueError("max_windows must be positive")
        self.reset()

    def reset(self) -> None:
        self.action_count = 0
        self.last_action_boundary: int | None = None
        self.ratio_sum = 0.0
        self.action_history: list[dict] = []
        self.last_decision: dict | None = None

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        tokens = int(observation.generated_tokens)
        eligible = (
            tokens >= int(self.decision_start)
            and (tokens - int(self.decision_start)) % int(self.cadence_tokens) == 0
        )
        finite = all(
            value == value and abs(float(value)) != float("inf")
            for value in (observation.entropy, observation.confidence)
        )
        cooldown_ok = (
            self.last_action_boundary is None
            or tokens - self.last_action_boundary
            >= int(self.window_tokens) + int(self.cooldown_tokens)
        )
        activate = bool(
            finite and eligible and cooldown_ok and self.action_count < int(self.max_windows)
        )
        previous_distance = (
            None if self.last_action_boundary is None else tokens - self.last_action_boundary
        )
        if activate:
            self.action_count += 1
            self.last_action_boundary = tokens
            self.ratio_sum += float(self.ratio)
            self.action_history.append(
                {
                    "action_index": self.action_count,
                    "generated_tokens": tokens,
                    "duration_tokens": int(self.window_tokens),
                    "ratio": float(self.ratio),
                }
            )
        reason = "action_selected"
        if not finite:
            reason = "non_finite_observation"
        elif self.action_count >= int(self.max_windows) and not activate:
            reason = "maximum_action_windows_reached"
        elif not eligible:
            reason = "ineligible_boundary"
        elif not cooldown_ok:
            reason = "dense_cooldown"
        self.last_decision = {
            "controller": "fixed_multi_window",
            "reason": reason,
            "window_activated": activate,
            "generated_tokens": tokens,
            "action_index": self.action_count if activate else None,
            "action_count": self.action_count,
            "selected_ratio": float(self.ratio) if activate else 0.0,
            "previous_action_distance": previous_distance,
            "cumulative_action_ratio_mass": self.ratio_sum,
            "cumulative_pruned_token_ratio_mass": self.ratio_sum * int(self.window_tokens),
            "action_history": [dict(value) for value in self.action_history],
            "cadence_tokens": int(self.cadence_tokens),
            "max_windows": int(self.max_windows),
            "window_tokens": int(self.window_tokens),
            "cooldown_tokens": int(self.cooldown_tokens),
        }
        return float(self.ratio) if activate else 0.0


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
