from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.rasp.budget_controller import RuntimeObservation


ACTION_RISK_CHECKPOINT_SCHEMA = "rasp_action_risk_single_window_v1"
CONTEXT_FEATURE_NAMES = (
    "entropy",
    "confidence",
    "generated_tokens_over_max_new_tokens",
    "log1p_generated_tokens",
    "candidate_ratio",
    "candidate_ratio_squared",
)


def context_features(
    *,
    entropy: float,
    confidence: float,
    generated_tokens: int,
    max_new_tokens: int,
    ratio: float,
) -> np.ndarray:
    return np.asarray(
        [
            float(entropy),
            float(confidence),
            int(generated_tokens) / max(1, int(max_new_tokens)),
            float(np.log1p(int(generated_tokens))),
            float(ratio),
            float(ratio) ** 2,
        ],
        dtype=np.float64,
    )


def predict_context_risks(bundle: dict[str, Any], features: np.ndarray) -> np.ndarray:
    scaled = bundle["context_scaler"].transform(features)
    return bundle["context_model"].predict_proba(scaled)[:, 1]


def predict_hidden_risks(
    bundle: dict[str, Any], features: np.ndarray, hidden_state: np.ndarray
) -> np.ndarray:
    hidden = np.asarray(hidden_state, dtype=np.float64).reshape(1, -1)
    hidden = np.repeat(hidden, len(features), axis=0)
    hidden_scaled = bundle["hidden_scaler"].transform(hidden)
    hidden_pca = bundle["hidden_pca"].transform(hidden_scaled)
    context_scaled = bundle["context_scaler"].transform(features)
    model_input = np.concatenate([hidden_pca, context_scaled], axis=1)
    return bundle["hidden_model"].predict_proba(model_input)[:, 1]


def monotonic_risk_envelope(risks: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(risks, dtype=np.float64))


@dataclass
class ActionRiskSingleWindowController:
    checkpoint_path: str
    policy_variant: str
    operating_point: str
    max_new_tokens: int
    eligible_boundaries: tuple[int, ...] = (32, 96, 160)
    window_tokens: int = 16
    max_action_windows: int = 1
    runtime_ratios: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if int(self.max_action_windows) != 1:
            raise ValueError("Action-Risk pilot requires max_action_windows=1")
        if any(
            int(boundary) <= 0 or int(boundary) % int(self.window_tokens) != 0
            for boundary in self.eligible_boundaries
        ):
            raise ValueError("Action-Risk eligible boundaries must be positive and window-aligned")
        self.bundle = joblib.load(Path(self.checkpoint_path))
        if self.bundle.get("schema") != ACTION_RISK_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported Action-Risk checkpoint schema")
        if tuple(self.bundle.get("context_feature_names", ())) != CONTEXT_FEATURE_NAMES:
            raise ValueError("Action-Risk checkpoint feature schema mismatch")
        if int(self.bundle["max_new_tokens"]) != int(self.max_new_tokens):
            raise ValueError("Action-Risk checkpoint max_new_tokens mismatch")
        if int(self.bundle["window_tokens"]) != int(self.window_tokens):
            raise ValueError("Action-Risk checkpoint window_tokens mismatch")
        if tuple(self.bundle["eligible_boundaries"]) != tuple(self.eligible_boundaries):
            raise ValueError("Action-Risk checkpoint eligible_boundaries mismatch")
        if self.policy_variant not in {"context_only", "hidden_veto"}:
            raise ValueError(f"Unsupported Action-Risk policy variant: {self.policy_variant}")
        operating_points = self.bundle.get("operating_points", {})
        if self.operating_point not in operating_points:
            raise ValueError(f"Unavailable Action-Risk operating point: {self.operating_point}")
        point = operating_points[self.operating_point]
        if not point.get("eligible", False):
            raise ValueError(f"Action-Risk operating point is not eligible: {self.operating_point}")
        if self.policy_variant == "hidden_veto" and not point.get("hidden_veto_eligible", False):
            raise ValueError("Checkpoint does not admit hidden veto for this operating point")
        self.risk_threshold = float(point["context_risk_threshold"])
        self.hidden_veto_threshold = (
            float(point["hidden_veto_threshold"])
            if self.policy_variant == "hidden_veto"
            else None
        )
        self.ratios = tuple(float(value) for value in self.bundle["candidate_ratios"])
        if self.ratios != tuple(sorted(set(self.ratios))) or any(ratio <= 0.0 for ratio in self.ratios):
            raise ValueError("Action-Risk checkpoint candidate ratios must be sorted positive values")
        if self.runtime_ratios is not None:
            configured = tuple(sorted(set(float(value) for value in self.runtime_ratios)))
            expected = (0.0, *self.ratios)
            if configured != expected:
                raise ValueError("Action-Risk runtime ratio grid mismatch")
        self.last_decision: dict[str, Any] | None = None
        self.reset()

    def reset(self) -> None:
        self.action_executed = False
        self.last_decision = None

    def _dense(self, reason: str, observation: RuntimeObservation) -> float:
        self.last_decision = {
            "controller": "action_risk_single_window",
            "policy_variant": self.policy_variant,
            "operating_point": self.operating_point,
            "reason": reason,
            "window_activated": False,
            "generated_tokens": int(observation.generated_tokens),
        }
        return 0.0

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        if self.action_executed:
            return self._dense("maximum_action_windows_reached", observation)
        if int(observation.generated_tokens) not in self.eligible_boundaries:
            return self._dense("ineligible_boundary", observation)
        values = [observation.entropy, observation.confidence]
        if not all(np.isfinite(float(value)) for value in values):
            return self._dense("non_finite_uncertainty", observation)
        features = np.stack(
            [
                context_features(
                    entropy=observation.entropy,
                    confidence=observation.confidence,
                    generated_tokens=observation.generated_tokens,
                    max_new_tokens=self.max_new_tokens,
                    ratio=ratio,
                )
                for ratio in self.ratios
            ]
        )
        context_raw = predict_context_risks(self.bundle, features)
        context_envelope = monotonic_risk_envelope(context_raw)
        accepted = context_envelope <= self.risk_threshold
        hidden_raw = hidden_envelope = None
        if self.policy_variant == "hidden_veto":
            if observation.hidden_state is None:
                return self._dense("missing_hidden_state", observation)
            hidden = np.asarray(observation.hidden_state, dtype=np.float64).reshape(-1)
            if not np.isfinite(hidden).all():
                return self._dense("non_finite_hidden_state", observation)
            hidden_raw = predict_hidden_risks(self.bundle, features, hidden)
            hidden_envelope = monotonic_risk_envelope(hidden_raw)
            accepted &= hidden_envelope <= float(self.hidden_veto_threshold)
        accepted_indices = np.flatnonzero(accepted)
        selected = float(self.ratios[int(accepted_indices[-1])]) if len(accepted_indices) else 0.0
        self.action_executed = selected > 0.0
        self.last_decision = {
            "controller": "action_risk_single_window",
            "policy_variant": self.policy_variant,
            "operating_point": self.operating_point,
            "reason": "action_selected" if selected > 0.0 else "no_action_below_risk_threshold",
            "window_activated": selected > 0.0,
            "generated_tokens": int(observation.generated_tokens),
            "selected_ratio": selected,
            "candidate_ratios": list(self.ratios),
            "context_raw_risks": context_raw.tolist(),
            "context_monotonic_risks": context_envelope.tolist(),
            "context_risk_threshold": self.risk_threshold,
            "hidden_raw_risks": hidden_raw.tolist() if hidden_raw is not None else None,
            "hidden_monotonic_risks": hidden_envelope.tolist() if hidden_envelope is not None else None,
            "hidden_veto_threshold": self.hidden_veto_threshold,
        }
        return selected
