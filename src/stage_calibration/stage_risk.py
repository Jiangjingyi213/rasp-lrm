from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


STAGE_RISK_CHECKPOINT_SCHEMA = "stage_risk_adaptive_controller_v1"
STAGE_RISK_FEATURE_NAMES = (
    "stage_local_tokens",
    "entropy",
    "confidence",
    "activation_ood",
    "activation_drift",
    "mask_score_margin",
    "candidate_ratio",
)


def stage_risk_features(
    *,
    stage_local_tokens: int,
    entropy: float,
    confidence: float,
    activation_ood: float,
    activation_drift: float,
    mask_score_margin: float,
    candidate_ratio: float,
) -> np.ndarray:
    """Causal features shared by offline OOF analysis and runtime inference."""

    return np.asarray(
        [
            float(stage_local_tokens),
            float(entropy),
            float(confidence),
            float(activation_ood),
            float(activation_drift),
            float(mask_score_margin),
            float(candidate_ratio),
        ],
        dtype=np.float64,
    )


@dataclass
class StageRiskController:
    checkpoint_path: str
    action_ratios: tuple[float, ...]
    risk_thresholds: dict[str, float]

    def __post_init__(self) -> None:
        self.bundle = joblib.load(Path(self.checkpoint_path))
        if self.bundle.get("schema") != STAGE_RISK_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported stage-risk controller checkpoint")
        if tuple(self.bundle.get("feature_names", ())) != STAGE_RISK_FEATURE_NAMES:
            raise ValueError("Stage-risk controller feature schema mismatch")
        if self.bundle.get("calibration") != "grouped_training_fold_platt":
            raise ValueError("Stage-risk controller must use grouped training-fold Platt calibration")
        if not bool(self.bundle.get("mechanism_gate_passed", False)):
            raise ValueError("Stage-risk mechanism gate did not pass; online controller is forbidden")
        trained_ratios = tuple(float(value) for value in self.bundle["action_ratios"])
        configured_ratios = tuple(value for value in sorted(set(self.action_ratios)) if value > 0.0)
        if configured_ratios != trained_ratios:
            raise ValueError("Stage-risk runtime ratios do not match checkpoint")
        self.stages = tuple(str(value) for value in self.bundle["stages"])

    def predict_risks(self, stage: str, feature_rows: np.ndarray) -> np.ndarray:
        if stage not in self.stages:
            raise ValueError(f"Checkpoint does not support stage {stage!r}")
        stage_one_hot = np.zeros((len(feature_rows), len(self.stages)), dtype=np.float64)
        stage_one_hot[:, self.stages.index(stage)] = 1.0
        # Explicit stage-feature interactions let the small model learn different
        # risk slopes without relying on an opaque hidden-state router.
        interactions = np.concatenate(
            [feature_rows * stage_one_hot[:, index : index + 1] for index in range(len(self.stages))],
            axis=1,
        )
        values = np.concatenate([feature_rows, stage_one_hot, interactions], axis=1)
        scaled = self.bundle["scaler"].transform(values)
        raw = self.bundle["model"].decision_function(scaled)
        return self.bundle["platt_model"].predict_proba(raw.reshape(-1, 1))[:, 1]

    def choose_ratio(
        self,
        *,
        stage: str,
        candidates: list[tuple[float, np.ndarray]],
        minimum_ratio: float = 0.0,
    ) -> tuple[float, dict[str, Any]]:
        if not candidates:
            return 0.0, {"reason": "no_candidates", "risks": {}}
        ratios = [float(ratio) for ratio, _features in candidates]
        features = np.stack([row for _ratio, row in candidates])
        risks = np.maximum.accumulate(self.predict_risks(stage, features))
        threshold = float(self.risk_thresholds[stage])
        accepted = [
            ratio
            for ratio, risk in zip(ratios, risks)
            if risk <= threshold and ratio + 1e-12 >= float(minimum_ratio)
        ]
        selected = max(accepted) if accepted else 0.0
        return selected, {
            "reason": "risk_accepted" if selected > 0.0 else "no_safe_ratio",
            "risks": {f"{ratio:.2f}": float(risk) for ratio, risk in zip(ratios, risks)},
            "risk_threshold": threshold,
            "minimum_ratio": float(minimum_ratio),
        }
