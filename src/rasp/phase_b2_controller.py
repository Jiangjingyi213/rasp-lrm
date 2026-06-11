from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.rasp.budget_controller import RuntimeObservation, conservative_ratio_cap
from src.rasp.phase_b2 import PHASE_B2_SCHEMA, build_phase_b2_model, build_phase_b2_state_features
from src.rasp.safe_oracle import available_prefix_budget
from src.rasp.train_policy import budget_key


@dataclass
class PhaseB2UncertaintyController:
    checkpoint_path: str
    target_average_ratio: float = 0.20
    max_new_tokens: int = 512
    policy_horizon_tokens: int | None = None
    default_max_ratio: float | None = None
    early_tokens: int = 0
    early_max_ratio: float | None = None
    high_entropy_threshold: float | None = None
    high_entropy_max_ratio: float | None = None
    low_confidence_threshold: float | None = None
    low_confidence_max_ratio: float | None = None
    risk_threshold: float | None = None
    _model: nn.Module = field(init=False, repr=False)
    _ratios: list[float] = field(init=False, repr=False)
    _selected_ratios: list[float] = field(default_factory=list, init=False, repr=False)
    metadata: dict[str, Any] = field(init=False, repr=False)
    last_decision: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        checkpoint = torch.load(Path(self.checkpoint_path), map_location="cpu")
        self.metadata = dict(checkpoint.get("metadata", {}))
        if self.metadata.get("schema") != PHASE_B2_SCHEMA:
            raise ValueError("Phase B2 uncertainty checkpoint schema mismatch")
        if self.metadata.get("variant") != "uncertainty_flip_only":
            raise ValueError(
                "Online Phase B2 controller requires an uncertainty_flip_only checkpoint"
            )
        if self.metadata.get("feature_set") != "uncertainty" or int(self.metadata.get("dim", -1)) != 3:
            raise ValueError("Phase B2 uncertainty checkpoint has incompatible features")
        self._ratios = [float(value) for value in self.metadata["ratios"]]
        if not self._ratios or abs(self._ratios[0]) > 1e-12:
            raise ValueError("Phase B2 uncertainty checkpoint ratios must start with dense ratio 0")
        self._model = build_phase_b2_model(
            str(self.metadata["model_type"]),
            int(self.metadata["dim"]),
            int(self.metadata["hidden_dim"]),
        )
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()
        if self.risk_threshold is None:
            calibration = dict(self.metadata.get("calibration", {}))
            key = budget_key(self.target_average_ratio)
            if key not in calibration:
                raise ValueError(
                    f"Phase B2 checkpoint has no calibration for budget {key}; "
                    f"available budgets: {sorted(calibration)}"
                )
            self.risk_threshold = float(calibration[key]["threshold"])
        if self.default_max_ratio is None:
            self.default_max_ratio = max(self._ratios)

    def reset(self) -> None:
        self._selected_ratios.clear()
        self.last_decision = None

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        if observation.generated_tokens == 0:
            self.reset()
        available_budget = available_prefix_budget(self.target_average_ratio, self._selected_ratios)
        ratio_cap, cap_reasons = conservative_ratio_cap(
            observation,
            default_max_ratio=float(self.default_max_ratio),
            early_tokens=int(self.early_tokens),
            early_max_ratio=self.early_max_ratio,
            high_entropy_threshold=self.high_entropy_threshold,
            high_entropy_max_ratio=self.high_entropy_max_ratio,
            low_confidence_threshold=self.low_confidence_threshold,
            low_confidence_max_ratio=self.low_confidence_max_ratio,
        )
        if (
            self.policy_horizon_tokens is not None
            and observation.generated_tokens >= int(self.policy_horizon_tokens)
        ):
            ratio_cap = 0.0
            cap_reasons.append("outside_policy_horizon")
        features = build_phase_b2_state_features(
            torch.empty(0),
            {
                "entropy": float(observation.entropy),
                "confidence": float(observation.confidence),
                "position": min(1.0, observation.generated_tokens / max(1, self.max_new_tokens)),
            },
            "uncertainty",
        )
        with torch.no_grad():
            ratios = torch.tensor(self._ratios, dtype=torch.float32).unsqueeze(0)
            logits = self._model(features.unsqueeze(0), ratios)["flip_logits"].squeeze(0)
            raw_risks = torch.sigmoid(logits)
            raw_risks[0] = 0.0
            risks = torch.cummax(raw_risks, dim=0).values
        allowed = [
            index
            for index, ratio in enumerate(self._ratios)
            if ratio <= available_budget + 1e-9
            and ratio <= ratio_cap + 1e-9
            and (ratio <= 0.0 or float(risks[index].item()) <= float(self.risk_threshold))
        ]
        selected_index = max(allowed, key=lambda index: self._ratios[index]) if allowed else 0
        selected_ratio = self._ratios[selected_index]
        self._selected_ratios.append(selected_ratio)
        self.last_decision = {
            "available_budget_before_selection": available_budget,
            "target_average_ratio": self.target_average_ratio,
            "policy_horizon_tokens": self.policy_horizon_tokens,
            "ratio_cap": ratio_cap,
            "cap_reasons": cap_reasons,
            "risk_threshold": float(self.risk_threshold),
            "selected_predicted_risk": float(risks[selected_index].item()),
            "checkpoint_seed": self.metadata.get("seed"),
            "checkpoint_variant": self.metadata.get("variant"),
            "candidate_scores": [
                {
                    "ratio": ratio,
                    "raw_predicted_risk": float(raw_risks[index].item()),
                    "predicted_risk": float(risks[index].item()),
                    "allowed": index in allowed,
                }
                for index, ratio in enumerate(self._ratios)
            ],
        }
        return selected_ratio
