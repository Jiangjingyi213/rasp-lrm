from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any

import torch

from src.probes.rasp_train_dataset import build_policy_features
from src.rasp.budget_controller import RuntimeObservation, conservative_ratio_cap
from src.rasp.safe_oracle import available_prefix_budget
from src.rasp.train_policy import POLICY_FEATURE_SCHEMA, ActionRiskPolicyNet


@dataclass
class RaspTrainPolicyController:
    checkpoint_path: str
    dataset: str
    target_average_ratio: float = 0.20
    max_new_tokens: int = 512
    window_tokens: int = 16
    default_max_ratio: float | None = None
    early_tokens: int = 0
    early_max_ratio: float | None = None
    high_entropy_threshold: float | None = None
    high_entropy_max_ratio: float | None = None
    low_confidence_threshold: float | None = None
    low_confidence_max_ratio: float | None = None
    risk_threshold: float | None = None
    _model: ActionRiskPolicyNet = field(init=False, repr=False)
    _ratios: list[float] = field(init=False, repr=False)
    _selected_ratios: list[float] = field(default_factory=list, init=False, repr=False)
    last_decision: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        checkpoint = torch.load(Path(self.checkpoint_path), map_location="cpu")
        metadata = dict(checkpoint.get("metadata", {}))
        if metadata.get("feature_schema") != POLICY_FEATURE_SCHEMA:
            raise ValueError(
                "RASP-Train checkpoint uses an incompatible feature schema; "
                "regenerate data and retrain the policy"
            )
        self._ratios = [float(value) for value in checkpoint["ratios"]]
        self._model = ActionRiskPolicyNet(
            int(checkpoint["dim"]),
            hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        )
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()
        if self.risk_threshold is None:
            self.risk_threshold = float(metadata["calibrated_threshold"])
        if self.default_max_ratio is None:
            self.default_max_ratio = max(self._ratios)

    @property
    def total_decisions(self) -> int:
        return 1 + ceil(self.max_new_tokens / self.window_tokens)

    def reset(self) -> None:
        self._selected_ratios.clear()
        self.last_decision = None

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        if observation.hidden_state is None:
            raise ValueError("RASP-Train policy requires hidden-state observations")
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
        features = build_policy_features(
            torch.as_tensor(observation.hidden_state).detach().float().cpu(),
            entropy=float(observation.entropy),
            confidence=float(observation.confidence),
            position=min(1.0, observation.generated_tokens / max(1, self.max_new_tokens)),
            target_budget=float(self.target_average_ratio),
            available_budget=available_budget,
            dataset=self.dataset,
        )
        with torch.no_grad():
            ratio_tensor = torch.tensor(self._ratios, dtype=torch.float32).unsqueeze(0)
            logits = self._model(features.unsqueeze(0), ratio_tensor).squeeze(0)
            probs = torch.sigmoid(logits)
        allowed = [
            index
            for index, ratio in enumerate(self._ratios)
            if float(ratio) <= available_budget + 1e-9
            and float(ratio) <= ratio_cap + 1e-9
            and (float(ratio) <= 0.0 or float(probs[index].item()) <= float(self.risk_threshold))
        ]
        if not allowed:
            selected_index = 0
        else:
            selected_index = max(allowed, key=lambda index: self._ratios[index])
        selected_ratio = float(self._ratios[selected_index])
        self._selected_ratios.append(selected_ratio)
        self.last_decision = {
            "available_budget_before_selection": available_budget,
            "target_average_ratio": self.target_average_ratio,
            "ratio_cap": ratio_cap,
            "cap_reasons": cap_reasons,
            "risk_threshold": float(self.risk_threshold),
            "selected_predicted_risk": float(probs[selected_index].item()),
            "candidate_scores": [
                {
                    "ratio": float(ratio),
                    "logit": float(logits[index].item()),
                    "predicted_risk": float(probs[index].item()),
                    "allowed": index in allowed,
                }
                for index, ratio in enumerate(self._ratios)
            ],
        }
        return selected_ratio
