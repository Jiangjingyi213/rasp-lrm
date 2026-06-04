from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any

import torch

from src.probes.action_conditioned_dataset import build_action_features
from src.probes.train_probe import LinearRiskProbe
from src.rasp.budget_controller import RuntimeObservation, conservative_ratio_cap


@dataclass
class ActionConditionedRiskController:
    checkpoint_path: str
    dataset: str
    ratios: list[float]
    runtime_layers: list[int]
    risk_threshold: float = 0.35
    target_average_ratio: float = 0.20
    max_new_tokens: int = 512
    window_tokens: int = 16
    module: str = "mlp_intermediate_channels"
    default_max_ratio: float | None = None
    early_tokens: int = 0
    early_max_ratio: float | None = None
    high_entropy_threshold: float | None = None
    high_entropy_max_ratio: float | None = None
    low_confidence_threshold: float | None = None
    low_confidence_max_ratio: float | None = None
    _model: LinearRiskProbe = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)
    _selected_ratios: list[float] = field(default_factory=list, init=False, repr=False)
    last_decision: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        checkpoint = torch.load(Path(self.checkpoint_path), map_location="cpu")
        self._metadata = dict(checkpoint.get("metadata", {}))
        self._model = LinearRiskProbe(int(checkpoint["dim"]))
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()
        if bool(self._metadata.get("include_stage", False)):
            raise ValueError("Runtime RASP-Zero v1 does not load stage-conditioned routers yet")
        trained_layer_dim = int(self._metadata.get("layer_dim", len(self.runtime_layers)))
        if trained_layer_dim != len(self.runtime_layers):
            raise ValueError(
                f"Router checkpoint expects {trained_layer_dim} runtime layers, got {len(self.runtime_layers)}"
            )
        trained_module = str(self._metadata.get("module", self.module))
        if trained_module != self.module:
            raise ValueError(f"Router checkpoint expects module={trained_module}, got {self.module}")
        self.ratios = sorted({0.0, *(float(ratio) for ratio in self.ratios)})
        if self.default_max_ratio is None:
            self.default_max_ratio = max(self.ratios)

    @property
    def total_decisions(self) -> int:
        return 1 + ceil(self.max_new_tokens / self.window_tokens)

    def reset(self) -> None:
        self._selected_ratios.clear()
        self.last_decision = None

    def _score(self, observation: RuntimeObservation, ratio: float) -> float:
        if observation.hidden_state is None:
            raise ValueError("Action-conditioned router requires hidden-state observations")
        features = build_action_features(
            torch.as_tensor(observation.hidden_state),
            entropy=observation.entropy,
            confidence=observation.confidence,
            position=min(1.0, observation.generated_tokens / max(1, self.max_new_tokens)),
            ratio=ratio,
            module=self.module,
            dataset=self.dataset,
            pruned_layers=self.runtime_layers,
            layer_dim=len(self.runtime_layers),
        )
        with torch.no_grad():
            return float(torch.sigmoid(self._model(features.unsqueeze(0))).item())

    def choose_ratio(self, observation: RuntimeObservation) -> float:
        if observation.generated_tokens == 0:
            self.reset()
        available_budget = max(
            0.0,
            self.target_average_ratio * (len(self._selected_ratios) + 1) - sum(self._selected_ratios),
        )
        scored = []
        selected_ratio = 0.0
        selected_risk = 0.0
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
        for ratio in sorted(self.ratios, reverse=True):
            if ratio <= 0.0:
                continue
            risk = self._score(observation, ratio)
            scored.append({"ratio": ratio, "predicted_risk": risk})
            if (
                selected_ratio <= 0.0
                and ratio <= available_budget + 1e-9
                and ratio <= ratio_cap + 1e-9
                and risk <= self.risk_threshold
            ):
                selected_ratio = ratio
                selected_risk = risk
        self._selected_ratios.append(selected_ratio)
        self.last_decision = {
            "predicted_risk": selected_risk,
            "risk_threshold": self.risk_threshold,
            "available_budget_before_selection": available_budget,
            "target_average_ratio": self.target_average_ratio,
            "ratio_cap": ratio_cap,
            "cap_reasons": cap_reasons,
            "candidate_scores": scored,
        }
        return selected_ratio
