from __future__ import annotations

from typing import Any

import torch

from src.rasp.stage_probe import STAGES, StageProbeNet, classify_operational_stage, transform_stage_features


class RuntimeStageProbe:
    def __init__(
        self,
        checkpoint_path: str,
        reasoning_threshold: float,
        confidence_threshold: float | None = None,
        require_causal_features: bool = False,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.metadata = checkpoint["metadata"]
        self.transform = checkpoint["transform"]
        self.reasoning_threshold = float(reasoning_threshold)
        self.confidence_threshold = float(
            reasoning_threshold if confidence_threshold is None else confidence_threshold
        )
        self.variant = str(self.metadata["variant"])
        self.require_causal_features = bool(require_causal_features)
        causal_variants = {
            "uncertainty_only",
            "hidden_pca_linear",
            "hidden_pca_nonlinear",
            "hidden_uncertainty",
        }
        self.uses_future_position = self.variant not in causal_variants
        if self.require_causal_features and self.uses_future_position:
            raise ValueError(
                f"Stage checkpoint variant {self.variant!r} is not approved for causal runtime"
            )
        self.model = StageProbeNet(
            self.metadata["dim"],
            self.metadata["model_type"],
            self.metadata["model_dim"],
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @torch.no_grad()
    def classify(
        self,
        *,
        hidden_state: torch.Tensor,
        entropy: float,
        confidence: float,
        position: float,
        recent_text: str,
        boundary_index: int,
        num_boundaries: int,
    ) -> dict[str, Any]:
        rule_stage = classify_operational_stage(
            recent_text,
            boundary_index,
            (
                max(1, boundary_index + 2)
                if self.require_causal_features
                else max(1, num_boundaries)
            ),
        )
        if rule_stage == "verification":
            return {
                "operational_stage": "verification",
                "trusted_stage": "explicit_verification",
                "stage_source": "explicit_verification_rule",
                "reasoning_accepted": False,
                "stage_confidence": 1.0,
                "stage_probabilities": {stage: None for stage in STAGES},
                "stage_probe_variant": self.variant,
                "stage_probe_causal": (
                    self.require_causal_features and not self.uses_future_position
                ),
            }
        row = {
            "position": 0.0 if self.require_causal_features else position,
            "entropy": entropy,
            "confidence": confidence,
        }
        features = transform_stage_features(
            [row],
            hidden_state.detach().float().cpu().reshape(1, -1),
            self.transform,
            self.metadata["variant"],
        )
        probabilities = torch.softmax(self.model(features), dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        predicted_stage = STAGES[predicted_index]
        reasoning_probability = float(probabilities[STAGES.index("reasoning")].item())
        confidence = float(probabilities[predicted_index].item())
        trusted_stage = "unknown"
        if confidence >= self.confidence_threshold:
            if predicted_stage == "reasoning":
                if reasoning_probability >= self.reasoning_threshold:
                    trusted_stage = "accepted_reasoning"
            else:
                trusted_stage = {
                    "setup": "confident_setup",
                    "final": "confident_final",
                }[predicted_stage]
        return {
            "operational_stage": predicted_stage,
            "trusted_stage": trusted_stage,
            "stage_source": "hidden_stage_probe",
            "reasoning_accepted": (
                predicted_stage == "reasoning"
                and reasoning_probability >= self.reasoning_threshold
            ),
            "stage_confidence": confidence,
            "stage_probabilities": {
                stage: float(probabilities[index].item()) for index, stage in enumerate(STAGES)
            },
            "stage_probe_variant": self.variant,
            "stage_probe_causal": (
                self.require_causal_features and not self.uses_future_position
            ),
        }
