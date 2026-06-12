from __future__ import annotations

from typing import Any

import torch

from src.rasp.stage_probe import STAGES, StageProbeNet, classify_operational_stage, transform_stage_features


class RuntimeStageProbe:
    def __init__(self, checkpoint_path: str, reasoning_threshold: float) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.metadata = checkpoint["metadata"]
        self.transform = checkpoint["transform"]
        self.reasoning_threshold = float(reasoning_threshold)
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
        rule_stage = classify_operational_stage(recent_text, boundary_index, max(1, num_boundaries))
        if rule_stage == "verification":
            return {
                "operational_stage": "verification",
                "stage_source": "explicit_verification_rule",
                "reasoning_accepted": False,
                "stage_confidence": 1.0,
                "stage_probabilities": {stage: None for stage in STAGES},
            }
        row = {"position": position, "entropy": entropy, "confidence": confidence}
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
        return {
            "operational_stage": predicted_stage,
            "stage_source": "hidden_stage_probe",
            "reasoning_accepted": (
                predicted_stage == "reasoning"
                and reasoning_probability >= self.reasoning_threshold
            ),
            "stage_confidence": float(probabilities[predicted_index].item()),
            "stage_probabilities": {
                stage: float(probabilities[index].item()) for index, stage in enumerate(STAGES)
            },
        }

