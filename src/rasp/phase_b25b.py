from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.rasp.phase_b2 import PHASE_B2_SCHEMA, build_phase_b2_model, build_phase_b2_state_features
from src.rasp.fair_benchmark import monotonic_risk_envelope


PHASE_B25B_SCHEMA = "rasp_phase_b25b_frozen_uncertainty_residual_v1"


class HiddenActionResidual(nn.Module):
    def __init__(self, hidden_pca_dim: int, model_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_pca_dim + 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1, bias=False),
        )
        nn.init.zeros_(self.network[-1].weight)

    def forward(self, hidden_pca: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0).expand(hidden_pca.shape[0], -1)
        hidden = hidden_pca.unsqueeze(1).expand(-1, ratios.shape[1], -1)
        action = torch.stack([ratios, ratios.square()], dim=-1)
        return self.network(torch.cat([hidden, action], dim=-1)).squeeze(-1)


@torch.no_grad()
def frozen_uncertainty_logits(
    checkpoint: dict[str, Any],
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    metadata = checkpoint["metadata"]
    if (
        metadata.get("schema") != PHASE_B2_SCHEMA
        or metadata.get("variant") != "uncertainty_flip_only"
        or metadata.get("feature_set") != "uncertainty"
    ):
        raise ValueError("Phase B2.5b requires a v3 uncertainty_flip_only checkpoint")
    model = build_phase_b2_model(metadata["model_type"], metadata["dim"], metadata["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ratios = torch.tensor(metadata["ratios"], dtype=torch.float32, device=device)
    logits = []
    for index, row in enumerate(rows):
        features = build_phase_b2_state_features(hidden[index], row, "uncertainty").to(device)
        logits.append(model(features.unsqueeze(0), ratios)["flip_logits"].squeeze(0).cpu())
    return torch.stack(logits)


def combined_risks(
    base_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    alpha: float,
) -> list[list[float]]:
    risks = torch.sigmoid(base_logits + float(alpha) * residual_logits).clone()
    risks[:, 0] = 0.0
    return monotonic_risk_envelope(risks.tolist())
