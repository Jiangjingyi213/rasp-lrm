from __future__ import annotations

from typing import Any

import torch
from torch import nn


PHASE_B25_SCHEMA = "rasp_phase_b25_hidden_increment_v1"
PHASE_B25_VARIANTS = {
    "uncertainty_nonlinear": "uncertainty",
    "hidden_pca_linear": "hidden",
    "hidden_pca_nonlinear": "hidden",
    "uncertainty_hidden_residual": "residual",
}


def raw_phase_b25_features(
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    uncertainty = torch.tensor(
        [
            [float(row.get("entropy", 0.0)), float(row.get("confidence", 0.0)), float(row.get("position", 0.0))]
            for row in rows
        ],
        dtype=torch.float32,
    )
    return uncertainty, hidden.float().flatten(start_dim=1)


def _fit_standardizer(values: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"mean": mean, "scale": scale}


def _standardize(values: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    return (values - state["mean"]) / state["scale"]


def fit_phase_b25_transform(
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    train_indices: list[int],
    *,
    pca_dim: int,
) -> dict[str, Any]:
    uncertainty, hidden_values = raw_phase_b25_features(rows, hidden)
    train = torch.tensor(train_indices, dtype=torch.long)
    uncertainty_state = _fit_standardizer(uncertainty[train])
    hidden_state = _fit_standardizer(hidden_values[train])
    standardized_hidden = _standardize(hidden_values[train], hidden_state)
    rank = min(int(pca_dim), standardized_hidden.shape[0] - 1, standardized_hidden.shape[1])
    if rank < 1:
        raise ValueError("Phase B2.5 PCA requires at least two train rows")
    _u, _s, projection = torch.pca_lowrank(standardized_hidden, q=rank, center=False)
    return {
        "uncertainty": uncertainty_state,
        "hidden": hidden_state,
        "hidden_projection": projection,
        "pca_dim": rank,
        "fit_split": "train",
        "fit_row_count": len(train_indices),
    }


def transform_phase_b25_features(
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    transform: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    uncertainty, hidden_values = raw_phase_b25_features(rows, hidden)
    uncertainty = _standardize(uncertainty, transform["uncertainty"])
    hidden_values = _standardize(hidden_values, transform["hidden"]) @ transform["hidden_projection"]
    return uncertainty, hidden_values


class PhaseB25ActionNet(nn.Module):
    def __init__(self, variant: str, hidden_pca_dim: int, model_dim: int = 32) -> None:
        super().__init__()
        if variant not in PHASE_B25_VARIANTS:
            raise ValueError(f"Unknown Phase B2.5 variant: {variant}")
        self.variant = variant
        if variant == "uncertainty_nonlinear":
            self.state_encoder = nn.Sequential(nn.Linear(3, model_dim), nn.GELU())
            self.action_head = nn.Sequential(
                nn.Linear(model_dim + 2, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
            )
        elif variant == "hidden_pca_linear":
            self.action_head = nn.Linear(hidden_pca_dim + 2, 1)
        elif variant == "hidden_pca_nonlinear":
            self.state_encoder = nn.Sequential(nn.Linear(hidden_pca_dim, model_dim), nn.GELU())
            self.action_head = nn.Sequential(
                nn.Linear(model_dim + 2, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
            )
        else:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(3 + 2, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
            )
            self.hidden_residual = nn.Sequential(
                nn.Linear(hidden_pca_dim + 2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, 1, bias=False),
            )

    def forward(
        self,
        uncertainty: torch.Tensor,
        hidden_pca: torch.Tensor,
        ratios: torch.Tensor,
    ) -> torch.Tensor:
        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0).expand(uncertainty.shape[0], -1)
        action = torch.stack([ratios, ratios.square()], dim=-1)
        if self.variant == "hidden_pca_linear":
            state = hidden_pca.unsqueeze(1).expand(-1, ratios.shape[1], -1)
            return self.action_head(torch.cat([state, action], dim=-1)).squeeze(-1)
        if self.variant == "uncertainty_hidden_residual":
            unc = uncertainty.unsqueeze(1).expand(-1, ratios.shape[1], -1)
            base = self.uncertainty_head(torch.cat([unc, action], dim=-1)).squeeze(-1)
            hidden = hidden_pca.unsqueeze(1).expand(-1, ratios.shape[1], -1)
            residual = self.hidden_residual(torch.cat([hidden, action], dim=-1)).squeeze(-1)
            return base + residual
        source = uncertainty if self.variant == "uncertainty_nonlinear" else hidden_pca
        state = self.state_encoder(source).unsqueeze(1).expand(-1, ratios.shape[1], -1)
        return self.action_head(torch.cat([state, action], dim=-1)).squeeze(-1)


@torch.no_grad()
def predict_phase_b25(
    model: nn.Module,
    uncertainty: torch.Tensor,
    hidden_pca: torch.Tensor,
    ratios: list[float],
    device: torch.device,
) -> list[list[float]]:
    model.eval()
    ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)
    risks = torch.sigmoid(model(uncertainty.to(device), hidden_pca.to(device), ratio_tensor)).cpu()
    risks[:, 0] = 0.0
    return torch.cummax(risks, dim=1).values.tolist()


def boundary_any_flip_metrics(
    rows: list[dict[str, Any]],
    risks: list[list[float]],
) -> tuple[list[int], list[float]]:
    labels = [int(any(bool(value) for value in row["candidate_flipped"][1:])) for row in rows]
    scores = [max(float(value) for value in row_risks[1:]) for row_risks in risks]
    return labels, scores
