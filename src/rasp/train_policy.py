from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader


POLICY_FEATURE_SCHEMA = "hidden_entropy_confidence_position_target_available_dataset_v2"


class RatioPolicyNet(nn.Module):
    def __init__(self, dim: int, num_ratios: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ratios),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RatioPolicyEval:
    val_loss: float
    oracle_match_accuracy: float
    unsafe_selection_rate: float
    average_selected_ratio: float
    average_oracle_ratio: float
    budget_utilization: float


def ratio_policy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    unsafe_mask: torch.Tensor,
    ratios: torch.Tensor,
    target_budget: torch.Tensor,
    *,
    unsafe_weight: float,
    budget_weight: float,
) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, labels)
    probs = torch.softmax(logits, dim=-1)
    unsafe_penalty = (probs * unsafe_mask).sum(dim=-1).mean()
    expected_ratio = (probs * ratios).sum(dim=-1)
    budget_penalty = (expected_ratio.mean() - target_budget.float().mean()).pow(2)
    return ce + float(unsafe_weight) * unsafe_penalty + float(budget_weight) * budget_penalty


@torch.no_grad()
def evaluate_ratio_policy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    unsafe_weight: float,
    budget_weight: float,
) -> RatioPolicyEval:
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    unsafe_selected = 0
    selected_ratios: list[float] = []
    oracle_ratios: list[float] = []
    target_budgets: list[float] = []
    for x, labels, unsafe_mask, ratios, target_budget, _indices in loader:
        x = x.to(device)
        labels = labels.to(device)
        unsafe_mask = unsafe_mask.to(device)
        ratios = ratios.to(device)
        target_budget = target_budget.to(device)
        logits = model(x)
        loss = ratio_policy_loss(
            logits,
            labels,
            unsafe_mask,
            ratios,
            target_budget,
            unsafe_weight=unsafe_weight,
            budget_weight=budget_weight,
        )
        batch = int(labels.shape[0])
        total_loss += float(loss.item()) * batch
        total += batch
        preds = torch.argmax(logits, dim=-1)
        labels_all.extend(int(value) for value in labels.cpu().tolist())
        preds_all.extend(int(value) for value in preds.cpu().tolist())
        selected = ratios.gather(1, preds.unsqueeze(1)).squeeze(1)
        oracle = ratios.gather(1, labels.unsqueeze(1)).squeeze(1)
        unsafe_selected += int(unsafe_mask.gather(1, preds.unsqueeze(1)).sum().item())
        selected_ratios.extend(float(value) for value in selected.cpu().tolist())
        oracle_ratios.extend(float(value) for value in oracle.cpu().tolist())
        target_budgets.extend(float(value) for value in target_budget.cpu().tolist())
    avg_selected = sum(selected_ratios) / len(selected_ratios) if selected_ratios else 0.0
    avg_oracle = sum(oracle_ratios) / len(oracle_ratios) if oracle_ratios else 0.0
    avg_budget = sum(target_budgets) / len(target_budgets) if target_budgets else 0.0
    return RatioPolicyEval(
        val_loss=total_loss / max(1, total),
        oracle_match_accuracy=float(accuracy_score(labels_all, preds_all)) if labels_all else 0.0,
        unsafe_selection_rate=unsafe_selected / max(1, total),
        average_selected_ratio=avg_selected,
        average_oracle_ratio=avg_oracle,
        budget_utilization=avg_selected / avg_budget if avg_budget > 0 else 0.0,
    )


def checkpoint_metadata(
    *,
    ratios: list[float],
    feature_set: str,
    best: dict[str, Any],
) -> dict[str, Any]:
    return {
        "feature_set": feature_set,
        "feature_schema": POLICY_FEATURE_SCHEMA,
        "ratios": [float(value) for value in ratios],
        "best": best,
    }
