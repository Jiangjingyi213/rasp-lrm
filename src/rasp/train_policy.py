from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.probes.rasp_train_dataset import build_policy_features
from src.rasp.safe_oracle import available_prefix_budget


POLICY_FEATURE_SCHEMA = "hidden_entropy_confidence_position_dataset_v4_shared_action_risk"


def budget_key(value: float) -> str:
    return f"{float(value):.2f}"


def threshold_for_budget(metadata: dict[str, Any], target_budget: float) -> float:
    thresholds = dict(metadata.get("calibrated_thresholds", {}))
    key = budget_key(target_budget)
    if key not in thresholds:
        raise ValueError(
            f"RASP-Train checkpoint has no calibrated threshold for budget {key}; "
            f"available budgets: {sorted(thresholds)}"
        )
    return float(thresholds[key])


class ActionRiskPolicyNet(nn.Module):
    """Predict pruning risk for each candidate ratio from one reasoning state."""

    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_features: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(state_features)
        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0).expand(state.shape[0], -1)
        expanded = state.unsqueeze(1).expand(-1, ratios.shape[1], -1)
        action = torch.stack([ratios, ratios.square()], dim=-1)
        return self.action_head(torch.cat([expanded, action], dim=-1)).squeeze(-1)


@dataclass
class ActionRiskEval:
    val_loss: float
    roc_auc: float | None
    pr_auc: float | None
    monotonic_violation_rate: float


def action_risk_loss(
    logits: torch.Tensor,
    unsafe_mask: torch.Tensor,
    *,
    positive_weight: float,
    monotonic_weight: float,
    ranking_weight: float,
    ranking_margin: float = 0.20,
) -> torch.Tensor:
    pos_weight = torch.tensor(float(positive_weight), device=logits.device, dtype=logits.dtype)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, unsafe_mask, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    monotonic = torch.relu(probs[:, :-1] - probs[:, 1:]).mean()

    safe = unsafe_mask < 0.5
    unsafe = unsafe_mask >= 0.5
    has_pair = safe.any(dim=1) & unsafe.any(dim=1)
    if bool(has_pair.any()):
        safe_max = probs.masked_fill(~safe, float("-inf")).max(dim=1).values
        unsafe_min = probs.masked_fill(~unsafe, float("inf")).min(dim=1).values
        ranking = torch.relu(float(ranking_margin) + safe_max[has_pair] - unsafe_min[has_pair]).mean()
    else:
        ranking = logits.new_zeros(())
    return bce + float(monotonic_weight) * monotonic + float(ranking_weight) * ranking


@torch.no_grad()
def evaluate_action_risk(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    positive_weight: float,
    monotonic_weight: float,
    ranking_weight: float,
) -> ActionRiskEval:
    model.eval()
    total_loss = 0.0
    total = 0
    labels: list[int] = []
    scores: list[float] = []
    violations = 0
    comparisons = 0
    for x, unsafe_mask, _flipped_mask, ratios, _target_budget, _indices in loader:
        x = x.to(device)
        unsafe_mask = unsafe_mask.to(device)
        ratios = ratios.to(device)
        logits = model(x, ratios)
        loss = action_risk_loss(
            logits,
            unsafe_mask,
            positive_weight=positive_weight,
            monotonic_weight=monotonic_weight,
            ranking_weight=ranking_weight,
        )
        batch = int(x.shape[0])
        total_loss += float(loss.item()) * batch
        total += batch
        probs = torch.sigmoid(logits)
        nonzero = ratios > 0.0
        labels.extend(int(value) for value in unsafe_mask[nonzero].cpu().flatten().tolist())
        scores.extend(float(value) for value in probs[nonzero].cpu().flatten().tolist())
        violations += int((probs[:, :-1] > probs[:, 1:] + 1e-8).sum().item())
        comparisons += int(probs[:, :-1].numel())
    if len(set(labels)) < 2:
        roc_auc = pr_auc = None
    else:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
    return ActionRiskEval(
        val_loss=total_loss / max(1, total),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        monotonic_violation_rate=violations / max(1, comparisons),
    )


def _problem_sequences(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row.get("dataset") or "unknown"), str(row["id"]))].append(index)
    sequences = list(grouped.values())
    for indices in sequences:
        indices.sort(
            key=lambda index: (
                int(rows[index].get("segment_index", rows[index].get("segment_id", 0))),
                int(rows[index].get("segment_id", 0)),
            )
        )
    return sequences


@torch.no_grad()
def predict_action_risks(
    model: ActionRiskPolicyNet,
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    ratios: list[float],
    device: torch.device,
) -> list[list[float]]:
    model.eval()
    risks: list[list[float]] = [[] for _ in rows]
    ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=device).unsqueeze(0)
    for row_index, row in enumerate(rows):
        features = build_policy_features(
            hidden[row_index],
            entropy=float(row.get("entropy", 0.0)),
            confidence=float(row.get("confidence", 0.0)),
            position=float(row.get("position", 0.0)),
            dataset=str(row.get("dataset") or "unknown"),
        ).to(device)
        probabilities = torch.sigmoid(model(features.unsqueeze(0), ratio_tensor)).squeeze(0).cpu().tolist()
        risks[row_index] = [float(value) for value in probabilities]
    return risks


def causal_action_risk_indices_from_risks(
    rows: list[dict[str, Any]],
    risks: list[list[float]],
    ratios: list[float],
    threshold: float,
    *,
    target_budget: float | None = None,
    default_max_ratio: float | None = None,
) -> list[int]:
    if len(rows) != len(risks):
        raise ValueError("Rows and predicted risks must have matching lengths")
    selected = [0] * len(rows)
    max_ratio = max(ratios) if default_max_ratio is None else float(default_max_ratio)
    for sequence in _problem_sequences(rows):
        history: list[float] = []
        for row_index in sequence:
            row = rows[row_index]
            budget = float(row["target_budget"]) if target_budget is None else float(target_budget)
            available = available_prefix_budget(budget, history)
            probabilities = risks[row_index]
            allowed = [
                index
                for index, ratio in enumerate(ratios)
                if float(ratio) <= available + 1e-9
                and float(ratio) <= max_ratio + 1e-9
                and (float(ratio) <= 0.0 or float(probabilities[index]) <= float(threshold))
            ]
            selected_index = max(allowed, key=lambda index: ratios[index]) if allowed else 0
            selected[row_index] = selected_index
            history.append(float(ratios[selected_index]))
    return selected


@torch.no_grad()
def causal_action_risk_indices(
    model: ActionRiskPolicyNet,
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    ratios: list[float],
    threshold: float,
    device: torch.device,
    *,
    target_budget: float | None = None,
    default_max_ratio: float | None = None,
) -> tuple[list[int], list[list[float]]]:
    risks = predict_action_risks(model, rows, hidden, ratios, device)
    selected = causal_action_risk_indices_from_risks(
        rows,
        risks,
        ratios,
        threshold,
        target_budget=target_budget,
        default_max_ratio=default_max_ratio,
    )
    return selected, risks


def selection_metrics(
    selected: list[int],
    rows: list[dict[str, Any]],
    ratios: list[float],
    *,
    target_budget: float | None = None,
) -> dict[str, float]:
    chosen_ratios = [float(ratios[index]) for index in selected]
    flips = [bool(row["candidate_flipped"][index]) for row, index in zip(rows, selected)]
    unsafe = [
        bool(row.get("candidate_unsafe", row["candidate_flipped"])[index])
        for row, index in zip(rows, selected)
    ]
    target = (
        sum(float(row["target_budget"]) for row in rows) / max(1, len(rows))
        if target_budget is None
        else float(target_budget)
    )
    average = sum(chosen_ratios) / max(1, len(chosen_ratios))
    return {
        "average_selected_ratio": average,
        "budget_utilization": average / target if target > 0 else 0.0,
        "flip_rate": sum(int(value) for value in flips) / max(1, len(flips)),
        "conservative_unsafe_rate": sum(int(value) for value in unsafe) / max(1, len(unsafe)),
    }


def checkpoint_metadata(
    *,
    ratios: list[float],
    best: dict[str, Any],
    calibrated_thresholds: dict[str, float],
    split_problem_keys: dict[str, list[list[str]]],
) -> dict[str, Any]:
    return {
        "method": "rasp_train_v2_1_shared_action_risk",
        "feature_schema": POLICY_FEATURE_SCHEMA,
        "ratios": [float(value) for value in ratios],
        "calibrated_thresholds": {
            str(key): float(value) for key, value in calibrated_thresholds.items()
        },
        "split_problem_keys": split_problem_keys,
        "best": best,
    }
