from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from src.probes.rasp_train_dataset import DEFAULT_RATIOS, build_policy_features
from src.rasp.train_policy import causal_action_risk_indices_from_risks, selection_metrics
from src.utils.io import read_jsonl


FAIR_BENCHMARK_SCHEMA = "rasp_train_fair_benchmark_v1"
VARIANTS = {
    "ratio_only_linear": ("ratio_only", "linear"),
    "position_ratio_linear": ("position", "linear"),
    "uncertainty_ratio_linear": ("uncertainty", "linear"),
    "hidden_ratio_linear": ("hidden", "linear"),
    "hidden_ratio_nonlinear": ("hidden", "nonlinear"),
}
LABEL_FIELDS = {
    "flipped": "candidate_flipped",
    "unsafe": "candidate_unsafe",
}


def problem_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset") or "unknown"), str(row["id"])


def create_split_manifest(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    problems = sorted({problem_key(row) for row in rows})
    if len(problems) < 3:
        raise ValueError("Fair benchmark requires at least three problems")
    random.Random(seed).shuffle(problems)
    holdout_count = max(2, int(len(problems) * 0.30))
    holdout_count = min(holdout_count, len(problems) - 1)
    calibration_count = max(1, holdout_count // 2)
    split = {
        "calibration": problems[:calibration_count],
        "test": problems[calibration_count:holdout_count],
        "train": problems[holdout_count:],
    }
    return {
        "schema": FAIR_BENCHMARK_SCHEMA,
        "seed": int(seed),
        "split_problem_keys": {
            name: [list(key) for key in sorted(keys)]
            for name, keys in split.items()
        },
        "problem_counts": {name: len(keys) for name, keys in split.items()},
    }


def indices_for_split(rows: list[dict[str, Any]], manifest: dict[str, Any], split: str) -> list[int]:
    keys = {
        (str(dataset), str(problem_id))
        for dataset, problem_id in manifest["split_problem_keys"][split]
    }
    indices = [index for index, row in enumerate(rows) if problem_key(row) in keys]
    if not indices:
        raise ValueError(f"Manifest split {split!r} does not match any dataset rows")
    return indices


def validate_equivalent_rows(rows: list[dict[str, Any]], other_rows: list[dict[str, Any]]) -> None:
    def state_labels(source: list[dict[str, Any]]) -> dict[tuple[str, str, int], tuple[Any, Any]]:
        labels = {
            (*problem_key(row), int(row.get("segment_id", 0))): (
                row["candidate_flipped"],
                row.get("candidate_unsafe", row["candidate_flipped"]),
            )
            for row in source
        }
        if len(labels) != len(source):
            raise ValueError("Fair benchmark source contains duplicate state keys")
        return labels

    if state_labels(rows) != state_labels(other_rows):
        raise ValueError("B15 and B20 datasets do not contain equivalent state-action labels")


def build_fair_state_features(
    hidden_state: torch.Tensor,
    row: dict[str, Any],
    feature_set: str,
) -> torch.Tensor:
    if feature_set == "ratio_only":
        return torch.empty(0, dtype=torch.float32)
    if feature_set == "position":
        return torch.tensor([float(row.get("position", 0.0))], dtype=torch.float32)
    if feature_set == "uncertainty":
        return torch.tensor(
            [
                float(row.get("entropy", 0.0)),
                float(row.get("confidence", 0.0)),
                float(row.get("position", 0.0)),
            ],
            dtype=torch.float32,
        )
    if feature_set == "hidden":
        return build_policy_features(
            hidden_state,
            entropy=float(row.get("entropy", 0.0)),
            confidence=float(row.get("confidence", 0.0)),
            position=float(row.get("position", 0.0)),
            dataset=str(row.get("dataset") or "unknown"),
        )
    raise ValueError(f"Unknown fair benchmark feature set: {feature_set}")


class FairActionRiskDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        hidden_path: str | Path,
        *,
        feature_set: str,
        label_type: str,
        ratios: list[float] | None = None,
    ) -> None:
        self.rows = read_jsonl(jsonl_path)
        self.hidden = torch.load(hidden_path, map_location="cpu")
        self.ratios = [float(value) for value in (ratios or DEFAULT_RATIOS)]
        self.feature_set = feature_set
        self.label_field = LABEL_FIELDS[label_type]
        if len(self.rows) != len(self.hidden):
            raise ValueError("Fair benchmark rows and hidden states differ")
        for row in self.rows:
            for field in ("candidate_flipped", "candidate_unsafe"):
                values = row.get(field, row["candidate_flipped"])
                if len(values) != len(self.ratios):
                    raise ValueError(f"{field} length does not match fair benchmark ratios")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        features = build_fair_state_features(self.hidden[index], row, self.feature_set)
        labels = row.get(self.label_field, row["candidate_flipped"])
        return (
            features,
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(self.ratios, dtype=torch.float32),
            index,
        )


class LinearActionRiskNet(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim + 2, 1)

    def forward(self, state_features: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0).expand(state_features.shape[0], -1)
        expanded = state_features.unsqueeze(1).expand(-1, ratios.shape[1], -1)
        action = torch.stack([ratios, ratios.square()], dim=-1)
        return self.linear(torch.cat([expanded, action], dim=-1)).squeeze(-1)


class NonlinearActionRiskNet(nn.Module):
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


def build_model(model_type: str, dim: int, hidden_dim: int = 256) -> nn.Module:
    if model_type == "linear":
        return LinearActionRiskNet(dim)
    if model_type == "nonlinear":
        if dim <= 0:
            raise ValueError("Nonlinear model requires non-empty state features")
        return NonlinearActionRiskNet(dim, hidden_dim)
    raise ValueError(f"Unknown fair benchmark model type: {model_type}")


def monotonic_risk_envelope(risks: list[list[float]]) -> list[list[float]]:
    output = []
    for values in risks:
        running = float("-inf")
        row = []
        for value in values:
            running = max(running, float(value))
            row.append(running)
        output.append(row)
    return output


@torch.no_grad()
def predict_risks(
    model: nn.Module,
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    ratios: list[float],
    feature_set: str,
    device: torch.device,
) -> list[list[float]]:
    model.eval()
    ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=device).unsqueeze(0)
    output = []
    for index, row in enumerate(rows):
        features = build_fair_state_features(hidden[index], row, feature_set).to(device)
        probabilities = torch.sigmoid(model(features.unsqueeze(0), ratio_tensor)).squeeze(0)
        output.append([float(value) for value in probabilities.cpu().tolist()])
    return monotonic_risk_envelope(output)


def calibrate_thresholds(
    rows: list[dict[str, Any]],
    risks: list[list[float]],
    ratios: list[float],
    budgets: list[float],
    max_flip_rates: list[float],
    max_unsafe_rates: list[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for budget, max_flip, max_unsafe in zip(budgets, max_flip_rates, max_unsafe_rates):
        frontier = []
        for value in range(1, 100):
            threshold = value / 100.0
            selected = causal_action_risk_indices_from_risks(
                rows, risks, ratios, threshold, target_budget=budget
            )
            frontier.append(
                {
                    "threshold": threshold,
                    **selection_metrics(selected, rows, ratios, target_budget=budget),
                }
            )
        feasible = [
            row for row in frontier
            if row["flip_rate"] <= max_flip + 1e-12
            and row["conservative_unsafe_rate"] <= max_unsafe + 1e-12
        ]
        if feasible:
            selected_row = max(
                feasible,
                key=lambda row: (
                    row["average_selected_ratio"],
                    -row["conservative_unsafe_rate"],
                    -row["flip_rate"],
                ),
            )
        else:
            selected_row = min(
                frontier,
                key=lambda row: (
                    row["flip_rate"],
                    row["conservative_unsafe_rate"],
                    -row["average_selected_ratio"],
                ),
            )
        output[f"{budget:.2f}"] = {
            "threshold": selected_row["threshold"],
            "constraints_satisfied": bool(selected_row in feasible),
            "selected": selected_row,
            "frontier": frontier,
        }
    return output
