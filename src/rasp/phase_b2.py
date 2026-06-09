from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from src.probes.rasp_train_dataset import DEFAULT_RATIOS, build_policy_features
from src.rasp.fair_benchmark import monotonic_risk_envelope, problem_key
from src.rasp.train_policy import causal_action_risk_indices_from_risks, selection_metrics
from src.utils.io import read_jsonl


PHASE_B2_SCHEMA = "rasp_phase_b2_multitask_v3"
PHASE_B2_VARIANTS = {
    "hidden_multitask": ("hidden", "nonlinear", True),
    "hidden_flip_only": ("hidden", "nonlinear", False),
    "hidden_flip_linear": ("hidden", "linear", False),
    "uncertainty_multitask": ("uncertainty", "nonlinear", True),
    "uncertainty_flip_only": ("uncertainty", "nonlinear", False),
    "uncertainty_flip_linear": ("uncertainty", "linear", False),
    "position_flip_only": ("position", "nonlinear", False),
    "position_flip_linear": ("position", "linear", False),
    "ratio_only_flip_only": ("ratio_only", "nonlinear", False),
    "ratio_only_flip_linear": ("ratio_only", "linear", False),
}


def boundary_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (*problem_key(row), int(row["boundary_index"]))


def problem_risk_strata(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    by_problem: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[problem_key(row)].append(row)
    positive_by_dataset: dict[str, list[tuple[float, tuple[str, str]]]] = defaultdict(list)
    output: dict[tuple[str, str], str] = {}
    for key, values in by_problem.items():
        labels = [bool(label) for row in values for label in row["candidate_flipped"][1:]]
        rate = sum(int(label) for label in labels) / max(1, len(labels))
        if rate <= 0.0:
            output[key] = "zero"
        else:
            positive_by_dataset[key[0]].append((rate, key))
    for dataset, values in positive_by_dataset.items():
        ordered = sorted(values, key=lambda item: (item[0], item[1]))
        if len(ordered) < 8:
            for _rate, key in ordered:
                output[key] = "positive"
            continue
        midpoint = len(ordered) // 2
        for index, (_rate, key) in enumerate(ordered):
            output[key] = "positive_low" if index < midpoint else "positive_high"
    return output


def build_phase_b2_state_features(
    hidden_state: torch.Tensor,
    row: dict[str, Any],
    feature_set: str,
) -> torch.Tensor:
    if feature_set == "ratio_only":
        return torch.tensor([0.0], dtype=torch.float32)
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
    raise ValueError(f"Unknown Phase B2 feature set: {feature_set}")


def stratified_problem_split(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_problem: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[problem_key(row)].append(row)
    risk_strata = problem_risk_strata(rows)
    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key in by_problem:
        strata[(key[0], risk_strata[key])].append(key)
    split = {"train": [], "validation": [], "calibration": [], "test": []}
    rng = random.Random(seed)
    for keys in strata.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        count = len(keys)
        if count < 4:
            raise ValueError(
                "Phase B2 four-way stratified split requires at least four problems "
                f"per dataset/positive stratum, got {count}"
            )
        validation_count = max(1, int(round(count * 0.10)))
        calibration_count = max(1, int(round(count * 0.15)))
        test_count = max(1, int(round(count * 0.15)))
        holdout_count = validation_count + calibration_count + test_count
        while holdout_count >= count:
            largest = max(
                ("validation", validation_count),
                ("calibration", calibration_count),
                ("test", test_count),
                key=lambda item: item[1],
            )[0]
            if largest == "validation" and validation_count > 1:
                validation_count -= 1
            elif largest == "calibration" and calibration_count > 1:
                calibration_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError(f"Cannot create non-empty Phase B2 train split for stratum of size {count}")
            holdout_count = validation_count + calibration_count + test_count
        validation_end = validation_count
        calibration_end = validation_end + calibration_count
        test_end = calibration_end + test_count
        split["validation"].extend(keys[:validation_end])
        split["calibration"].extend(keys[validation_end:calibration_end])
        split["test"].extend(keys[calibration_end:test_end])
        split["train"].extend(keys[test_end:])
    return {
        "schema": PHASE_B2_SCHEMA,
        "seed": seed,
        "split_strategy": "problem_level_dataset_and_positive_burden_stratified_60_10_15_15",
        "split_problem_keys": {
            name: [list(key) for key in sorted(keys)]
            for name, keys in split.items()
        },
        "problem_counts": {name: len(keys) for name, keys in split.items()},
    }


def validate_phase_b2_manifest(rows: list[dict[str, Any]], manifest: dict[str, Any], seed: int) -> None:
    if manifest.get("schema") != PHASE_B2_SCHEMA or int(manifest.get("seed", -1)) != int(seed):
        raise ValueError("Phase B2 manifest schema or seed mismatch")
    expected = {"train", "validation", "calibration", "test"}
    split_keys = manifest.get("split_problem_keys", {})
    if set(split_keys) != expected:
        raise ValueError(f"Phase B2 manifest must contain exactly {sorted(expected)}")
    known = {problem_key(row) for row in rows}
    seen: set[tuple[str, str]] = set()
    actual_counts: dict[str, int] = {}
    for split in sorted(expected):
        raw_keys = [(str(dataset), str(problem_id)) for dataset, problem_id in split_keys[split]]
        keys = set(raw_keys)
        if len(keys) != len(raw_keys):
            raise ValueError(f"Phase B2 manifest split {split} contains duplicate problems")
        if not keys:
            raise ValueError(f"Phase B2 manifest split {split} is empty")
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"Phase B2 manifest contains cross-split problems: {sorted(overlap)[:3]}")
        seen.update(keys)
        actual_counts[split] = len(keys)
    if seen != known:
        raise ValueError("Phase B2 manifest does not cover exactly the dataset problems")
    if manifest.get("problem_counts") != actual_counts:
        raise ValueError("Phase B2 manifest problem_counts do not match split contents")


def indices_for_split(rows: list[dict[str, Any]], manifest: dict[str, Any], split: str) -> list[int]:
    keys = {tuple(value) for value in manifest["split_problem_keys"][split]}
    indices = [index for index, row in enumerate(rows) if problem_key(row) in keys]
    if not indices:
        raise ValueError(f"Phase B2 split {split} contains no rows")
    return indices


class PhaseB2Dataset(Dataset):
    def __init__(self, dataset_path: str | Path, hidden_path: str | Path, feature_set: str) -> None:
        self.rows = read_jsonl(dataset_path)
        self.hidden = torch.load(hidden_path, map_location="cpu")
        self.feature_set = feature_set
        if len(self.rows) != len(self.hidden):
            raise ValueError("Phase B2 rows and hidden states differ")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return (
            build_phase_b2_state_features(self.hidden[index], row, self.feature_set),
            torch.tensor(row["candidate_flipped"], dtype=torch.float32),
            torch.tensor(row["candidate_token_divergence"], dtype=torch.float32),
            torch.tensor(row["candidate_hidden_cosine_distance"], dtype=torch.float32),
            torch.tensor(row["candidate_ratios"], dtype=torch.float32),
            index,
        )


class PhaseB2MultiTaskNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.flip_head = nn.Linear(hidden_dim, 1)
        self.divergence_head = nn.Linear(hidden_dim, 1)
        self.hidden_drift_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, ratios: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.state_encoder(features)
        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0).expand(state.shape[0], -1)
        expanded = state.unsqueeze(1).expand(-1, ratios.shape[1], -1)
        action = torch.stack([ratios, ratios.square()], dim=-1)
        shared = self.action_encoder(torch.cat([expanded, action], dim=-1))
        return {
            "flip_logits": self.flip_head(shared).squeeze(-1),
            "token_divergence": torch.sigmoid(self.divergence_head(shared).squeeze(-1)),
            "hidden_drift": torch.sigmoid(self.hidden_drift_head(shared).squeeze(-1)),
        }


class PhaseB2LinearFlipNet(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.flip_head = nn.Linear(dim + 2, 1)

    def forward(self, features: torch.Tensor, ratios: torch.Tensor) -> dict[str, torch.Tensor]:
        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0).expand(features.shape[0], -1)
        expanded = features.unsqueeze(1).expand(-1, ratios.shape[1], -1)
        action = torch.stack([ratios, ratios.square()], dim=-1)
        logits = self.flip_head(torch.cat([expanded, action], dim=-1)).squeeze(-1)
        zeros = torch.zeros_like(logits)
        return {"flip_logits": logits, "token_divergence": zeros, "hidden_drift": zeros}


def build_phase_b2_model(model_type: str, dim: int, hidden_dim: int) -> nn.Module:
    if model_type == "linear":
        return PhaseB2LinearFlipNet(dim)
    if model_type == "nonlinear":
        return PhaseB2MultiTaskNet(dim, hidden_dim)
    raise ValueError(f"Unknown Phase B2 model type: {model_type}")


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    flipped: torch.Tensor,
    divergence: torch.Tensor,
    hidden_drift: torch.Tensor,
    ratios: torch.Tensor,
    *,
    positive_weight: float,
    divergence_weight: float,
    hidden_drift_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_weight = torch.tensor(positive_weight, device=flipped.device)
    nonzero = ratios > 0
    flip_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["flip_logits"][nonzero], flipped[nonzero], pos_weight=pos_weight
    )
    divergence_loss = nn.functional.smooth_l1_loss(
        outputs["token_divergence"][nonzero], divergence[nonzero]
    )
    hidden_loss = nn.functional.smooth_l1_loss(
        outputs["hidden_drift"][nonzero], hidden_drift[nonzero]
    )
    total = (
        flip_loss
        + float(divergence_weight) * divergence_loss
        + float(hidden_drift_weight) * hidden_loss
    )
    return total, {
        "flip_loss": float(flip_loss.detach().item()),
        "divergence_loss": float(divergence_loss.detach().item()),
        "hidden_drift_loss": float(hidden_loss.detach().item()),
    }


@torch.no_grad()
def predict_phase_b2(
    model: nn.Module,
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    ratios: list[float],
    feature_set: str,
    device: torch.device,
) -> dict[str, list[list[float]]]:
    model.eval()
    ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=device).unsqueeze(0)
    risks, divergence, drift = [], [], []
    for index, row in enumerate(rows):
        features = build_phase_b2_state_features(hidden[index], row, feature_set).to(device)
        outputs = model(features.unsqueeze(0), ratio_tensor)
        row_risks = torch.sigmoid(outputs["flip_logits"]).squeeze(0).cpu().tolist()
        row_risks[0] = 0.0
        risks.append(row_risks)
        divergence.append(outputs["token_divergence"].squeeze(0).cpu().tolist())
        drift.append(outputs["hidden_drift"].squeeze(0).cpu().tolist())
    return {
        "risks": monotonic_risk_envelope(risks),
        "token_divergence": divergence,
        "hidden_drift": drift,
    }


def calibrate_problem_folds(
    rows: list[dict[str, Any]],
    risks: list[list[float]],
    ratios: list[float],
    *,
    budgets: list[float],
    max_flip_rates: list[float],
    folds: int,
    seed: int,
) -> dict[str, Any]:
    by_problem: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_problem[problem_key(row)].append(index)
    problems = sorted(by_problem)
    fold_indices = [[] for _ in range(max(1, min(folds, len(problems))))]
    risk_strata = problem_risk_strata(rows)
    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key in by_problem:
        strata[(key[0], risk_strata[key])].append(key)
    rng = random.Random(seed)
    offset = 0
    for keys in strata.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        for index, key in enumerate(keys):
            fold_indices[(offset + index) % len(fold_indices)].extend(by_problem[key])
        offset += len(keys)
    output = {}
    for budget, max_flip in zip(budgets, max_flip_rates):
        frontier = []
        for value in range(1, 100):
            threshold = value / 100.0
            selected = causal_action_risk_indices_from_risks(
                rows, risks, ratios, threshold, target_budget=budget
            )
            fold_metrics = []
            for indices in fold_indices:
                fold_rows = [rows[index] for index in indices]
                fold_risks = [risks[index] for index in indices]
                fold_selected = causal_action_risk_indices_from_risks(
                    fold_rows, fold_risks, ratios, threshold, target_budget=budget
                )
                fold_metrics.append(selection_metrics(fold_selected, fold_rows, ratios, target_budget=budget))
            frontier.append(
                {
                    "threshold": threshold,
                    **selection_metrics(selected, rows, ratios, target_budget=budget),
                    "max_fold_flip_rate": max(item["flip_rate"] for item in fold_metrics),
                }
            )
        feasible = [
            item for item in frontier
            if item["flip_rate"] <= max_flip + 1e-12
            and item["max_fold_flip_rate"] <= max_flip + 1e-12
        ]
        best = max(feasible, key=lambda item: item["average_selected_ratio"]) if feasible else min(
            frontier, key=lambda item: (item["max_fold_flip_rate"], item["flip_rate"], -item["average_selected_ratio"])
        )
        output[f"{budget:.2f}"] = {
            "threshold": best["threshold"],
            "constraints_satisfied": bool(best in feasible),
            "selected": best,
            "frontier": frontier,
        }
    return output
