from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn


STAGES = ["understanding", "planning", "derivation", "verification", "final"]
STAGE_PROBE_SCHEMA = "rasp_stage_probe_s1_v1"
STAGE_VARIANTS = {
    "position_only": "linear",
    "uncertainty_only": "nonlinear",
    "hidden_pca_linear": "linear",
    "hidden_pca_nonlinear": "nonlinear",
    "hidden_uncertainty": "nonlinear",
}


def stage_index(stage: str) -> int:
    if stage not in STAGES:
        raise ValueError(f"Unknown reasoning stage: {stage}")
    return STAGES.index(stage)


def problem_stage_split(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_problem: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_problem[(str(row.get("dataset") or "unknown"), str(row["id"]))].append(index)
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in by_problem:
        grouped[key[0]].append(key)
    rng = random.Random(seed)
    split_keys = {"train": [], "validation": [], "test": []}
    for dataset, keys in grouped.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        tie_break = {key: rng.random() for key in keys}
        targets = {
            "validation": max(1, round(len(keys) * 0.15)),
            "test": max(1, round(len(keys) * 0.15)),
        }
        targets["train"] = len(keys) - targets["validation"] - targets["test"]
        if targets["train"] < 1:
            raise ValueError(f"Not enough {dataset} problems for three-way split")
        stage_counts = {
            key: Counter(str(rows[index]["stage"]) for index in by_problem[key])
            for key in keys
        }
        # Place rare-stage problems first, then greedily fill each split's
        # problem quota while balancing stage counts.
        rarity = Counter(stage for counts in stage_counts.values() for stage in counts)
        keys.sort(key=lambda key: (sum(rarity[s] for s in stage_counts[key]), tie_break[key]))
        assigned_counts = {name: Counter() for name in split_keys}
        desired = {
            name: Counter(
                {
                    stage: sum(stage_counts[key][stage] for key in keys) * targets[name] / len(keys)
                    for stage in STAGES
                }
            )
            for name in split_keys
        }
        dataset_split_keys = {name: [] for name in split_keys}
        for key in keys:
            candidates = [name for name in split_keys if len(dataset_split_keys[name]) < targets[name]]
            selected = min(
                candidates,
                key=lambda name: (
                    sum(
                        abs(
                            assigned_counts[name][stage]
                            + stage_counts[key][stage]
                            - desired[name][stage]
                        )
                        for stage in STAGES
                    ),
                    len(dataset_split_keys[name]) / targets[name],
                    name,
                ),
            )
            dataset_split_keys[selected].append(key)
            assigned_counts[selected].update(stage_counts[key])
        for name in split_keys:
            split_keys[name].extend(dataset_split_keys[name])
    return {
        "schema": STAGE_PROBE_SCHEMA,
        "seed": seed,
        "split_strategy": "problem_level_dataset_balanced_stage_greedy_70_15_15",
        "split_problem_keys": {
            name: [list(key) for key in sorted(keys)] for name, keys in split_keys.items()
        },
        "problem_counts": {name: len(keys) for name, keys in split_keys.items()},
        "stage_counts": {
            name: dict(
                Counter(
                    str(row["stage"])
                    for row in rows
                    if (str(row.get("dataset") or "unknown"), str(row["id"])) in set(keys)
                )
            )
            for name, keys in split_keys.items()
        },
    }


def indices_for_stage_split(rows: list[dict[str, Any]], manifest: dict[str, Any], split: str) -> list[int]:
    keys = {tuple(value) for value in manifest["split_problem_keys"][split]}
    indices = [
        index
        for index, row in enumerate(rows)
        if (str(row.get("dataset") or "unknown"), str(row["id"])) in keys
    ]
    if not indices:
        raise ValueError(f"Stage split {split} is empty")
    return indices


def validate_stage_manifest(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    seed: int,
) -> None:
    if manifest.get("schema") != STAGE_PROBE_SCHEMA or int(manifest.get("seed", -1)) != seed:
        raise ValueError("Stage manifest schema or seed mismatch")
    split_keys = manifest.get("split_problem_keys", {})
    if set(split_keys) != {"train", "validation", "test"}:
        raise ValueError("Stage manifest must contain train, validation, and test splits")
    known = {(str(row.get("dataset") or "unknown"), str(row["id"])) for row in rows}
    seen: set[tuple[str, str]] = set()
    for split in ("train", "validation", "test"):
        keys = {tuple(value) for value in split_keys[split]}
        if not keys:
            raise ValueError(f"Stage manifest split {split} is empty")
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"Stage manifest has cross-split problem overlap: {sorted(overlap)[:3]}")
        seen.update(keys)
    if seen != known:
        missing = known - seen
        unknown = seen - known
        raise ValueError(
            f"Stage manifest problem coverage mismatch: missing={len(missing)}, unknown={len(unknown)}"
        )


def fit_stage_transform(
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    train_indices: list[int],
    *,
    pca_dim: int,
) -> dict[str, Any]:
    position = torch.tensor([[float(row["position"])] for row in rows], dtype=torch.float32)
    uncertainty = torch.tensor(
        [[float(row["entropy"]), float(row["confidence"])] for row in rows],
        dtype=torch.float32,
    )
    hidden_values = hidden.float().flatten(start_dim=1)
    train = torch.tensor(train_indices, dtype=torch.long)

    def standardizer(values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mean": values[train].mean(dim=0),
            "scale": values[train].std(dim=0, unbiased=False).clamp_min(1e-6),
        }

    hidden_state = standardizer(hidden_values)
    standardized_hidden = (hidden_values[train] - hidden_state["mean"]) / hidden_state["scale"]
    rank = min(int(pca_dim), standardized_hidden.shape[0] - 1, standardized_hidden.shape[1])
    if rank < 1:
        raise ValueError("Stage PCA requires at least two train rows")
    _u, _s, projection = torch.pca_lowrank(standardized_hidden, q=rank, center=False)
    return {
        "position": standardizer(position),
        "uncertainty": standardizer(uncertainty),
        "hidden": hidden_state,
        "hidden_projection": projection,
        "pca_dim": rank,
        "fit_split": "train",
        "fit_row_count": len(train_indices),
    }


def transform_stage_features(
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    transform: dict[str, Any],
    variant: str,
) -> torch.Tensor:
    if variant not in STAGE_VARIANTS:
        raise ValueError(f"Unknown stage probe variant: {variant}")

    def standardize(values: torch.Tensor, name: str) -> torch.Tensor:
        return (values - transform[name]["mean"]) / transform[name]["scale"]

    position = standardize(
        torch.tensor([[float(row["position"])] for row in rows], dtype=torch.float32),
        "position",
    )
    uncertainty = standardize(
        torch.tensor(
            [[float(row["entropy"]), float(row["confidence"])] for row in rows],
            dtype=torch.float32,
        ),
        "uncertainty",
    )
    hidden_pca = (
        standardize(hidden.float().flatten(start_dim=1), "hidden")
        @ transform["hidden_projection"]
    )
    if variant == "position_only":
        return position
    if variant == "uncertainty_only":
        return uncertainty
    if variant in {"hidden_pca_linear", "hidden_pca_nonlinear"}:
        return hidden_pca
    return torch.cat([hidden_pca, uncertainty], dim=1)


class StageProbeNet(nn.Module):
    def __init__(self, dim: int, model_type: str, model_dim: int = 64) -> None:
        super().__init__()
        if model_type == "linear":
            self.net = nn.Linear(dim, len(STAGES))
        elif model_type == "nonlinear":
            self.net = nn.Sequential(
                nn.Linear(dim, model_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(model_dim, len(STAGES)),
            )
        else:
            raise ValueError(f"Unknown stage model type: {model_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def stage_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    recalls = recall_score(labels, predictions, labels=list(range(len(STAGES))), average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(len(STAGES))), average="macro", zero_division=0)),
        "per_stage_recall": {stage: float(value) for stage, value in zip(STAGES, recalls)},
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(STAGES)))
        ).tolist(),
    }
