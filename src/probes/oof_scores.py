from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.probes.dataset import HiddenStateRiskDataset
from src.probes.action_conditioned_dataset import ActionConditionedRiskDataset
from src.probes.train_probe import LinearRiskProbe
from src.utils.io import read_jsonl


def _problem_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]))


def _problem_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> list[list[int]]:
    by_problem: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_problem[_problem_key(row)].append(index)
    problems = list(by_problem)
    if len(problems) < 2:
        raise ValueError("Need at least two problems for out-of-fold predictions")
    random.Random(seed).shuffle(problems)
    folds = max(2, min(int(folds), len(problems)))
    val_folds: list[list[int]] = [[] for _ in range(folds)]
    for fold_index, problem in enumerate(problems):
        val_folds[fold_index % folds].extend(by_problem[problem])
    return val_folds


def _evaluate_scores(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    if len(set(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def _predict(model: nn.Module, dataset: Dataset, indices: list[int], batch_size: int, device: torch.device) -> list[float]:
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size)
    scores: list[float] = []
    with torch.no_grad():
        for x, _y, _rows in loader:
            logits = model(x.to(device))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


def generate_oof_scores(
    run_dirs: list[str],
    feature_set: str = "hidden",
    folds: int = 5,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datasets = []
    action_layer_dim = None
    if feature_set in {"action_hidden", "action_hidden_stage"}:
        action_rows = [
            row
            for run_dir in run_dirs
            for row in read_jsonl(Path(run_dir) / "05_probe_dataset.jsonl")
        ]
        action_layer_dim = max(
            [int(layer) for row in action_rows for layer in row.get("pruned_layers", []) if layer is not None] + [0]
        ) + 1
    for run_dir in run_dirs:
        root = Path(run_dir)
        if feature_set in {"action_hidden", "action_hidden_stage"}:
            datasets.append(
                ActionConditionedRiskDataset(
                    root / "05_probe_dataset.jsonl",
                    root / "05_probe_hidden_states.pt",
                    include_stage=feature_set == "action_hidden_stage",
                    layer_dim=action_layer_dim,
                )
            )
        else:
            datasets.append(
                HiddenStateRiskDataset(
                    root / "05_probe_dataset.jsonl",
                    root / "05_probe_hidden_states.pt" if feature_set in {"hidden", "combined"} else None,
                    root / "05_probe_activation_features.pt" if feature_set in {"activation", "combined"} else None,
                    feature_set=feature_set,
                )
            )
    dataset = ConcatDataset(datasets)
    rows = [row for shard in datasets for row in shard.rows]
    val_folds = _problem_folds(rows, folds=folds, seed=seed)
    all_indices = set(range(len(rows)))
    scores: list[float | None] = [None] * len(rows)
    fold_summaries = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_x, _, _ = dataset[0]
    dim = int(sample_x.shape[-1])
    loss_fn = nn.BCEWithLogitsLoss()

    for fold_index, val_indices in enumerate(val_folds):
        train_indices = sorted(all_indices - set(val_indices))
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
        model = LinearRiskProbe(dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        for _epoch in range(epochs):
            model.train()
            for x, y, _rows in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
        fold_scores = _predict(model, dataset, val_indices, batch_size=batch_size, device=device)
        for row_index, score in zip(val_indices, fold_scores):
            scores[row_index] = float(score)
        fold_labels = [int(bool(rows[index]["flipped"])) for index in val_indices]
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_problem_count": len({_problem_key(rows[index]) for index in train_indices}),
                "val_problem_count": len({_problem_key(rows[index]) for index in val_indices}),
                "train_rows": len(train_indices),
                "val_rows": len(val_indices),
                "positive_rate_val": sum(fold_labels) / len(fold_labels),
                **_evaluate_scores(fold_labels, fold_scores),
            }
        )

    if any(score is None for score in scores):
        raise RuntimeError("Some rows did not receive an out-of-fold score")
    output_rows = []
    for row, score in zip(rows, scores):
        output_rows.append(
            {
                "dataset": row.get("dataset"),
                "id": row["id"],
                "segment_id": row["segment_id"],
                "segment_type": row.get("segment_type", "unknown"),
                "module": row.get("module"),
                "ratio": row.get("ratio"),
                "pruned_layers": row.get("pruned_layers", []),
                "flipped": bool(row["flipped"]),
                "risk_score": float(score),
                "feature_set": feature_set,
            }
        )
    labels = [int(bool(row["flipped"])) for row in rows]
    score_values = [float(score) for score in scores]
    summary = {
        "feature_set": feature_set,
        "split": "problem_oof",
        "folds": len(val_folds),
        "rows": len(rows),
        "problem_count": len({_problem_key(row) for row in rows}),
        "positive_rate": sum(labels) / len(labels),
        "feature_dim": dim,
        **_evaluate_scores(labels, score_values),
        "fold_summaries": fold_summaries,
    }
    return output_rows, summary
