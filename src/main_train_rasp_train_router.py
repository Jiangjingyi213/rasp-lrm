from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.probes.rasp_train_dataset import RaspTrainPolicyDataset
from src.rasp.train_policy import (
    ActionRiskPolicyNet,
    action_risk_loss,
    causal_action_risk_indices,
    checkpoint_metadata,
    evaluate_action_risk,
    selection_metrics,
)
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def _problem_key(row: dict) -> tuple[str, str]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]))


def problem_level_three_way_split(
    rows: list[dict],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, list[list[str]]]]:
    by_problem: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        by_problem.setdefault(_problem_key(row), []).append(index)
    problems = list(by_problem)
    if len(problems) < 3:
        raise ValueError("RASP-Train requires at least three problems for train/calibration/test splits")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    random.Random(seed).shuffle(problems)
    holdout_count = max(2, int(len(problems) * float(holdout_fraction)))
    holdout_count = min(holdout_count, len(problems) - 1)
    calibration_count = max(1, holdout_count // 2)
    calibration_keys = problems[:calibration_count]
    test_keys = problems[calibration_count:holdout_count]
    train_keys = problems[holdout_count:]

    def indices(keys: list[tuple[str, str]]) -> list[int]:
        return [index for key in keys for index in by_problem[key]]

    split_keys = {
        "train": [list(key) for key in sorted(train_keys)],
        "calibration": [list(key) for key in sorted(calibration_keys)],
        "test": [list(key) for key in sorted(test_keys)],
    }
    return indices(train_keys), indices(calibration_keys), indices(test_keys), split_keys


def _split_summary(rows: list[dict], split_indices: dict[str, list[int]]) -> dict:
    output = {"split": "problem_train_calibration_test"}
    for name, indices in split_indices.items():
        subset = [rows[index] for index in indices]
        values = [bool(value) for row in subset for value in row["candidate_unsafe"]]
        output[f"{name}_problem_count"] = len({_problem_key(row) for row in subset})
        output[f"{name}_rows"] = len(subset)
        output[f"{name}_unsafe_candidate_rate"] = sum(int(value) for value in values) / max(1, len(values))
    return output


def _subset_rows_hidden(dataset: RaspTrainPolicyDataset, indices: list[int]) -> tuple[list[dict], torch.Tensor]:
    rows = [dataset.rows[index] for index in indices]
    hidden = dataset.hidden[torch.tensor(indices, dtype=torch.long)]
    return rows, hidden


def _calibrate_threshold(
    model: ActionRiskPolicyNet,
    rows: list[dict],
    hidden: torch.Tensor,
    ratios: list[float],
    device: torch.device,
    *,
    max_flip_rate: float,
    max_unsafe_rate: float,
) -> tuple[float, list[dict]]:
    candidates = [value / 100.0 for value in range(1, 51)]
    summaries = []
    for threshold in candidates:
        selected, _risks = causal_action_risk_indices(model, rows, hidden, ratios, threshold, device)
        summaries.append(
            {
                "threshold": threshold,
                **selection_metrics(selected, rows, ratios),
            }
        )
    feasible = [
        row
        for row in summaries
        if float(row["flip_rate"]) <= float(max_flip_rate) + 1e-12
        and float(row["conservative_unsafe_rate"]) <= float(max_unsafe_rate) + 1e-12
    ]
    if feasible:
        best = max(
            feasible,
            key=lambda row: (
                float(row["average_selected_ratio"]),
                -float(row["conservative_unsafe_rate"]),
                -float(row["flip_rate"]),
            ),
        )
    else:
        best = min(
            summaries,
            key=lambda row: (
                float(row["flip_rate"]),
                float(row["conservative_unsafe_rate"]),
                -float(row["average_selected_ratio"]),
            ),
        )
    return float(best["threshold"]), summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--monotonic-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.30)
    parser.add_argument("--max-calibration-flip-rate", type=float, default=0.06)
    parser.add_argument("--max-calibration-unsafe-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = RaspTrainPolicyDataset(args.dataset, args.hidden_states)
    train_indices, calibration_indices, test_indices, split_keys = problem_level_three_way_split(
        dataset.rows, args.holdout_fraction, args.seed
    )
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    calibration_loader = DataLoader(Subset(dataset, calibration_indices), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample, unsafe_sample, _flipped, ratios_sample, _target, _index = dataset[0]
    model = ActionRiskPolicyNet(int(sample.shape[-1]), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    train_unsafe = torch.stack(
        [dataset[index][1] for index in train_indices]
    )
    positives = float(train_unsafe.sum().item())
    negatives = float(train_unsafe.numel() - positives)
    positive_weight = negatives / max(1.0, positives)

    output = Path(args.output)
    ensure_dir(output.parent)
    best = {"val_loss": float("inf")}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, unsafe_mask, _flipped_mask, ratios, _target_budget, _indices in train_loader:
            x = x.to(device)
            unsafe_mask = unsafe_mask.to(device)
            ratios = ratios.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, ratios)
            loss = action_risk_loss(
                logits,
                unsafe_mask,
                positive_weight=positive_weight,
                monotonic_weight=args.monotonic_weight,
                ranking_weight=args.ranking_weight,
            )
            loss.backward()
            optimizer.step()
        result = evaluate_action_risk(
            model,
            calibration_loader,
            device,
            positive_weight=positive_weight,
            monotonic_weight=args.monotonic_weight,
            ranking_weight=args.ranking_weight,
        )
        if result.val_loss < float(best["val_loss"]):
            best = {
                **result.__dict__,
                "epoch": epoch,
                "feature_set": "action_conditioned_hidden_risk",
                "positive_weight": positive_weight,
                "monotonic_weight": float(args.monotonic_weight),
                "ranking_weight": float(args.ranking_weight),
                **_split_summary(
                    dataset.rows,
                    {
                        "train": train_indices,
                        "calibration": calibration_indices,
                        "test": test_indices,
                    },
                ),
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "dim": int(sample.shape[-1]),
                    "hidden_dim": int(args.hidden_dim),
                    "ratios": [float(value) for value in dataset.ratios],
                    "best": best,
                },
                output,
            )
        print(
            f"epoch={epoch:02d} calibration_loss={result.val_loss:.4f} "
            f"roc_auc={result.roc_auc} pr_auc={result.pr_auc}"
        )

    checkpoint = torch.load(output, map_location=device)
    model.load_state_dict(checkpoint["model"])
    calibration_rows, calibration_hidden = _subset_rows_hidden(dataset, calibration_indices)
    threshold, threshold_summaries = _calibrate_threshold(
        model,
        calibration_rows,
        calibration_hidden,
        dataset.ratios,
        device,
        max_flip_rate=float(args.max_calibration_flip_rate),
        max_unsafe_rate=float(args.max_calibration_unsafe_rate),
    )
    best["calibrated_threshold"] = threshold
    best["max_calibration_flip_rate"] = float(args.max_calibration_flip_rate)
    best["max_calibration_unsafe_rate"] = float(args.max_calibration_unsafe_rate)
    best["threshold_calibration"] = threshold_summaries
    best["calibrated_selection"] = next(
        row for row in threshold_summaries if abs(float(row["threshold"]) - threshold) < 1e-12
    )
    best["calibration_constraints_satisfied"] = bool(
        float(best["calibrated_selection"]["flip_rate"]) <= float(args.max_calibration_flip_rate) + 1e-12
        and float(best["calibrated_selection"]["conservative_unsafe_rate"])
        <= float(args.max_calibration_unsafe_rate) + 1e-12
    )
    checkpoint["best"] = best
    checkpoint["metadata"] = checkpoint_metadata(
        ratios=dataset.ratios,
        best=best,
        calibrated_threshold=threshold,
        split_problem_keys=split_keys,
    )
    torch.save(checkpoint, output)
    write_json(args.metrics_output, best)


if __name__ == "__main__":
    main()
