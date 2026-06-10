from __future__ import annotations

import argparse
import random

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.rasp.phase_b2 import PhaseB2Dataset, indices_for_split, validate_phase_b2_manifest
from src.rasp.phase_b25 import (
    PHASE_B25_SCHEMA,
    PhaseB25ActionNet,
    boundary_any_flip_metrics,
    predict_phase_b25,
    transform_phase_b25_features,
)
from src.rasp.train_policy import causal_action_risk_indices_from_risks, selection_metrics
from src.utils.io import read_json, write_json


def metric_pair(labels: list[int], scores: list[float]) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def problem_bootstrap(
    rows: list[dict],
    risks: list[list[float]],
    *,
    mode: str,
    seed: int,
    samples: int = 500,
) -> dict[str, list[float]]:
    by_problem: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        by_problem.setdefault((str(row.get("dataset")), str(row["id"])), []).append(index)
    problems = sorted(by_problem)
    rng = random.Random(seed)
    values = {"roc_auc": [], "pr_auc": []}
    for _ in range(samples):
        sampled = [rng.choice(problems) for _ in problems]
        indices = [index for key in sampled for index in by_problem[key]]
        sampled_rows = [rows[index] for index in indices]
        sampled_risks = [risks[index] for index in indices]
        if mode == "action":
            labels = [int(value) for row in sampled_rows for value in row["candidate_flipped"][1:]]
            scores = [float(value) for row_risk in sampled_risks for value in row_risk[1:]]
        else:
            labels, scores = boundary_any_flip_metrics(sampled_rows, sampled_risks)
        if len(set(labels)) < 2:
            continue
        values["roc_auc"].append(float(roc_auc_score(labels, scores)))
        values["pr_auc"].append(float(average_precision_score(labels, scores)))
    output = {}
    for name, metric_values in values.items():
        ordered = sorted(metric_values)
        if not ordered:
            output[name] = [float("nan"), float("nan")]
            continue
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        output[name] = [low, high]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    if metadata.get("schema") != PHASE_B25_SCHEMA:
        raise ValueError("Expected a Phase B2.5 checkpoint")
    if metadata.get("transform_fit_split") != "train" or checkpoint["transform"].get("fit_split") != "train":
        raise ValueError("Phase B2.5 preprocessing must be fit only on train rows")
    dataset = PhaseB2Dataset(args.dataset, args.hidden_states, "ratio_only")
    manifest = read_json(args.manifest)
    validate_phase_b2_manifest(dataset.rows, manifest, int(metadata["seed"]))
    indices = indices_for_split(dataset.rows, manifest, "test")
    rows = [dataset.rows[index] for index in indices]
    hidden = dataset.hidden[torch.tensor(indices)]
    uncertainty, hidden_pca = transform_phase_b25_features(rows, hidden, checkpoint["transform"])
    model = PhaseB25ActionNet(metadata["variant"], metadata["pca_dim"], metadata["model_dim"])
    model.load_state_dict(checkpoint["model"])
    risks = predict_phase_b25(model, uncertainty, hidden_pca, metadata["ratios"], torch.device("cpu"))
    action_labels = [int(value) for row in rows for value in row["candidate_flipped"][1:]]
    action_scores = [float(value) for row in risks for value in row[1:]]
    boundary_labels, boundary_scores = boundary_any_flip_metrics(rows, risks)
    action_metrics = metric_pair(action_labels, action_scores)
    action_metrics["problem_bootstrap_95ci"] = problem_bootstrap(
        rows, risks, mode="action", seed=int(metadata["seed"])
    )
    boundary_metrics = metric_pair(boundary_labels, boundary_scores)
    boundary_metrics["problem_bootstrap_95ci"] = problem_bootstrap(
        rows, risks, mode="boundary", seed=int(metadata["seed"])
    )
    result = {
        "schema": PHASE_B25_SCHEMA,
        "variant": metadata["variant"],
        "seed": metadata["seed"],
        "checkpoint_selection_split": metadata["checkpoint_selection_split"],
        "transform_fit_split": metadata["transform_fit_split"],
        "test_problems": len({(row["dataset"], row["id"]) for row in rows}),
        "test_boundaries": len(rows),
        "action": action_metrics,
        "boundary_any_flip": boundary_metrics,
        "budgets": {},
    }
    for budget, calibration in metadata["calibration"].items():
        selected = causal_action_risk_indices_from_risks(
            rows, risks, metadata["ratios"], calibration["threshold"], target_budget=float(budget)
        )
        result["budgets"][budget] = {
            "threshold": calibration["threshold"],
            "calibration_constraints_satisfied": calibration["constraints_satisfied"],
            **selection_metrics(selected, rows, metadata["ratios"], target_budget=float(budget)),
        }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
