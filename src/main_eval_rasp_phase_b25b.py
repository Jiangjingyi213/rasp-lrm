from __future__ import annotations

import argparse
import random

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.rasp.phase_b2 import PhaseB2Dataset, indices_for_split, validate_phase_b2_manifest
from src.rasp.phase_b25 import boundary_any_flip_metrics, transform_phase_b25_features
from src.rasp.phase_b25b import PHASE_B25B_SCHEMA, HiddenActionResidual, combined_risks, frozen_uncertainty_logits
from src.rasp.train_policy import causal_action_risk_indices_from_risks, selection_metrics
from src.utils.io import read_json, write_json


def metric_pair(labels: list[int], scores: list[float]) -> dict[str, float]:
    return {"roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores))}


def score_rows(rows: list[dict], risks: list[list[float]], mode: str) -> tuple[list[int], list[float]]:
    if mode == "action":
        return (
            [int(value) for row in rows for value in row["candidate_flipped"][1:]],
            [float(value) for row_risk in risks for value in row_risk[1:]],
        )
    return boundary_any_flip_metrics(rows, risks)


def paired_problem_bootstrap(
    rows: list[dict],
    base_risks: list[list[float]],
    combined: list[list[float]],
    *,
    mode: str,
    seed: int,
    samples: int = 1000,
) -> dict[str, object]:
    by_problem: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        by_problem.setdefault((str(row.get("dataset")), str(row["id"])), []).append(index)
    problems = sorted(by_problem)
    rng = random.Random(seed)
    deltas = {"roc_auc": [], "pr_auc": []}
    for _ in range(samples):
        sampled = [rng.choice(problems) for _ in problems]
        indices = [index for key in sampled for index in by_problem[key]]
        sampled_rows = [rows[index] for index in indices]
        base_labels, base_scores = score_rows(sampled_rows, [base_risks[index] for index in indices], mode)
        combined_labels, combined_scores = score_rows(sampled_rows, [combined[index] for index in indices], mode)
        if len(set(base_labels)) < 2:
            continue
        deltas["roc_auc"].append(roc_auc_score(combined_labels, combined_scores) - roc_auc_score(base_labels, base_scores))
        deltas["pr_auc"].append(
            average_precision_score(combined_labels, combined_scores) - average_precision_score(base_labels, base_scores)
        )
    output = {}
    for metric, values in deltas.items():
        ordered = sorted(float(value) for value in values)
        if not ordered:
            raise ValueError("Paired problem bootstrap produced no valid two-class resamples")
        output[metric] = {
            "mean": sum(ordered) / len(ordered),
            "ci95": [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]],
            "positive_fraction": sum(value > 0 for value in ordered) / len(ordered),
        }
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
    if metadata.get("schema") != PHASE_B25B_SCHEMA or metadata.get("transform_fit_split") != "train":
        raise ValueError("Expected a train-only Phase B2.5b checkpoint")
    base_checkpoint = checkpoint.get("frozen_base_checkpoint")
    if base_checkpoint is None:
        raise ValueError("Phase B2.5b checkpoint does not embed its frozen baseline")
    if int(base_checkpoint["metadata"].get("seed", -1)) != int(metadata["seed"]):
        raise ValueError("Frozen base checkpoint seed mismatch")
    if (
        base_checkpoint["metadata"].get("variant") != "uncertainty_flip_only"
        or base_checkpoint["metadata"].get("checkpoint_selection_split") != "validation"
    ):
        raise ValueError("Phase B2.5b evaluation requires the validation-selected v3 uncertainty baseline")
    if base_checkpoint["metadata"].get("ratios") != metadata.get("ratios"):
        raise ValueError("Frozen base checkpoint and residual checkpoint ratio grids differ")
    dataset = PhaseB2Dataset(args.dataset, args.hidden_states, "ratio_only")
    manifest = read_json(args.manifest)
    validate_phase_b2_manifest(dataset.rows, manifest, int(metadata["seed"]))
    indices = indices_for_split(dataset.rows, manifest, "test")
    rows = [dataset.rows[index] for index in indices]
    hidden = dataset.hidden[torch.tensor(indices)]
    _uncertainty, hidden_pca = transform_phase_b25_features(rows, hidden, checkpoint["transform"])
    device = torch.device("cpu")
    base_logits = frozen_uncertainty_logits(base_checkpoint, rows, hidden, device)
    model = HiddenActionResidual(metadata["pca_dim"], metadata["model_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        residual = model(hidden_pca, torch.tensor(metadata["ratios"], dtype=torch.float32))
    base_risks = combined_risks(base_logits, torch.zeros_like(residual), 0.0)
    risks = combined_risks(base_logits, residual, metadata["best"]["alpha"])
    result = {
        "schema": PHASE_B25B_SCHEMA,
        "seed": metadata["seed"],
        "alpha": metadata["best"]["alpha"],
        "checkpoint_selection_split": metadata["checkpoint_selection_split"],
        "alpha_selection_split": metadata["alpha_selection_split"],
        "transform_fit_split": metadata["transform_fit_split"],
        "base": {},
        "combined": {},
        "paired_delta": {},
        "budgets": {},
    }
    for mode in ("action", "boundary"):
        labels, base_scores = score_rows(rows, base_risks, mode)
        _labels, scores = score_rows(rows, risks, mode)
        result["base"][mode] = metric_pair(labels, base_scores)
        result["combined"][mode] = metric_pair(labels, scores)
        result["paired_delta"][mode] = paired_problem_bootstrap(
            rows, base_risks, risks, mode=mode, seed=int(metadata["seed"])
        )
    for budget, calibration in metadata["calibration"].items():
        selected = causal_action_risk_indices_from_risks(
            rows, risks, metadata["ratios"], calibration["threshold"], target_budget=float(budget)
        )
        base_calibration = base_checkpoint["metadata"]["calibration"][budget]
        base_selected = causal_action_risk_indices_from_risks(
            rows, base_risks, metadata["ratios"], base_calibration["threshold"], target_budget=float(budget)
        )
        result["budgets"][budget] = {
            "base": selection_metrics(base_selected, rows, metadata["ratios"], target_budget=float(budget)),
            "combined": selection_metrics(selected, rows, metadata["ratios"], target_budget=float(budget)),
        }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
