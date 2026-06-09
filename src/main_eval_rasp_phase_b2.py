from __future__ import annotations

import argparse

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.rasp.phase_b2 import (
    PHASE_B2_SCHEMA,
    PhaseB2Dataset,
    build_phase_b2_model,
    indices_for_split,
    predict_phase_b2,
    validate_phase_b2_manifest,
)
from src.rasp.train_policy import causal_action_risk_indices_from_risks, selection_metrics
from src.utils.io import read_json, write_json


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
    manifest = read_json(args.manifest)
    if metadata.get("schema") != PHASE_B2_SCHEMA or int(metadata["seed"]) != int(manifest["seed"]):
        raise ValueError("Phase B2 checkpoint and manifest do not match")
    dataset = PhaseB2Dataset(args.dataset, args.hidden_states, metadata["feature_set"])
    validate_phase_b2_manifest(dataset.rows, manifest, int(metadata["seed"]))
    indices = indices_for_split(dataset.rows, manifest, "test")
    rows = [dataset.rows[index] for index in indices]
    hidden = dataset.hidden[torch.tensor(indices)]
    model = build_phase_b2_model(metadata["model_type"], metadata["dim"], metadata["hidden_dim"])
    model.load_state_dict(checkpoint["model"])
    predictions = predict_phase_b2(
        model, rows, hidden, metadata["ratios"], metadata["feature_set"], torch.device("cpu")
    )
    labels = [int(value) for row in rows for value in row["candidate_flipped"][1:]]
    scores = [float(value) for risk in predictions["risks"] for value in risk[1:]]
    result = {
        "schema": PHASE_B2_SCHEMA,
        "variant": metadata["variant"],
        "seed": metadata["seed"],
        "checkpoint_selection_split": metadata.get("checkpoint_selection_split"),
        "split_problem_counts": metadata.get("split_problem_counts"),
        "test_problems": len({(row["dataset"], row["id"]) for row in rows}),
        "test_boundaries": len(rows),
        "risk_roc_auc": float(roc_auc_score(labels, scores)),
        "risk_pr_auc": float(average_precision_score(labels, scores)),
        "budgets": {},
    }
    for budget, calibration in metadata["calibration"].items():
        selected = causal_action_risk_indices_from_risks(
            rows, predictions["risks"], metadata["ratios"], calibration["threshold"], target_budget=float(budget)
        )
        result["budgets"][budget] = {
            "threshold": calibration["threshold"],
            "calibration_constraints_satisfied": calibration["constraints_satisfied"],
            **selection_metrics(selected, rows, metadata["ratios"], target_budget=float(budget)),
        }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
