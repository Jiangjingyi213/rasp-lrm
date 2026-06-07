from __future__ import annotations

import argparse

import torch

from src.rasp.fair_benchmark import (
    FAIR_BENCHMARK_SCHEMA,
    FairActionRiskDataset,
    build_model,
    indices_for_split,
    predict_risks,
)
from src.rasp.train_policy import causal_action_risk_indices_from_risks, selection_metrics
from src.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b15-dataset", required=True)
    parser.add_argument("--b15-hidden-states", required=True)
    parser.add_argument("--b20-dataset", required=True)
    parser.add_argument("--b20-hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    if metadata.get("schema") != FAIR_BENCHMARK_SCHEMA:
        raise ValueError("Checkpoint is not a fair benchmark checkpoint")
    manifest = read_json(args.manifest)
    if (
        manifest.get("schema") != FAIR_BENCHMARK_SCHEMA
        or int(manifest["seed"]) != int(metadata["seed"])
    ):
        raise ValueError("Checkpoint and fair benchmark manifest do not match")
    model = build_model(metadata["model_type"], metadata["dim"], metadata["hidden_dim"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cpu")
    model.to(device)
    results = {
        "schema": FAIR_BENCHMARK_SCHEMA,
        "variant": metadata["variant"],
        "label_type": metadata["label_type"],
        "seed": metadata["seed"],
        "checkpoint": str(args.checkpoint),
        "budgets": {},
    }
    inputs = [
        ("0.15", args.b15_dataset, args.b15_hidden_states),
        ("0.20", args.b20_dataset, args.b20_hidden_states),
    ]
    for budget_key, dataset_path, hidden_path in inputs:
        dataset = FairActionRiskDataset(
            dataset_path,
            hidden_path,
            feature_set=metadata["feature_set"],
            label_type=metadata["label_type"],
            ratios=metadata["ratios"],
        )
        indices = indices_for_split(dataset.rows, manifest, "test")
        rows = [dataset.rows[index] for index in indices]
        hidden = dataset.hidden[torch.tensor(indices)]
        risks = predict_risks(model, rows, hidden, dataset.ratios, metadata["feature_set"], device)
        threshold = float(metadata["calibration"][budget_key]["threshold"])
        selected = causal_action_risk_indices_from_risks(
            rows, risks, dataset.ratios, threshold, target_budget=float(budget_key)
        )
        results["budgets"][budget_key] = {
            "threshold": threshold,
            "n": len(rows),
            **selection_metrics(selected, rows, dataset.ratios, target_budget=float(budget_key)),
            "calibration_constraints_satisfied": metadata["calibration"][budget_key][
                "constraints_satisfied"
            ],
        }
    write_json(args.output, results)
    print(results)


if __name__ == "__main__":
    main()
