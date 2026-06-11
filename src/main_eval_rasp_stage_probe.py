from __future__ import annotations

import argparse

import torch

from src.rasp.stage_probe import (
    STAGE_PROBE_SCHEMA,
    StageProbeNet,
    indices_for_stage_split,
    stage_index,
    stage_metrics,
    transform_stage_features,
    validate_stage_manifest,
)
from src.utils.io import read_json, read_jsonl, write_json


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
    if metadata.get("schema") != STAGE_PROBE_SCHEMA or int(metadata["seed"]) != int(manifest["seed"]):
        raise ValueError("Stage checkpoint and manifest do not match")
    rows = read_jsonl(args.dataset)
    hidden = torch.load(args.hidden_states, map_location="cpu")
    validate_stage_manifest(rows, manifest, int(metadata["seed"]))
    indices = indices_for_stage_split(rows, manifest, "test")
    features = transform_stage_features(rows, hidden, checkpoint["transform"], metadata["variant"])
    model = StageProbeNet(metadata["dim"], metadata["model_type"], metadata["model_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(features[indices]), dim=1)
    labels = [stage_index(str(rows[index]["stage"])) for index in indices]
    predictions = probabilities.argmax(dim=1).tolist()
    per_dataset = {}
    for dataset in sorted({str(rows[index]["dataset"]) for index in indices}):
        positions = [
            position for position, index in enumerate(indices)
            if str(rows[index]["dataset"]) == dataset
        ]
        per_dataset[dataset] = {
            "test_rows": len(positions),
            **stage_metrics(
                [labels[position] for position in positions],
                [predictions[position] for position in positions],
            ),
        }
    result = {
        "schema": STAGE_PROBE_SCHEMA,
        "variant": metadata["variant"],
        "seed": metadata["seed"],
        "checkpoint_selection_split": metadata["checkpoint_selection_split"],
        "test_rows": len(indices),
        "test_problems": len({(rows[index]["dataset"], rows[index]["id"]) for index in indices}),
        "mean_max_probability": float(probabilities.max(dim=1).values.mean().item()),
        "per_dataset": per_dataset,
        **stage_metrics(labels, predictions),
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
