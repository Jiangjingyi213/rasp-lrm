from __future__ import annotations

import argparse

import torch

from src.rasp.stage_probe import (
    STAGE_PROBE_SCHEMA,
    StageProbeNet,
    indices_for_stage_split,
    stage_index,
    transform_stage_features,
    validate_stage_manifest,
)
from src.rasp.stage_selective import calibrate_reasoning_threshold, evaluate_reasoning_threshold
from src.utils.io import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-setup-false-accept-rate", type=float, default=0.10)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    manifest = read_json(args.manifest)
    if metadata.get("schema") != STAGE_PROBE_SCHEMA or int(metadata["seed"]) != int(manifest["seed"]):
        raise ValueError("Stage checkpoint and manifest do not match")
    rows = read_jsonl(args.dataset)
    hidden = torch.load(args.hidden_states, map_location="cpu")
    validate_stage_manifest(rows, manifest, int(metadata["seed"]))
    features = transform_stage_features(rows, hidden, checkpoint["transform"], metadata["variant"])
    model = StageProbeNet(metadata["dim"], metadata["model_type"], metadata["model_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    split_results = {}
    probabilities_by_split = {}
    labels_by_split = {}
    with torch.no_grad():
        for split in ("validation", "test"):
            indices = indices_for_stage_split(rows, manifest, split)
            probabilities_by_split[split] = torch.softmax(model(features[indices]), dim=1)
            labels_by_split[split] = [stage_index(str(rows[index]["stage"])) for index in indices]
    calibration = calibrate_reasoning_threshold(
        labels_by_split["validation"],
        probabilities_by_split["validation"],
        max_setup_false_accept_rate=args.max_setup_false_accept_rate,
    )
    for split in ("validation", "test"):
        split_results[split] = evaluate_reasoning_threshold(
            labels_by_split[split],
            probabilities_by_split[split],
            calibration["threshold"],
        )
    write_json(
        args.output,
        {
            "schema": "rasp_stage_probe_s1_5_selective_v1",
            "seed": metadata["seed"],
            "variant": metadata["variant"],
            "checkpoint": args.checkpoint,
            "threshold_selection_split": "validation",
            "max_setup_false_accept_rate": args.max_setup_false_accept_rate,
            "calibration": calibration,
            "validation": split_results["validation"],
            "test": split_results["test"],
        },
    )


if __name__ == "__main__":
    main()
