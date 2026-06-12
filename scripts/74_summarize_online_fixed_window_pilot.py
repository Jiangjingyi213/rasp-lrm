#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.answer_match import answer_match
from src.utils.io import read_json, read_jsonl, write_json


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset") or "unknown"), str(row["id"])


def correct(row: dict[str, Any]) -> bool:
    return answer_match(str(row.get("completion", "")), str(row.get("gold", "")))


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def paired_bootstrap(pairs: list[tuple[dict[str, Any], dict[str, Any]]], seed: int, samples: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    deltas, flips = [], []
    for _ in range(samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(sum(int(correct(policy)) - int(correct(dense)) for dense, policy in sample) / len(sample))
        dense_correct = [pair for pair in sample if correct(pair[0])]
        flips.append(
            sum(int(not correct(policy)) for _dense, policy in dense_correct) / max(1, len(dense_correct))
        )
    return {
        "accuracy_delta_95ci": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "dense_correct_flip_rate_95ci": [percentile(flips, 0.025), percentile(flips, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/07_stage_aware/07_online_fixed_window_pilot")
    parser.add_argument("--manifest", default="configs/generated_online_fixed_window_pilot/manifest.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cells = []
    for dataset in ("gsm8k", "math500"):
        dense_path = Path(args.root) / dataset / "dense" / "01_trajectories.jsonl"
        if not dense_path.exists():
            raise ValueError(f"Missing dense reference: {dense_path}")
        dense_rows = {key(row): row for row in read_jsonl(dense_path)}
        for job in [row for row in manifest if row["dataset"] == dataset and row["tag"] != "dense"]:
            root = Path(job["run_dir"])
            policy_path = root / "01_trajectories.jsonl"
            summary_path = root / "00_runtime_summary.json"
            if not policy_path.exists() or not summary_path.exists():
                raise ValueError(f"Incomplete online job: {root}")
            policy_rows = {key(row): row for row in read_jsonl(policy_path)}
            if set(dense_rows) != set(policy_rows):
                raise ValueError(f"Dense/policy task mismatch: {root}")
            pairs = [(dense_rows[item], policy_rows[item]) for item in sorted(dense_rows)]
            if any(
                dense.get("prompt") != policy.get("prompt")
                or dense.get("question") != policy.get("question")
                or dense.get("gold") != policy.get("gold")
                for dense, policy in pairs
            ):
                raise ValueError(f"Dense/policy prompt or task contents differ: {root}")
            dense_correct = sum(int(correct(dense)) for dense, _policy in pairs)
            policy_correct = sum(int(correct(policy)) for _dense, policy in pairs)
            flips = sum(int(correct(dense) and not correct(policy)) for dense, policy in pairs)
            improvements = sum(int(not correct(dense) and correct(policy)) for dense, policy in pairs)
            activations = sum(
                int(
                    any(
                        event.get("decision", {}).get("window_activated")
                        for event in policy["runtime"]["router_events"]
                        if isinstance(event.get("decision"), dict)
                    )
                )
                for _dense, policy in pairs
            )
            runtime_summary = read_json(summary_path)
            cells.append(
                {
                    "dataset": dataset,
                    "boundary_tokens": job["boundary_tokens"],
                    "ratio": job["ratio"],
                    "examples": len(pairs),
                    "dense_correct": dense_correct,
                    "dense_accuracy": dense_correct / len(pairs),
                    "policy_correct": policy_correct,
                    "policy_accuracy": policy_correct / len(pairs),
                    "paired_accuracy_delta": (policy_correct - dense_correct) / len(pairs),
                    "dense_correct_flips": flips,
                    "dense_correct_flip_rate": flips / max(1, dense_correct),
                    "dense_wrong_policy_correct": improvements,
                    "executed_window_examples": activations,
                    "executed_window_rate": activations / len(pairs),
                    "average_theoretical_mlp_pruning_exposure": runtime_summary[
                        "average_decode_pruning_ratio"
                    ],
                    **paired_bootstrap(pairs, seed=1, samples=args.bootstrap_samples),
                }
            )
    write_json(
        Path(args.root) / "online_fixed_window_summary.json",
        {
            "schema": "rasp_online_fixed_single_window_pilot_v1",
            "cells": cells,
            "logical_mask_only": True,
            "real_speedup_claimed": False,
            "note": "Pilot fixed-action diagnostics; not a learned-controller safety claim.",
        },
    )
    print(f"summarized {len(cells)} online fixed-window cells")


if __name__ == "__main__":
    main()
