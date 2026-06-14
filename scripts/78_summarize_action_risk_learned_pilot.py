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


def selected_event(row: dict[str, Any]) -> dict[str, Any] | None:
    for event in row["runtime"]["router_events"]:
        decision = event.get("decision")
        if isinstance(decision, dict) and decision.get("window_activated"):
            return event
    return None


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def bootstrap(flags: list[tuple[bool, bool]], seed: int, samples: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    deltas, flips = [], []
    for _ in range(samples):
        sample = [flags[rng.randrange(len(flags))] for _ in flags]
        deltas.append(sum(int(policy) - int(dense) for dense, policy in sample) / len(sample))
        dense_correct = [item for item in sample if item[0]]
        flips.append(sum(int(not policy) for _dense, policy in dense_correct) / max(1, len(dense_correct)))
    return {
        "accuracy_delta_95ci": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "dense_correct_flip_rate_95ci": [percentile(flips, 0.025), percentile(flips, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/07_stage_aware/08_action_risk_learned_single_window_pilot")
    parser.add_argument("--manifest", default="configs/generated_action_risk_learned_pilot/manifest.json")
    parser.add_argument("--fixed-summary", default="runs/07_stage_aware/07_online_fixed_window_pilot/online_fixed_window_summary.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fixed_cells = read_json(args.fixed_summary)["cells"] if Path(args.fixed_summary).exists() else []
    cells = []
    for dataset in ("gsm8k", "math500"):
        dense_rows = {
            key(row): row for row in read_jsonl(root / "online" / dataset / "dense" / "01_trajectories.jsonl")
        }
        for job in [item for item in manifest if item["dataset"] == dataset and item["tag"] != "dense"]:
            run_dir = Path(job["run_dir"])
            policy_rows = {key(row): row for row in read_jsonl(run_dir / "01_trajectories.jsonl")}
            if set(dense_rows) != set(policy_rows):
                raise ValueError(f"Dense/policy task mismatch: {run_dir}")
            pairs = [(dense_rows[item], policy_rows[item]) for item in sorted(dense_rows)]
            if any(
                dense.get("prompt") != policy.get("prompt")
                or dense.get("question") != policy.get("question")
                or dense.get("gold") != policy.get("gold")
                for dense, policy in pairs
            ):
                raise ValueError(f"Dense/policy prompt or task mismatch: {run_dir}")
            flags = [(correct(dense), correct(policy)) for dense, policy in pairs]
            events = [selected_event(policy) for _dense, policy in pairs]
            selected_ratios = [float(event["selected_ratio"]) for event in events if event is not None]
            selected_boundaries = [int(event["generated_tokens"]) for event in events if event is not None]
            dense_correct = sum(int(item[0]) for item in flags)
            policy_correct = sum(int(item[1]) for item in flags)
            flips = sum(int(dense and not policy) for dense, policy in flags)
            executed_dense_correct = [
                (dense, policy)
                for (dense, policy), event in zip(flags, events)
                if dense and event is not None
            ]
            executed_flips = sum(int(not policy) for _dense, policy in executed_dense_correct)
            runtime_summary = read_json(run_dir / "00_runtime_summary.json")
            average_action_ratio = sum(selected_ratios) / len(pairs)
            candidates = [cell for cell in fixed_cells if cell["dataset"] == dataset]
            matched = min(
                candidates,
                key=lambda cell: abs(
                    float(cell["average_theoretical_mlp_pruning_exposure"])
                    - float(runtime_summary["average_decode_pruning_ratio"])
                ),
                default=None,
            )
            cells.append(
                {
                    "dataset": dataset,
                    "policy_variant": job["policy_variant"],
                    "operating_point": job["operating_point"],
                    "examples": len(pairs),
                    "dense_accuracy": dense_correct / len(pairs),
                    "policy_accuracy": policy_correct / len(pairs),
                    "paired_accuracy_delta": (policy_correct - dense_correct) / len(pairs),
                    "dense_correct_flips": flips,
                    "dense_correct_flip_rate": flips / max(1, dense_correct),
                    "executed_dense_correct": len(executed_dense_correct),
                    "executed_dense_correct_flips": executed_flips,
                    "executed_dense_correct_flip_rate": (
                        executed_flips / len(executed_dense_correct)
                        if executed_dense_correct
                        else None
                    ),
                    "executed_windows": len(selected_ratios),
                    "action_execution_rate": len(selected_ratios) / len(pairs),
                    "average_action_ratio": average_action_ratio,
                    "selected_average_ratio": sum(selected_ratios) / max(1, len(selected_ratios)),
                    "selected_ratio_counts": {
                        f"{ratio:.2f}": selected_ratios.count(ratio) for ratio in sorted(set(selected_ratios))
                    },
                    "selected_boundary_counts": {
                        str(boundary): selected_boundaries.count(boundary)
                        for boundary in sorted(set(selected_boundaries))
                    },
                    "average_theoretical_mlp_pruning_exposure": runtime_summary[
                        "average_decode_pruning_ratio"
                    ],
                    "matched_fixed_baseline": matched,
                    **bootstrap(flags, seed=1, samples=args.bootstrap_samples),
                }
            )
    context_cells = [cell for cell in cells if cell["policy_variant"] == "context_only"]
    passing_points = []
    for point in ("conservative", "balanced", "aggressive"):
        point_cells = [cell for cell in context_cells if cell["operating_point"] == point]
        if len(point_cells) != 2:
            continue
        if all(
            cell["paired_accuracy_delta"] >= -0.05
            and cell["dense_correct_flip_rate"] <= 0.10
            and cell["average_action_ratio"] >= 0.08
            and cell["executed_windows"] > 0
            and (
                cell["matched_fixed_baseline"] is None
                or (
                    (
                        cell["dense_correct_flip_rate"]
                        < cell["matched_fixed_baseline"]["dense_correct_flip_rate"]
                        or cell["paired_accuracy_delta"]
                        > cell["matched_fixed_baseline"]["paired_accuracy_delta"]
                    )
                    and cell["dense_correct_flip_rate"]
                    <= cell["matched_fixed_baseline"]["dense_correct_flip_rate"]
                    and cell["paired_accuracy_delta"]
                    >= cell["matched_fixed_baseline"]["paired_accuracy_delta"]
                )
            )
            for cell in point_cells
        ):
            passing_points.append(point)
    write_json(
        root / "online_learned_summary.json",
        {
            "schema": "rasp_action_risk_learned_single_window_pilot_v1",
            "cells": cells,
            "passing_context_only_operating_points": passing_points,
            "pilot_passed": bool(passing_points),
            "logical_mask_only": True,
            "real_speedup_claimed": False,
        },
    )
    print(f"summarized {len(cells)} learned Action-Risk cells; passing={passing_points}")


if __name__ == "__main__":
    main()
