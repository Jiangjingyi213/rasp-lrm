from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.metrics.answer_match import answer_match
from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import read_json, read_jsonl, read_yaml, write_json


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["id"])


def correct(row: dict[str, Any]) -> bool:
    return answer_match(str(row.get("completion", "")), str(row.get("gold", "")))


def selected_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in row["runtime"]["router_events"]
        if isinstance(event.get("decision"), dict)
        and bool(event["decision"].get("window_activated"))
    ]


def source_isolated(config: dict[str, Any], dataset: str) -> bool:
    data = config["data"]
    if dataset == "gsm8k":
        return data.get("dataset") == "gsm8k" and data.get("split") == "train"
    return (
        data.get("dataset") == "math500"
        and data.get("dataset_label") == "math_train"
        and data.get("name_or_path") == "rasbt/math_full_minus_math500"
        and data.get("split") == "train"
    )


def comparison_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        name: config.get(name)
        for name in ("model", "prompt", "data", "generation")
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def bootstrap(flags: list[tuple[bool, bool]], seed: int = 1, samples: int = 2000) -> dict[str, Any]:
    rng = random.Random(seed)
    deltas, flips = [], []
    for _ in range(samples):
        sample = [flags[rng.randrange(len(flags))] for _ in flags]
        deltas.append(sum(int(policy) - int(dense) for dense, policy in sample) / len(sample))
        dense_correct = [item for item in sample if item[0]]
        flips.append(
            sum(int(not policy) for _dense, policy in dense_correct)
            / max(1, len(dense_correct))
        )
    return {
        "accuracy_delta_95ci": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "dense_correct_flip_rate_95ci": [percentile(flips, 0.025), percentile(flips, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cells = []
    for dataset in ("gsm8k", "math_train"):
        jobs = [item for item in manifest if item["dataset"] == dataset]
        configs = {item["tag"]: read_yaml(item["config"]) for item in jobs}
        if not source_isolated(configs["dense"], dataset):
            raise ValueError(f"Fixed multi-window dev source is not isolated: {dataset}")
        dense_signature = comparison_signature(configs["dense"])
        if any(comparison_signature(config) != dense_signature for config in configs.values()):
            raise ValueError(f"Dense/policy generation configuration mismatch: {dataset}")
        dense_rows = {
            key(row): row
            for row in read_jsonl(root / "online" / dataset / "dense" / "01_trajectories.jsonl")
        }
        for job in [item for item in jobs if item["tag"] != "dense"]:
            run_dir = Path(job["run_dir"])
            policy_rows = {key(row): row for row in read_jsonl(run_dir / "01_trajectories.jsonl")}
            if set(dense_rows) != set(policy_rows):
                raise ValueError(f"Dense/policy task mismatch: {run_dir}")
            pairs = [(dense_rows[item], policy_rows[item]) for item in sorted(dense_rows)]
            if any(
                dense.get("question") != policy.get("question")
                or dense.get("gold") != policy.get("gold")
                or dense.get("prompt") != policy.get("prompt")
                for dense, policy in pairs
            ):
                raise ValueError(f"Dense/policy prompt mismatch: {run_dir}")
            flags = [(correct(dense), correct(policy)) for dense, policy in pairs]
            events = [selected_events(policy) for _dense, policy in pairs]
            dense_correct = sum(int(item[0]) for item in flags)
            policy_correct = sum(int(item[1]) for item in flags)
            flips = sum(int(dense and not policy) for dense, policy in flags)
            windows = sum(len(value) for value in events)
            examples_with_actions = sum(int(bool(value)) for value in events)
            configured_max = int(job["max_windows"])
            if any(len(value) > configured_max for value in events):
                raise ValueError(f"Runtime exceeded max_windows: {run_dir}")
            runtime_summary = read_json(run_dir / "00_runtime_summary.json")
            expected_fingerprint = config_fingerprint(
                configs[job["tag"]],
                ("seed", "model", "prompt", "data", "generation", "runtime_rasp"),
            )
            if runtime_summary.get("runtime_config_fingerprint") != expected_fingerprint:
                raise ValueError(f"Runtime result/config fingerprint mismatch: {run_dir}")
            completed_or_terminal_windows = sum(
                int(
                    bool(event["decision"].get("action_completed_or_terminal"))
                )
                for values in events
                for event in values
            )
            cells.append(
                {
                    "dataset": dataset,
                    "tag": job["tag"],
                    "ratio": job["ratio"],
                    "cadence_tokens": job["cadence_tokens"],
                    "max_windows": job["max_windows"],
                    "examples": len(pairs),
                    "dense_accuracy": dense_correct / len(pairs),
                    "policy_accuracy": policy_correct / len(pairs),
                    "paired_accuracy_delta": (policy_correct - dense_correct) / len(pairs),
                    "dense_correct_flips": flips,
                    "dense_correct_flip_rate": flips / max(1, dense_correct),
                    "executed_windows": windows,
                    "average_windows_per_problem": windows / len(pairs),
                    "examples_with_actions": examples_with_actions,
                    "action_execution_rate": examples_with_actions / len(pairs),
                    "completed_or_terminal_windows": completed_or_terminal_windows,
                    "average_theoretical_pruning_exposure": runtime_summary[
                        "average_decode_pruning_ratio"
                    ],
                    **bootstrap(flags),
                }
            )
    passing = []
    for tag in sorted({cell["tag"] for cell in cells}):
        values = [cell for cell in cells if cell["tag"] == tag]
        if len(values) == 2 and all(
            float(cell["average_theoretical_pruning_exposure"]) >= 0.02
            and float(cell["paired_accuracy_delta"]) >= -0.10
            and float(cell["dense_correct_flip_rate"]) <= 0.15
            and int(cell["executed_windows"])
            == int(cell["completed_or_terminal_windows"])
            for cell in values
        ):
            passing.append(tag)
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda tag: sum(
                float(cell["average_theoretical_pruning_exposure"])
                for cell in cells
                if cell["tag"] == tag
            )
            / 2,
        )
    write_json(
        root / "fixed_multi_window_summary.json",
        {
            "schema": "rasp_fixed_multi_window_dev_v1",
            "cells": cells,
            "passing_policies": passing,
            "selected_behavior_policy": selected,
            "multi_window_feasible": selected is not None,
            "logical_mask_only": True,
            "real_speedup_claimed": False,
            "final_test_sources_used": False,
        },
    )
    write_json(
        root / "selected_behavior_policy.json",
        {
            "schema": "rasp_selected_behavior_policy_v1",
            "selected_behavior_policy": selected,
            "allowed_for_on_policy_smoke": selected is not None,
        },
    )
    write_json(
        root / "phase_gate.json",
        {
            "schema": "rasp_fixed_multi_window_dev_gate_v1",
            "passed": selected is not None,
            "passing_policies": passing,
            "selected_behavior_policy": selected,
            "checks": {
                "two_source_policy_passes_risk_exposure_gate": selected is not None,
                "final_test_sources_unused": True,
                "logical_mask_only": True,
            },
            "next_step": (
                "Continue to on-policy smoke."
                if selected is not None
                else "Stop before on-policy collection."
            ),
        },
    )
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
