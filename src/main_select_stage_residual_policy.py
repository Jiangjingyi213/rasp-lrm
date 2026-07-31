from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_json


def _records(phase_root: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(phase_root.glob(f"seed_*/*/06_final/*/{method}_seed*.jsonl")):
        match = re.search(r"_seed(\d+)\.jsonl$", path.name)
        if match is None:
            raise RuntimeError(f"Cannot infer generation seed from {path}")
        seed = int(match.group(1))
        for row in read_jsonl(path):
            row["seed"] = seed
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No completed rows found for {method} below {phase_root}")
    return rows


def _metric(rows: list[dict[str, Any]], key: str) -> float:
    if key == "actual_pruning":
        return sum(
            float(row.get("runtime_stage_mask", {}).get("actual_average_mlp_pruning_ratio", 0.0))
            for row in rows
        ) / len(rows)
    if key == "fallback":
        return sum(
            int(not bool(row.get("stage_protocol", {}).get("valid"))) for row in rows
        ) / len(rows)
    if key == "truncation":
        return sum(int(bool(row.get("truncated"))) for row in rows) / len(rows)
    raise ValueError(f"Unknown metric {key}")


def _paired_delta(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[int]:
    left = {
        (str(row["dataset"]), int(row.get("seed", 0)), str(row["id"])): row
        for row in candidate
        if bool(row.get("stage_protocol", {}).get("valid"))
    }
    right = {
        (str(row["dataset"]), int(row.get("seed", 0)), str(row["id"])): row
        for row in baseline
        if bool(row.get("stage_protocol", {}).get("valid"))
    }
    common = sorted(set(left) & set(right))
    if not common:
        raise RuntimeError("No paired protocol-valid rows between candidate and baseline")
    return [int(left[key]["correct"]) - int(right[key]["correct"]) for key in common]


def _bootstrap_ci(deltas: list[int], repetitions: int = 5000) -> tuple[float, float]:
    rng = random.Random(20260731)
    size = len(deltas)
    means = sorted(sum(deltas[rng.randrange(size)] for _ in range(size)) / size for _ in range(repetitions))
    return means[int(0.025 * repetitions)], means[max(0, int(0.975 * repetitions) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the residual-prior dev selection gate.")
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="dynamic_global_activation")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["dynamic_stage_residual_025_activation", "dynamic_stage_residual_050_activation"],
    )
    args = parser.parse_args()

    phase_root = Path(args.phase_root)
    baseline = _records(phase_root, args.baseline)
    report: dict[str, Any] = {
        "schema": "stage_residual_policy_selection_v1",
        "phase_root": str(phase_root),
        "baseline": args.baseline,
        "candidates": {},
        "selection": None,
    }
    eligible: list[tuple[float, str, str]] = []
    baseline_fallback = _metric(baseline, "fallback")
    baseline_truncation = _metric(baseline, "truncation")
    for name in args.candidates:
        rows = _records(phase_root, name)
        valid_rows = [row for row in rows if bool(row.get("stage_protocol", {}).get("valid"))]
        delta = _paired_delta(rows, baseline)
        ci_low, ci_high = _bootstrap_ci(delta)
        accuracy = sum(int(row["correct"]) for row in valid_rows) / len(valid_rows)
        actual_pruning = _metric(rows, "actual_pruning")
        fallback = _metric(rows, "fallback")
        truncation = _metric(rows, "truncation")
        passes = (
            ci_low >= -0.005
            and 0.335 <= actual_pruning <= 0.345
            and fallback <= baseline_fallback + 0.01
            and truncation <= baseline_truncation + 0.01
        )
        report["candidates"][name] = {
            "protocol_valid_accuracy": accuracy,
            "paired_protocol_valid_examples": len(delta),
            "paired_delta_mean": sum(delta) / len(delta),
            "paired_bootstrap_ci_95": [ci_low, ci_high],
            "actual_pruning": actual_pruning,
            "fallback_rate": fallback,
            "truncation_rate": truncation,
            "passes": passes,
        }
        if passes:
            policy = "stage_residual_025" if "_025_" in name else "stage_residual_050"
            eligible.append((accuracy, name, policy))
    if eligible:
        _, name, policy = max(eligible)
        report["selection"] = {
            "method": name,
            "prior_policy": policy,
            "status": "passed",
        }
    else:
        report["selection"] = {"status": "failed", "reason": "no_residual_candidate_passed"}
    write_json(args.output, report)
    if report["selection"]["status"] != "passed":
        raise SystemExit("Stage-residual selection gate failed; output-aware and full runs are blocked.")


if __name__ == "__main__":
    main()
