from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io import read_json, write_json


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        raise ValueError(f"Missing {field} in development summaries")
    return sum(values) / len(values)


def _rows(paths: list[str], method: str) -> list[dict[str, Any]]:
    selected = []
    for path in paths:
        summary = read_json(path)
        for row in summary.get("methods", []):
            if row.get("method", {}).get("name") == method:
                selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen Stage-Risk Adaptive v1 dev gate.")
    parser.add_argument("--dev-summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default="t30_math_safe_replay")
    parser.add_argument("--candidate", default="stage_risk_adaptive_v1")
    parser.add_argument("--required-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--min-pruning", type=float, default=0.335)
    parser.add_argument("--max-pruning", type=float, default=0.345)
    parser.add_argument("--min-protocol-valid-gain", type=float, default=0.015)
    args = parser.parse_args()

    reference = _rows(args.dev_summaries, args.reference)
    candidate = _rows(args.dev_summaries, args.candidate)
    required = set(args.required_seeds)
    reference_seeds = {int(row["seed"]) for row in reference}
    candidate_seeds = {int(row["seed"]) for row in candidate}
    if reference_seeds != required or candidate_seeds != required:
        raise ValueError(
            f"Expected exactly seeds {sorted(required)}; got reference={sorted(reference_seeds)}, "
            f"candidate={sorted(candidate_seeds)}"
        )
    reference_by_seed = {int(row["seed"]): row for row in reference}
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    per_seed = []
    for seed in sorted(required):
        base, proposal = reference_by_seed[seed], candidate_by_seed[seed]
        pruning = float(proposal["actual_average_mlp_pruning_ratio"])
        per_seed.append(
            {
                "seed": seed,
                "actual_pruning": pruning,
                "protocol_valid_gain": float(proposal["protocol_valid_accuracy"])
                - float(base["protocol_valid_accuracy"]),
                "overall_accuracy_gain": float(proposal["accuracy"]) - float(base["accuracy"]),
                "fallback_delta": float(proposal["fallback_rate"]) - float(base["fallback_rate"]),
                "truncation_delta": float(proposal["truncation_rate"]) - float(base["truncation_rate"]),
                "pruning_in_range": args.min_pruning <= pruning <= args.max_pruning,
            }
        )
    aggregate = {
        "actual_pruning": _mean(candidate, "actual_average_mlp_pruning_ratio"),
        "protocol_valid_gain": _mean(candidate, "protocol_valid_accuracy") - _mean(reference, "protocol_valid_accuracy"),
        "overall_accuracy_gain": _mean(candidate, "accuracy") - _mean(reference, "accuracy"),
        "fallback_delta": _mean(candidate, "fallback_rate") - _mean(reference, "fallback_rate"),
        "truncation_delta": _mean(candidate, "truncation_rate") - _mean(reference, "truncation_rate"),
    }
    passed = bool(
        all(row["pruning_in_range"] for row in per_seed)
        and aggregate["protocol_valid_gain"] >= args.min_protocol_valid_gain
        and aggregate["overall_accuracy_gain"] >= 0.0
        and aggregate["fallback_delta"] <= 0.0
        and aggregate["truncation_delta"] <= 0.0
    )
    write_json(
        Path(args.output),
        {
            "schema": "stage_risk_adaptive_dev_gate_v1",
            "status": "passed" if passed else "failed",
            "full_evaluation_allowed": passed,
            "reference": args.reference,
            "candidate": args.candidate,
            "required_seeds": sorted(required),
            "thresholds": {
                "actual_pruning": [args.min_pruning, args.max_pruning],
                "min_protocol_valid_gain": args.min_protocol_valid_gain,
                "minimum_overall_accuracy_gain": 0.0,
                "maximum_fallback_delta": 0.0,
                "maximum_truncation_delta": 0.0,
            },
            "aggregate": aggregate,
            "per_seed": per_seed,
            "input_summaries": args.dev_summaries,
        },
    )


if __name__ == "__main__":
    main()
