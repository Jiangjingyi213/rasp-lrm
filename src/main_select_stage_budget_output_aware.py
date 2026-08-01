from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import read_json, write_json


TARGET_MIN = 0.335
TARGET_MAX = 0.345


def _summary_rows(*roots: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for root in roots:
        for summary_path in sorted(root.glob("seed_*/*/06_final/summary.json")):
            seed_name = summary_path.parents[2].name
            try:
                seed = int(seed_name.removeprefix("seed_"))
            except ValueError:
                continue
            data = read_json(summary_path)
            for dataset, summaries in data.get("datasets", {}).items():
                for summary in summaries:
                    method = str(summary.get("method", {}).get("name", ""))
                    if method:
                        rows[(str(dataset), method, seed)] = summary
    return rows


def _metric(row: dict[str, Any] | None, key: str, default: float | None = None) -> float | None:
    if row is None:
        return default
    value = row.get(key)
    return None if value is None else float(value)


def _pruning_ok(row: dict[str, Any] | None) -> bool:
    value = _metric(row, "actual_average_mlp_pruning_ratio")
    return value is not None and TARGET_MIN <= value <= TARGET_MAX


def _nondegrade(
    candidate: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    key: str,
    *,
    tolerance: float = 0.0,
    larger_is_better: bool = True,
) -> bool:
    candidate_value = _metric(candidate, key)
    baseline_value = _metric(baseline, key)
    if candidate_value is None or baseline_value is None:
        return False
    if larger_is_better:
        return candidate_value + tolerance >= baseline_value
    return candidate_value <= baseline_value + tolerance


def _dataset_result(
    rows: dict[tuple[str, str, int], dict[str, Any]],
    *,
    dataset: str,
    seed: int,
    baseline_method: str,
    candidate_method: str,
    tolerance: float,
    require_plus: float = 0.0,
) -> dict[str, Any]:
    baseline = rows.get((dataset, baseline_method, seed))
    candidate = rows.get((dataset, candidate_method, seed))
    protocol_delta = (
        None
        if baseline is None or candidate is None
        else _metric(candidate, "protocol_valid_accuracy", 0.0)
        - _metric(baseline, "protocol_valid_accuracy", 0.0)
    )
    accuracy_delta = (
        None
        if baseline is None or candidate is None
        else _metric(candidate, "accuracy", 0.0) - _metric(baseline, "accuracy", 0.0)
    )
    passed = (
        baseline is not None
        and candidate is not None
        and _pruning_ok(candidate)
        and protocol_delta is not None
        and protocol_delta + tolerance >= require_plus
        and _nondegrade(
            candidate,
            baseline,
            "fallback_rate",
            tolerance=0.01,
            larger_is_better=False,
        )
        and _nondegrade(
            candidate,
            baseline,
            "truncation_rate",
            tolerance=0.01,
            larger_is_better=False,
        )
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "passed": bool(passed),
        "protocol_valid_accuracy_delta": protocol_delta,
        "accuracy_delta": accuracy_delta,
        "candidate_actual_pruning": _metric(candidate, "actual_average_mlp_pruning_ratio"),
        "baseline_protocol_valid_accuracy": _metric(baseline, "protocol_valid_accuracy"),
        "candidate_protocol_valid_accuracy": _metric(candidate, "protocol_valid_accuracy"),
        "baseline_fallback_rate": _metric(baseline, "fallback_rate"),
        "candidate_fallback_rate": _metric(candidate, "fallback_rate"),
        "baseline_truncation_rate": _metric(baseline, "truncation_rate"),
        "candidate_truncation_rate": _metric(candidate, "truncation_rate"),
    }


def _phase_a(rows: dict[tuple[str, str, int], dict[str, Any]], tolerance: float) -> dict[str, Any]:
    results = [
        _dataset_result(
            rows,
            dataset=dataset,
            seed=3,
            baseline_method="dynamic_global_activation_fixed_t30",
            candidate_method="dynamic_global_activation_budgeted",
            tolerance=tolerance,
        )
        for dataset in ("gsm8k", "math500")
    ]
    return {
        "phase": "A_budget_only",
        "phase_passed": all(row["passed"] for row in results),
        "results": results,
    }


def _phase_b(rows: dict[tuple[str, str, int], dict[str, Any]], tolerance: float) -> dict[str, Any]:
    global_results = [
        _dataset_result(
            rows,
            dataset=dataset,
            seed=3,
            baseline_method="dynamic_global_activation_fixed_t30",
            candidate_method="dynamic_global_output_aware_fixed_t30",
            tolerance=tolerance,
        )
        for dataset in ("gsm8k", "math500")
    ]
    stage_results = [
        _dataset_result(
            rows,
            dataset=dataset,
            seed=3,
            baseline_method="current_t30_math_safe",
            candidate_method="current_t30_stage_output_aware",
            tolerance=tolerance,
        )
        for dataset in ("gsm8k", "math500")
    ]
    global_passed = all(row["passed"] for row in global_results)
    stage_passed = all(row["passed"] for row in stage_results)
    return {
        "phase": "B_output_aware_only",
        "phase_passed": bool(global_passed or stage_passed),
        "global_output_aware_passed": bool(global_passed),
        "stage_output_aware_passed": bool(stage_passed),
        "selected_prior_for_phase_c": "trajectory_global" if global_passed else None,
        "results": {
            "global": global_results,
            "stage": stage_results,
        },
    }


def _phase_c(rows: dict[tuple[str, str, int], dict[str, Any]], tolerance: float) -> dict[str, Any]:
    results = [
        _dataset_result(
            rows,
            dataset=dataset,
            seed=3,
            baseline_method="dynamic_global_activation_fixed_t30",
            candidate_method="dynamic_global_output_aware_budgeted",
            tolerance=tolerance,
            require_plus=0.01,
        )
        for dataset in ("gsm8k", "math500")
    ]
    total_accuracy_ok = all(
        row["accuracy_delta"] is not None and row["accuracy_delta"] + tolerance >= 0.0
        for row in results
    )
    return {
        "phase": "C_combined",
        "phase_passed": bool(all(row["passed"] for row in results) and total_accuracy_ok),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b", "c"], required=True)
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--reference-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--nondegrade-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    rows = _summary_rows(*(Path(value) for value in args.reference_root), Path(args.phase_root))
    if args.phase == "a":
        selection = _phase_a(rows, args.nondegrade_tolerance)
    elif args.phase == "b":
        selection = _phase_b(rows, args.nondegrade_tolerance)
    else:
        selection = _phase_c(rows, args.nondegrade_tolerance)
    selection["phase_root"] = str(Path(args.phase_root))
    selection["summary_count"] = len(rows)
    write_json(args.output, selection)
    print(f"{selection['phase']}: passed={selection['phase_passed']} output={args.output}")


if __name__ == "__main__":
    main()
