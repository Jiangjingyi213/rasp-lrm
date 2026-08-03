from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import read_json, write_json


TARGET_MIN = 0.335
TARGET_MAX = 0.345
TARGET_CENTER = 0.34
DATASETS = ("gsm8k", "math500")


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
    if value is None and key == "protocol_valid_accuracy":
        value = row.get("accuracy")
    if value is None:
        return default
    return float(value)


def _pruning_ok(
    row: dict[str, Any] | None,
    *,
    target_min: float = TARGET_MIN,
    target_max: float = TARGET_MAX,
) -> bool:
    value = _metric(row, "actual_average_mlp_pruning_ratio")
    return value is not None and target_min <= value <= target_max


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


def _candidate_dataset_result(
    rows: dict[tuple[str, str, int], dict[str, Any]],
    *,
    dataset: str,
    seed: int,
    baseline_method: str,
    candidate_method: str,
    target_min: float = TARGET_MIN,
    target_max: float = TARGET_MAX,
    target_center: float = TARGET_CENTER,
    performance_reference_method: str | None = None,
) -> dict[str, Any]:
    baseline = rows.get((dataset, baseline_method, seed))
    candidate = rows.get((dataset, candidate_method, seed))
    performance_reference = (
        rows.get((dataset, performance_reference_method, seed))
        if performance_reference_method
        else None
    )
    actual = _metric(candidate, "actual_average_mlp_pruning_ratio")
    accuracy = _metric(candidate, "accuracy")
    protocol_accuracy = _metric(candidate, "protocol_valid_accuracy")
    baseline_accuracy = _metric(baseline, "accuracy")
    baseline_protocol_accuracy = _metric(baseline, "protocol_valid_accuracy")
    reference_accuracy = _metric(performance_reference, "accuracy")
    reference_protocol_accuracy = _metric(performance_reference, "protocol_valid_accuracy")
    fallback = _metric(candidate, "fallback_rate")
    baseline_fallback = _metric(baseline, "fallback_rate")
    truncation = _metric(candidate, "truncation_rate")
    baseline_truncation = _metric(baseline, "truncation_rate")
    return {
        "dataset": dataset,
        "seed": seed,
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "summary_found": candidate is not None,
        "baseline_found": baseline is not None,
        "performance_reference_method": performance_reference_method,
        "performance_reference_found": (
            performance_reference is not None if performance_reference_method else None
        ),
        "actual_pruning": actual,
        "target_pruning_range_passed": bool(
            actual is not None and target_min <= actual <= target_max
        ),
        "target_pruning_abs_gap": (
            abs(actual - target_center) if actual is not None else None
        ),
        "target_pruning_min": target_min,
        "target_pruning_max": target_max,
        "target_pruning_center": target_center,
        "accuracy": accuracy,
        "protocol_valid_accuracy": protocol_accuracy,
        "accuracy_delta_vs_baseline": (
            accuracy - baseline_accuracy
            if accuracy is not None and baseline_accuracy is not None
            else None
        ),
        "protocol_valid_accuracy_delta_vs_baseline": (
            protocol_accuracy - baseline_protocol_accuracy
            if protocol_accuracy is not None and baseline_protocol_accuracy is not None
            else None
        ),
        "accuracy_delta_vs_performance_reference": (
            accuracy - reference_accuracy
            if accuracy is not None and reference_accuracy is not None
            else None
        ),
        "protocol_valid_accuracy_delta_vs_performance_reference": (
            protocol_accuracy - reference_protocol_accuracy
            if protocol_accuracy is not None and reference_protocol_accuracy is not None
            else None
        ),
        "fallback_rate": fallback,
        "fallback_delta_vs_baseline": (
            fallback - baseline_fallback
            if fallback is not None and baseline_fallback is not None
            else None
        ),
        "truncation_rate": truncation,
        "truncation_delta_vs_baseline": (
            truncation - baseline_truncation
            if truncation is not None and baseline_truncation is not None
            else None
        ),
    }


def _phase_a2(
    rows: dict[tuple[str, str, int], dict[str, Any]],
    *,
    seed: int,
    datasets: tuple[str, ...],
    baseline_method: str,
    candidate_methods: tuple[str, ...],
    mode: str,
    phase_label: str = "A2_budget_v2",
    target_min: float = TARGET_MIN,
    target_max: float = TARGET_MAX,
    target_center: float = TARGET_CENTER,
    performance_reference_method: str | None = None,
    max_performance_reference_accuracy_drop: float | None = None,
    max_fallback_delta: float | None = None,
    max_truncation_delta: float | None = None,
) -> dict[str, Any]:
    candidates = []
    for method_name in candidate_methods:
        dataset_results = [
            _candidate_dataset_result(
                rows,
                dataset=dataset,
                seed=seed,
                baseline_method=baseline_method,
                candidate_method=method_name,
                target_min=target_min,
                target_max=target_max,
                target_center=target_center,
                performance_reference_method=performance_reference_method,
            )
            for dataset in datasets
        ]
        target_ok = all(row["target_pruning_range_passed"] for row in dataset_results)
        avg_protocol = _mean(
            row["protocol_valid_accuracy"] for row in dataset_results
        )
        avg_accuracy_delta = _mean(
            row["accuracy_delta_vs_baseline"] for row in dataset_results
        )
        avg_target_gap = _mean(row["target_pruning_abs_gap"] for row in dataset_results)
        observed_max_fallback_delta = _max_value(
            row["fallback_delta_vs_baseline"] for row in dataset_results
        )
        observed_max_truncation_delta = _max_value(
            row["truncation_delta_vs_baseline"] for row in dataset_results
        )
        avg_reference_delta = _mean(
            row["accuracy_delta_vs_performance_reference"] for row in dataset_results
        )
        reference_ok = True
        if max_performance_reference_accuracy_drop is not None:
            reference_ok = bool(
                avg_reference_delta is not None
                and avg_reference_delta >= -float(max_performance_reference_accuracy_drop)
            )
        fallback_ok = bool(
            max_fallback_delta is None
            or (
                observed_max_fallback_delta is not None
                and observed_max_fallback_delta <= float(max_fallback_delta)
            )
        )
        truncation_ok = bool(
            max_truncation_delta is None
            or (
                observed_max_truncation_delta is not None
                and observed_max_truncation_delta <= float(max_truncation_delta)
            )
        )
        if mode in {"smoke", "confirm"}:
            passed = bool(target_ok and reference_ok and fallback_ok and truncation_ok)
        else:
            passed = bool(
                target_ok
                and reference_ok
                and fallback_ok
                and truncation_ok
                and avg_accuracy_delta is not None
                and avg_accuracy_delta >= 0.01
                and all(
                    row["accuracy_delta_vs_baseline"] is not None
                    and row["accuracy_delta_vs_baseline"] >= -0.005
                    for row in dataset_results
                )
                and observed_max_fallback_delta is not None
                and observed_max_fallback_delta <= 0.01
                and observed_max_truncation_delta is not None
                and observed_max_truncation_delta <= 0.01
            )
        candidates.append(
            {
                "method": method_name,
                "passed": passed,
                "target_pruning_range_passed": target_ok,
                "avg_protocol_valid_accuracy": avg_protocol,
                "avg_accuracy_delta_vs_baseline": avg_accuracy_delta,
                "avg_accuracy_delta_vs_performance_reference": avg_reference_delta,
                "avg_target_pruning_abs_gap": avg_target_gap,
                "max_fallback_delta_vs_baseline": observed_max_fallback_delta,
                "max_truncation_delta_vs_baseline": observed_max_truncation_delta,
                "datasets": dataset_results,
            }
        )
    passing = [row for row in candidates if row["passed"]]
    target_ok_candidates = [row for row in candidates if row["target_pruning_range_passed"]]
    selection_pool = passing or target_ok_candidates or candidates
    selected = max(
        selection_pool,
        key=lambda row: (
            row["avg_protocol_valid_accuracy"]
            if row["avg_protocol_valid_accuracy"] is not None
            else -1.0,
            -(
                row["avg_target_pruning_abs_gap"]
                if row["avg_target_pruning_abs_gap"] is not None
                else 999.0
            ),
        ),
    ) if selection_pool else None
    if not (passing or target_ok_candidates) and candidates:
        selected = min(
            candidates,
            key=lambda row: (
                row["avg_target_pruning_abs_gap"]
                if row["avg_target_pruning_abs_gap"] is not None
                else 999.0
            ),
        )
    return {
        "phase": f"{phase_label}_{mode}",
        "phase_passed": bool(passing),
        "selection_mode": mode,
        "seed": seed,
        "datasets": list(datasets),
        "baseline_method": baseline_method,
        "target_min": target_min,
        "target_max": target_max,
        "target_center": target_center,
        "performance_reference_method": performance_reference_method,
        "max_performance_reference_accuracy_drop": max_performance_reference_accuracy_drop,
        "max_fallback_delta": max_fallback_delta,
        "max_truncation_delta": max_truncation_delta,
        "selected_candidate_method": selected["method"] if selected else None,
        "selected_candidate_passed": bool(selected and selected["passed"]),
        "selected_for_diagnosis_only": bool(selected and not selected["passed"]),
        "candidates": candidates,
    }


def _mean(values: Any) -> float | None:
    prepared = [float(value) for value in values if value is not None]
    return sum(prepared) / len(prepared) if prepared else None


def _max_value(values: Any) -> float | None:
    prepared = [float(value) for value in values if value is not None]
    return max(prepared) if prepared else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b", "c", "a2"], required=True)
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--reference-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--nondegrade-tolerance", type=float, default=0.005)
    parser.add_argument("--selection-mode", choices=["smoke", "dev", "confirm"], default="dev")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--baseline-method", default="dynamic_global_activation_fixed_t30")
    parser.add_argument("--candidate-methods", nargs="+", default=[])
    parser.add_argument("--phase-label", default="A2_budget_v2")
    parser.add_argument("--target-min", type=float, default=TARGET_MIN)
    parser.add_argument("--target-max", type=float, default=TARGET_MAX)
    parser.add_argument("--target-center", type=float, default=TARGET_CENTER)
    parser.add_argument("--performance-reference-method", default=None)
    parser.add_argument("--max-performance-reference-accuracy-drop", type=float, default=None)
    parser.add_argument("--max-fallback-delta", type=float, default=None)
    parser.add_argument("--max-truncation-delta", type=float, default=None)
    args = parser.parse_args()

    rows = _summary_rows(*(Path(value) for value in args.reference_root), Path(args.phase_root))
    if args.phase == "a":
        selection = _phase_a(rows, args.nondegrade_tolerance)
    elif args.phase == "b":
        selection = _phase_b(rows, args.nondegrade_tolerance)
    elif args.phase == "a2":
        if not args.candidate_methods:
            raise ValueError("--candidate-methods is required for phase a2")
        selection = _phase_a2(
            rows,
            seed=int(args.seed),
            datasets=tuple(str(value) for value in args.datasets),
            baseline_method=str(args.baseline_method),
            candidate_methods=tuple(str(value) for value in args.candidate_methods),
            mode=str(args.selection_mode),
            phase_label=str(args.phase_label),
            target_min=float(args.target_min),
            target_max=float(args.target_max),
            target_center=float(args.target_center),
            performance_reference_method=(
                str(args.performance_reference_method)
                if args.performance_reference_method
                else None
            ),
            max_performance_reference_accuracy_drop=args.max_performance_reference_accuracy_drop,
            max_fallback_delta=args.max_fallback_delta,
            max_truncation_delta=args.max_truncation_delta,
        )
    else:
        selection = _phase_c(rows, args.nondegrade_tolerance)
    selection["phase_root"] = str(Path(args.phase_root))
    selection["summary_count"] = len(rows)
    write_json(args.output, selection)
    print(f"{selection['phase']}: passed={selection['phase_passed']} output={args.output}")


if __name__ == "__main__":
    main()
