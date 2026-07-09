from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.stage_calibration.artifacts import file_sha256, stable_hash
from src.stage_calibration.protocol import STAGES
from src.utils.io import ensure_dir, read_json, write_json


DEFAULT_THRESHOLDS = {
    "minimum_protocol_rate": 0.90,
    "maximum_fallback_rate": 0.10,
    "maximum_truncation_rate": 0.05,
    "formal_safe_mean_drop": 0.05,
    "formal_safe_worst_drop": 0.10,
    "main_candidate_mean_drop": 0.08,
    "main_candidate_worst_drop": 0.15,
    "aggressive_boundary_min_pruning": 0.20,
    "aggressive_boundary_mean_drop": 0.12,
    "conservative_min_pruning": 0.08,
    "conservative_max_pruning": 0.15,
    "main_dynamic_min_pruning": 0.15,
    "main_dynamic_max_pruning": 0.25,
    "aggressive_selection_min_pruning": 0.30,
    "main_only_max_drop": 0.05,
}

STAGE_BUDGET_PRESETS = {
    "stage_budget_conservative": {
        "setup": 0.0,
        "reasoning": 0.0,
        "verify": 0.2,
        "final": 0.2,
    },
    "stage_budget_balanced": {
        "setup": 0.1,
        "reasoning": 0.1,
        "verify": 0.2,
        "final": 0.2,
    },
    "stage_budget_aggressive": {
        "setup": 0.2,
        "reasoning": 0.2,
        "verify": 0.3,
        "final": 0.3,
    },
}


def load_seed_run(path: str | Path) -> dict[str, Any]:
    root_or_summary = Path(path)
    if root_or_summary.name == "summary.json":
        summary_path = root_or_summary
        root = summary_path.parent.parent
    else:
        root = root_or_summary
        summary_path = root / "05_dev" / "summary.json"
    frozen_path = summary_path.parent / "frozen_policy.json"
    summary = read_json(summary_path)
    frozen = read_json(frozen_path) if frozen_path.exists() else None
    structured = _find_method(summary["methods"], "structured_dense")
    ordinary = _find_method(summary["methods"], "ordinary_dense")
    return {
        "root": str(root),
        "seed": int(structured.get("seed", summary.get("seed", 0))),
        "summary_path": str(summary_path),
        "frozen_policy_path": str(frozen_path) if frozen_path.exists() else None,
        "summary_sha256": file_sha256(summary_path),
        "frozen_policy_sha256": file_sha256(frozen_path) if frozen_path.exists() else None,
        "summary": summary,
        "frozen_policy": frozen,
        "ordinary_dense_accuracy": ordinary.get("accuracy"),
        "structured_dense_accuracy": structured.get("accuracy"),
        "prompt_gate_passed": bool(summary.get("prompt_gate", {}).get("passed")),
    }


def build_policy_selection(
    seed_runs: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
    include_stage_budget_presets: bool = True,
) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    aggregates = aggregate_methods(seed_runs, thresholds)
    selected = select_downstream_policies(
        aggregates,
        seed_runs=seed_runs,
        thresholds=thresholds,
        include_stage_budget_presets=include_stage_budget_presets,
    )
    manifest = [
        {
            "root": run["root"],
            "seed": run["seed"],
            "summary_path": run["summary_path"],
            "summary_sha256": run["summary_sha256"],
            "frozen_policy_path": run["frozen_policy_path"],
            "frozen_policy_sha256": run["frozen_policy_sha256"],
            "ordinary_dense_accuracy": run["ordinary_dense_accuracy"],
            "structured_dense_accuracy": run["structured_dense_accuracy"],
            "prompt_gate_passed": run["prompt_gate_passed"],
        }
        for run in seed_runs
    ]
    return {
        "schema": "stage_policy_selection_v1",
        "input_manifest_hash": stable_hash(manifest),
        "seed_runs": manifest,
        "thresholds": thresholds,
        "test_sets_consulted": False,
        "selection_policy": {
            "dense_reference": "structured_dense",
            "shuffled_control_may_not_be_main": True,
            "downstream_results_must_not_change_selection": True,
        },
        "method_aggregates": aggregates,
        "selected_policies": selected["selected_policies"],
        "downstream_methods": selected["downstream_methods"],
    }


def build_main_only_policy_selection(
    seed_runs: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
    candidate_ratios: tuple[float, ...] = (0.10, 0.15, 0.20),
) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    aggregates = aggregate_methods(seed_runs, thresholds)
    structured = _aggregate_by_name(aggregates, "structured_dense")
    if structured is None:
        raise ValueError("Missing structured_dense summary for main-only policy selection")
    dynamic = _pick_main_only_dynamic(aggregates, thresholds, candidate_ratios)
    if dynamic is None:
        raise ValueError("No stage_specific candidate passed main-only ratio gates")
    dynamic_method = _method_with_selection_metadata(
        {
            **dynamic["method"],
            "name": "dynamic_stage_main",
        },
        role="dynamic_stage_main",
        policy_id=f"main_only_dynamic:{dynamic['method_name']}:{stable_hash(dynamic['method'])[:12]}",
        tier=dynamic["selection_tier"],
        source="single_seed_dev_main_only",
        diagnostic_only=False,
    )
    dynamic_method["original_method_name"] = dynamic["method_name"]
    static_ratio = _average_ratio(dynamic_method["stage_ratios"])
    structured_prompt = dict(structured["method"].get("prompt", {}))
    static_method = _method_with_selection_metadata(
        {
            "name": "static_matched_global",
            "policy": "trajectory_global",
            "stage_ratios": {stage: static_ratio for stage in STAGES},
            "prompt": structured_prompt,
            "bias_compensation": True,
        },
        role="static_matched_global",
        policy_id=f"main_only_static:trajectory_global_{static_ratio:.4f}:{stable_hash(dynamic_method)[:12]}",
        tier="matched_baseline",
        source="matched_to_dynamic_stage_main",
        diagnostic_only=False,
    )
    structured_method = _method_with_selection_metadata(
        structured["method"],
        role="structured_dense",
        policy_id=f"main_only_structured:{stable_hash(structured['method'])[:12]}",
        tier="baseline",
        source="single_seed_dev_main_only",
        diagnostic_only=False,
    )
    manifest = _seed_run_manifest(seed_runs)
    return {
        "schema": "stage_policy_selection_v1",
        "selection_mode": "main_only",
        "input_manifest_hash": stable_hash(manifest),
        "seed_runs": manifest,
        "thresholds": thresholds,
        "candidate_ratios": [float(value) for value in candidate_ratios],
        "test_sets_consulted": False,
        "selection_policy": {
            "dense_reference": "structured_dense",
            "dynamic_policy": "stage_specific",
            "dynamic_ratio_rule": "highest_candidate_ratio_passing_dev_gates",
            "static_baseline": "trajectory_global at arithmetic mean of dynamic stage ratios",
            "downstream_results_must_not_change_selection": True,
        },
        "method_aggregates": aggregates,
        "selected_policies": {
            "structured_dense": {
                "role": "structured_dense",
                "policy_id": structured_method["selection_policy_id"],
                "selection_tier": "baseline",
                "selection_source": "single_seed_dev_main_only",
                "reason": "Explicit-stage dense reference for full evaluation.",
                "aggregate": structured,
                "method": structured_method,
                "diagnostic_only": False,
            },
            "dynamic_stage_main": {
                "role": "dynamic_stage_main",
                "policy_id": dynamic_method["selection_policy_id"],
                "selection_tier": dynamic["selection_tier"],
                "selection_source": "single_seed_dev_main_only",
                "reason": "Highest stage-specific candidate ratio passing main-only dev gates.",
                "aggregate": dynamic,
                "method": dynamic_method,
                "diagnostic_only": False,
            },
            "static_matched_global": {
                "role": "static_matched_global",
                "policy_id": static_method["selection_policy_id"],
                "selection_tier": "matched_baseline",
                "selection_source": "matched_to_dynamic_stage_main",
                "reason": "Static trajectory-global baseline at the dynamic policy average ratio.",
                "matched_dynamic_method": dynamic["method_name"],
                "matched_average_ratio": static_ratio,
                "method": static_method,
                "diagnostic_only": False,
            },
        },
        "downstream_methods": [structured_method, dynamic_method, static_method],
    }


def _seed_run_manifest(seed_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "root": run["root"],
            "seed": run["seed"],
            "summary_path": run["summary_path"],
            "summary_sha256": run["summary_sha256"],
            "frozen_policy_path": run["frozen_policy_path"],
            "frozen_policy_sha256": run["frozen_policy_sha256"],
            "ordinary_dense_accuracy": run["ordinary_dense_accuracy"],
            "structured_dense_accuracy": run["structured_dense_accuracy"],
            "prompt_gate_passed": run["prompt_gate_passed"],
        }
        for run in seed_runs
    ]


def aggregate_methods(
    seed_runs: list[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    grouped: dict[str, list[dict[str, Any]]] = {}
    expected_seed_count = len(seed_runs)
    for run in seed_runs:
        dense = _find_method(run["summary"]["methods"], "structured_dense")
        dense_accuracy = float(dense["accuracy"])
        for summary in run["summary"]["methods"]:
            method = dict(summary["method"])
            name = str(method["name"])
            grouped.setdefault(name, []).append(
                {
                    "seed": int(summary.get("seed", run["seed"])),
                    "accuracy": _float(summary.get("accuracy")),
                    "accuracy_drop_vs_structured_dense": dense_accuracy - _float(summary.get("accuracy")),
                    "valid_stage_protocol_rate": _float(summary.get("valid_stage_protocol_rate")),
                    "fallback_rate": _float(summary.get("fallback_rate")),
                    "truncation_rate": _float(summary.get("truncation_rate")),
                    "theoretical_average_mlp_pruning_ratio": _float(
                        summary.get("theoretical_average_mlp_pruning_ratio")
                    ),
                    "problems": int(summary.get("problems", 0)),
                    "correct": int(summary.get("correct", 0)),
                    "method_hash": stable_hash(_method_identity(method)),
                    "method": method,
                }
            )

    aggregates = []
    for name, rows in sorted(grouped.items()):
        method = dict(rows[0]["method"])
        method_hashes = sorted({row["method_hash"] for row in rows})
        drops = [row["accuracy_drop_vs_structured_dense"] for row in rows]
        protocols = [row["valid_stage_protocol_rate"] for row in rows]
        fallbacks = [row["fallback_rate"] for row in rows]
        truncations = [row["truncation_rate"] for row in rows]
        prunings = [row["theoretical_average_mlp_pruning_ratio"] for row in rows]
        aggregate = {
            "method_name": name,
            "method": method,
            "method_identity_hashes": method_hashes,
            "method_identity_consistent": len(method_hashes) == 1,
            "policy": method.get("policy"),
            "stage_ratios": method.get("stage_ratios", {}),
            "seed_count": len(rows),
            "expected_seed_count": expected_seed_count,
            "seeds": [row["seed"] for row in rows],
            "mean_accuracy": _mean(row["accuracy"] for row in rows),
            "mean_accuracy_drop_vs_structured_dense": _mean(drops),
            "worst_accuracy_drop_vs_structured_dense": max(drops) if drops else None,
            "best_accuracy_drop_vs_structured_dense": min(drops) if drops else None,
            "mean_valid_stage_protocol_rate": _mean(protocols),
            "min_valid_stage_protocol_rate": min(protocols) if protocols else None,
            "mean_fallback_rate": _mean(fallbacks),
            "max_fallback_rate": max(fallbacks) if fallbacks else None,
            "mean_truncation_rate": _mean(truncations),
            "max_truncation_rate": max(truncations) if truncations else None,
            "mean_theoretical_average_mlp_pruning_ratio": _mean(prunings),
            "per_seed": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"method", "method_hash"}
                }
                for row in rows
            ],
        }
        aggregate.update(_classify_aggregate(aggregate, thresholds))
        aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda row: (
            row["tier_rank"],
            -float(row.get("mean_theoretical_average_mlp_pruning_ratio") or 0.0),
            float(row.get("mean_accuracy_drop_vs_structured_dense") or 0.0),
            row["method_name"],
        ),
    )


def select_downstream_policies(
    aggregates: list[dict[str, Any]],
    *,
    seed_runs: list[dict[str, Any]],
    thresholds: dict[str, float],
    include_stage_budget_presets: bool,
) -> dict[str, Any]:
    structured = _aggregate_by_name(aggregates, "structured_dense")
    ordinary = _aggregate_by_name(aggregates, "ordinary_dense")
    structured_prompt = dict(structured["method"].get("prompt", {})) if structured else {}
    selected: dict[str, Any] = {
        "ordinary_dense": _selection_entry("ordinary_dense", ordinary, "baseline") if ordinary else None,
        "structured_dense": _selection_entry("structured_dense", structured, "baseline") if structured else None,
    }

    selected["conservative"] = _pick_conservative(aggregates, thresholds)
    selected["main_dynamic"] = _pick_main_dynamic(aggregates, thresholds)
    selected["aggressive"] = _pick_aggressive(aggregates, thresholds)
    selected["shuffled_control"] = _pick_shuffled_control(aggregates, thresholds)

    stage_budget_presets = []
    if include_stage_budget_presets:
        for name, ratios in STAGE_BUDGET_PRESETS.items():
            role = name.replace("stage_budget_", "stage_budget_")
            stage_budget_presets.append(
                {
                    "role": role,
                    "policy_id": f"predeclared:{name}",
                    "selection_tier": "predeclared",
                    "selection_source": "predeclared_stage_budget_grid",
                    "reason": (
                        "Fixed before downstream evaluation; included to test whether "
                        "stage-conditioned ratios beat uniform policies."
                    ),
                    "method": _method_with_selection_metadata(
                        {
                            "name": name,
                            "policy": "stage_specific",
                            "stage_ratios": {stage: float(ratios.get(stage, 0.0)) for stage in STAGES},
                            "prompt": structured_prompt,
                            "bias_compensation": True,
                        },
                        role=role,
                        policy_id=f"predeclared:{name}",
                        tier="predeclared",
                        source="predeclared_stage_budget_grid",
                        diagnostic_only=name.endswith("aggressive"),
                    ),
                }
            )
    selected["stage_budget_presets"] = stage_budget_presets

    downstream_methods = []
    for entry in (
        selected.get("ordinary_dense"),
        selected.get("structured_dense"),
        selected.get("conservative"),
        selected.get("main_dynamic"),
        selected.get("aggressive"),
        selected.get("shuffled_control"),
    ):
        if entry and entry.get("method"):
            downstream_methods.append(entry["method"])
    downstream_methods.extend(entry["method"] for entry in stage_budget_presets)
    return {
        "selected_policies": selected,
        "downstream_methods": _dedupe_methods(downstream_methods),
    }


def write_policy_selection_markdown(selection: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Cross-Seed Stage Policy Selection",
        "",
        f"- schema: `{selection['schema']}`",
        f"- input_manifest_hash: `{selection['input_manifest_hash']}`",
        f"- test_sets_consulted: `{selection['test_sets_consulted']}`",
        "",
        "## Seed Runs",
        "",
        "| seed | prompt gate | ordinary dense | structured dense | root |",
        "|---:|---|---:|---:|---|",
    ]
    for run in selection["seed_runs"]:
        lines.append(
            "| {seed} | {gate} | {ordinary:.4f} | {structured:.4f} | `{root}` |".format(
                seed=run["seed"],
                gate="pass" if run["prompt_gate_passed"] else "fail",
                ordinary=float(run["ordinary_dense_accuracy"]),
                structured=float(run["structured_dense_accuracy"]),
                root=run["root"],
            )
        )
    lines.extend(
        [
            "",
            "## Selected Policies",
            "",
            "| role | method | tier | mean drop | worst drop | pruning | note |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for role, entry in selection["selected_policies"].items():
        if role == "stage_budget_presets":
            continue
        if not entry:
            lines.append(f"| {role} | - | - | - | - | - | not selected |")
            continue
        aggregate = entry.get("aggregate") or {}
        lines.append(
            "| {role} | `{method}` | {tier} | {mean_drop} | {worst_drop} | {pruning} | {note} |".format(
                role=role,
                method=entry["method"]["name"],
                tier=entry.get("selection_tier", "-"),
                mean_drop=_fmt_pct(aggregate.get("mean_accuracy_drop_vs_structured_dense")),
                worst_drop=_fmt_pct(aggregate.get("worst_accuracy_drop_vs_structured_dense")),
                pruning=_fmt_pct(aggregate.get("mean_theoretical_average_mlp_pruning_ratio")),
                note=entry.get("reason", ""),
            )
        )
    for entry in selection["selected_policies"].get("stage_budget_presets", []):
        lines.append(
            "| {role} | `{method}` | predeclared | - | - | - | {reason} |".format(
                role=entry["role"],
                method=entry["method"]["name"],
                reason=entry["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            "| method | tier | mean drop | worst drop | min protocol | max fallback | max trunc | pruning |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selection["method_aggregates"]:
        lines.append(
            "| `{method}` | {tier} | {mean_drop} | {worst_drop} | {protocol} | {fallback} | {trunc} | {pruning} |".format(
                method=row["method_name"],
                tier=row["selection_tier"],
                mean_drop=_fmt_pct(row.get("mean_accuracy_drop_vs_structured_dense")),
                worst_drop=_fmt_pct(row.get("worst_accuracy_drop_vs_structured_dense")),
                protocol=_fmt_pct(row.get("min_valid_stage_protocol_rate")),
                fallback=_fmt_pct(row.get("max_fallback_rate")),
                trunc=_fmt_pct(row.get("max_truncation_rate")),
                pruning=_fmt_pct(row.get("mean_theoretical_average_mlp_pruning_ratio")),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_downstream_methods_from_selection(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = read_json(path)
    if selection.get("schema") != "stage_policy_selection_v1":
        raise ValueError(f"Unsupported policy selection schema: {selection.get('schema')}")
    if selection.get("test_sets_consulted") is not False:
        raise ValueError("Refusing policy selection that may have consulted final test sets")
    methods = selection.get("downstream_methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Policy selection does not contain downstream_methods")
    if selection.get("selection_mode") == "main_only":
        names = [str(method.get("name")) for method in methods]
        expected = ["structured_dense", "dynamic_stage_main", "static_matched_global"]
        if names != expected:
            raise ValueError(f"Main-only policy selection must contain exactly {expected}, got {names}")
    if selection.get("selection_mode") == "adaptive_griffin_main_only":
        names = [str(method.get("name")) for method in methods]
        allowed_adaptive = {
            "calibrated_stage_adaptive_griffin_main",
            "calibrated_stage_safe_dynamic_griffin_main",
        }
        if (
            len(names) != 3
            or names[0] != "structured_dense"
            or names[1] not in allowed_adaptive
            or names[2] != "static_matched_global"
        ):
            raise ValueError(
                "Adaptive GRIFFIN policy selection must contain exactly "
                "structured_dense, one approved adaptive method, and static_matched_global; "
                f"got {names}"
            )
    if selection.get("selection_mode") == "adaptive_griffin_sweep":
        names = [str(method.get("name")) for method in methods]
        if len(names) != len(set(names)):
            raise ValueError(f"Adaptive GRIFFIN sweep contains duplicate method names: {names}")
        if len(names) < 4 or names[0] != "structured_dense" or names[-1] != "static_matched_global":
            raise ValueError(
                "Adaptive GRIFFIN sweep must contain structured_dense, one or more adaptive "
                f"methods, and static_matched_global; got {names}"
            )
        allowed_policies = {
            "calibrated_stage_adaptive_griffin",
            "calibrated_stage_safe_dynamic_griffin",
            "calibrated_stage_static_core_residual_griffin",
        }
        invalid = [
            method
            for method in methods[1:-1]
            if str(method.get("policy")) not in allowed_policies
        ]
        if invalid:
            raise ValueError(
                "Adaptive GRIFFIN sweep middle methods must all be adaptive policies; "
                f"got {[method.get('name') for method in invalid]}"
            )
        if str(methods[0].get("policy")) != "trajectory_global":
            raise ValueError("Adaptive GRIFFIN sweep structured_dense must use trajectory_global policy")
        if str(methods[-1].get("policy")) != "trajectory_global":
            raise ValueError("Adaptive GRIFFIN sweep static_matched_global must use trajectory_global policy")
    return methods, selection


def _find_method(methods: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in methods:
        if row.get("method", {}).get("name") == name:
            return row
    raise KeyError(f"Missing method summary: {name}")


def _method_identity(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": method.get("policy"),
        "stage_ratios": method.get("stage_ratios"),
        "prompt": method.get("prompt"),
        "bias_compensation": method.get("bias_compensation", True),
    }


def _classify_aggregate(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    name = str(row["method_name"])
    is_baseline = name in {"ordinary_dense", "structured_dense"}
    common_reasons = []
    if row["seed_count"] != row["expected_seed_count"]:
        common_reasons.append("incomplete_seed_coverage")
    if not row["method_identity_consistent"]:
        common_reasons.append("inconsistent_method_identity")
    if _float(row.get("min_valid_stage_protocol_rate")) < thresholds["minimum_protocol_rate"]:
        common_reasons.append("protocol_below_threshold")
    if _float(row.get("max_fallback_rate")) > thresholds["maximum_fallback_rate"]:
        common_reasons.append("fallback_above_threshold")
    if _float(row.get("max_truncation_rate")) > thresholds["maximum_truncation_rate"]:
        common_reasons.append("truncation_above_threshold")
    if is_baseline:
        return {
            "selection_tier": "baseline",
            "tier_rank": 0,
            "selection_reasons": ["dense_reference"],
            "quality_passed": True,
            "is_baseline": True,
            "is_shuffled_control": False,
        }
    quality_passed = not common_reasons
    mean_drop = _float(row.get("mean_accuracy_drop_vs_structured_dense"))
    worst_drop = _float(row.get("worst_accuracy_drop_vs_structured_dense"))
    pruning = _float(row.get("mean_theoretical_average_mlp_pruning_ratio"))
    if quality_passed and mean_drop <= thresholds["formal_safe_mean_drop"] and worst_drop <= thresholds["formal_safe_worst_drop"]:
        tier = "formal_safe"
        rank = 1
        reasons = ["formal_safe_drop_and_quality_passed"]
    elif quality_passed and mean_drop <= thresholds["main_candidate_mean_drop"] and worst_drop <= thresholds["main_candidate_worst_drop"]:
        tier = "main_candidate"
        rank = 2
        reasons = ["main_candidate_drop_and_quality_passed"]
    elif quality_passed and pruning >= thresholds["aggressive_boundary_min_pruning"] and mean_drop <= thresholds["aggressive_boundary_mean_drop"]:
        tier = "aggressive_boundary"
        rank = 3
        reasons = ["aggressive_boundary_tradeoff"]
    else:
        tier = "rejected"
        rank = 4
        reasons = common_reasons or ["drop_or_pruning_threshold_not_met"]
    return {
        "selection_tier": tier,
        "tier_rank": rank,
        "selection_reasons": reasons,
        "quality_passed": quality_passed,
        "is_baseline": False,
        "is_shuffled_control": row.get("policy") == "shuffled_stage",
    }


def _pick_conservative(aggregates: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in aggregates
        if _selectable_method(row)
        and row["selection_tier"] == "formal_safe"
        and thresholds["conservative_min_pruning"]
        <= _float(row.get("mean_theoretical_average_mlp_pruning_ratio"))
        <= thresholds["conservative_max_pruning"]
    ]
    if not candidates:
        candidates = [
            row
            for row in aggregates
            if _selectable_method(row)
            and row["selection_tier"] == "main_candidate"
            and thresholds["conservative_min_pruning"]
            <= _float(row.get("mean_theoretical_average_mlp_pruning_ratio"))
            <= thresholds["conservative_max_pruning"]
        ]
    if not candidates:
        candidates = [
            row
            for row in aggregates
            if _selectable_method(row) and row["selection_tier"] in {"formal_safe", "main_candidate"}
        ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (
            _float(row.get("mean_theoretical_average_mlp_pruning_ratio")),
            -_float(row.get("worst_accuracy_drop_vs_structured_dense")),
            -_float(row.get("mean_accuracy_drop_vs_structured_dense")),
        ),
    )
    return _selection_entry("conservative", best, "stable low-pruning policy")


def _pick_main_dynamic(aggregates: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in aggregates
        if _selectable_method(row)
        and row["policy"] == "stage_specific"
        and row["selection_tier"] in {"formal_safe", "main_candidate", "aggressive_boundary"}
        and thresholds["main_dynamic_min_pruning"]
        <= _float(row.get("mean_theoretical_average_mlp_pruning_ratio"))
        <= thresholds["main_dynamic_max_pruning"]
    ]
    if not candidates:
        candidates = [
            row
            for row in aggregates
            if _selectable_method(row)
            and row["selection_tier"] in {"formal_safe", "main_candidate", "aggressive_boundary"}
            and thresholds["main_dynamic_min_pruning"]
            <= _float(row.get("mean_theoretical_average_mlp_pruning_ratio"))
            <= thresholds["main_dynamic_max_pruning"]
        ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (
            row["policy"] == "stage_specific",
            _float(row.get("mean_theoretical_average_mlp_pruning_ratio")),
            -_float(row.get("worst_accuracy_drop_vs_structured_dense")),
        ),
    )
    return _selection_entry("main_dynamic", best, "best cross-seed dynamic candidate in the 15%-25% band")


def _pick_aggressive(aggregates: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in aggregates
        if _selectable_method(row)
        and row["quality_passed"]
        and _float(row.get("mean_theoretical_average_mlp_pruning_ratio"))
        >= thresholds["aggressive_selection_min_pruning"]
    ]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda row: (
            _float(row.get("mean_accuracy_drop_vs_structured_dense")),
            _float(row.get("worst_accuracy_drop_vs_structured_dense")),
            -_float(row.get("mean_theoretical_average_mlp_pruning_ratio")),
        ),
    )
    entry = _selection_entry("aggressive", best, "boundary-curve candidate; not the main retention claim")
    entry["diagnostic_only"] = True
    entry["method"]["diagnostic_only"] = True
    return entry


def _pick_shuffled_control(aggregates: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in aggregates
        if row.get("is_shuffled_control")
        and row["selection_tier"] in {"formal_safe", "main_candidate", "aggressive_boundary"}
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (
            row["selection_tier"] == "formal_safe",
            _float(row.get("mean_theoretical_average_mlp_pruning_ratio")),
            -_float(row.get("mean_accuracy_drop_vs_structured_dense")),
        ),
    )
    return _selection_entry("shuffled_control", best, "control only; never selected as the main method")


def _pick_main_only_dynamic(
    aggregates: list[dict[str, Any]],
    thresholds: dict[str, float],
    candidate_ratios: tuple[float, ...],
) -> dict[str, Any] | None:
    allowed = {round(float(value), 8) for value in candidate_ratios}
    candidates = []
    for row in aggregates:
        if row.get("policy") != "stage_specific":
            continue
        ratios = {stage: float(row.get("stage_ratios", {}).get(stage, 0.0)) for stage in STAGES}
        if len({round(value, 8) for value in ratios.values()}) != 1:
            continue
        ratio = round(next(iter(ratios.values())), 8)
        if ratio not in allowed:
            continue
        if not row.get("quality_passed"):
            continue
        if _float(row.get("mean_accuracy_drop_vs_structured_dense")) > thresholds["main_only_max_drop"]:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            _average_ratio(row.get("stage_ratios", {})),
            -_float(row.get("mean_accuracy_drop_vs_structured_dense")),
            -_float(row.get("worst_accuracy_drop_vs_structured_dense")),
        ),
    )


def _selectable_method(row: dict[str, Any]) -> bool:
    name = str(row["method_name"])
    policy = str(row.get("policy"))
    if name.startswith("coordinate_"):
        return False
    if row.get("is_baseline") or row.get("is_shuffled_control"):
        return False
    if policy == "prompt_only_global":
        return False
    if name.endswith("_no_bias_compensation"):
        return False
    return policy in {"trajectory_global", "stage_balanced_global", "stage_specific"}


def _average_ratio(stage_ratios: dict[str, Any]) -> float:
    return sum(float(stage_ratios.get(stage, 0.0)) for stage in STAGES) / len(STAGES)


def _selection_entry(role: str, aggregate: dict[str, Any] | None, reason: str) -> dict[str, Any] | None:
    if aggregate is None:
        return None
    policy_id = f"{role}:{aggregate['method_name']}:{stable_hash(aggregate['method'])[:12]}"
    method = _method_with_selection_metadata(
        aggregate["method"],
        role=role,
        policy_id=policy_id,
        tier=aggregate["selection_tier"],
        source="cross_seed_dev",
        diagnostic_only=role == "aggressive",
    )
    return {
        "role": role,
        "policy_id": policy_id,
        "selection_tier": aggregate["selection_tier"],
        "selection_source": "cross_seed_dev",
        "reason": reason,
        "aggregate": aggregate,
        "method": method,
        "diagnostic_only": role == "aggressive",
    }


def _method_with_selection_metadata(
    method: dict[str, Any],
    *,
    role: str,
    policy_id: str,
    tier: str,
    source: str,
    diagnostic_only: bool,
) -> dict[str, Any]:
    output = dict(method)
    original_name = str(output["name"])
    if (
        role not in {"ordinary_dense", "structured_dense"}
        and original_name != role
        and not original_name.startswith(f"{role}_")
    ):
        output["name"] = f"{role}_{original_name}"
    output["selection_role"] = role
    output["selection_policy_id"] = policy_id
    output["selection_tier"] = tier
    output["selection_source"] = source
    output["diagnostic_only"] = bool(diagnostic_only)
    output["original_method_name"] = original_name
    return output


def _aggregate_by_name(aggregates: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in aggregates:
        if row["method_name"] == name:
            return row
    return None


def _dedupe_methods(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for method in methods:
        key = str(method["name"])
        if key in seen:
            continue
        seen.add(key)
        output.append(method)
    return output


def _float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _mean(values) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    return sum(materialized) / len(materialized) if materialized else None


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select frozen cross-seed stage pruning policies.")
    parser.add_argument("--roots", nargs="+", required=True, help="Run roots or 05_dev/summary.json files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-stage-budget-presets", action="store_true")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--main-only-candidate-ratios", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    parser.add_argument("--main-only-max-drop", type=float, default=DEFAULT_THRESHOLDS["main_only_max_drop"])
    args = parser.parse_args()

    seed_runs = [load_seed_run(path) for path in args.roots]
    if args.main_only:
        selection = build_main_only_policy_selection(
            seed_runs,
            thresholds={"main_only_max_drop": args.main_only_max_drop},
            candidate_ratios=tuple(args.main_only_candidate_ratios),
        )
    else:
        selection = build_policy_selection(
            seed_runs,
            include_stage_budget_presets=not args.no_stage_budget_presets,
        )
    output_dir = ensure_dir(args.output_dir)
    json_name = "policy_selection_main_only.json" if args.main_only else "policy_selection.json"
    md_name = "policy_selection_main_only.md" if args.main_only else "policy_selection.md"
    write_json(output_dir / json_name, selection)
    write_policy_selection_markdown(selection, output_dir / md_name)
    print(output_dir / json_name)


if __name__ == "__main__":
    main()
