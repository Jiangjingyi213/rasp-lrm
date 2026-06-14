from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("dense_smoke", "dense_pilot"), required=True)
    parser.add_argument("--phase-dir", required=True)
    args = parser.parse_args()
    root = Path(args.phase_dir)
    summary = read_json(root / "data" / "01_full_trajectory_data_summary.json")
    required = 4 if args.profile == "dense_smoke" else 20
    max_mismatch = 0.0 if args.profile == "dense_smoke" else 0.01
    source_counts = summary["source_eligible_problems"]
    checks = {
        "minimum_eligible_problems": all(
            int(source_counts.get(source, 0)) >= required
            for source in ("gsm8k", "math_train")
        ),
        "dense_replay_mismatch_within_limit": float(
            summary["dense_replay_mismatch_rate"]
        )
        <= max_mismatch,
        "causal_boundaries_nonzero": int(summary["causal_boundary_count"]) > 0,
        "tail_anchor_diagnostic_only": bool(summary["tail_anchor_is_diagnostic_only"]),
        "no_final_test_sources": not bool(summary["final_test_sources_used"]),
        "causal_features_exclude_future_length": bool(
            summary["causal_features_exclude_dense_trajectory_length"]
        ),
        "causal_soft_stage_verified": bool(summary["causal_soft_stage_verified"]),
        "dense_correctness_recomputed": bool(
            summary["dense_correctness_recomputed_from_completion"]
        ),
    }
    if args.profile == "dense_pilot":
        analysis = read_json(root / "analysis" / "02_full_trajectory_analysis.json")
        covered_deciles = sum(
            int(int(value) > 0)
            for value in analysis[
                "diagnostic_dense_trajectory_position_decile_counts"
            ].values()
        )
        checks.update(
            {
                "grouped_oof_completed": all(
                    len(value.get("folds", [])) == 5
                    for value in analysis["variants"].values()
                ),
                "positive_flip_labels_exist": int(analysis["positive_flips"]) > 0,
                "dose_response_reasonable": int(
                    analysis["dose_response_inversions"]
                )
                <= 1,
                "causal_feature_audit_passed": bool(
                    analysis["causal_feature_audit_passed"]
                ),
                "trajectory_coverage_spans_multiple_deciles": covered_deciles >= 4,
            }
        )
    passed = all(checks.values())
    write_json(
        root / "phase_gate.json",
        {
            "schema": "rasp_full_trajectory_bank_gate_v1",
            "profile": args.profile,
            "passed": passed,
            "checks": checks,
            "required_eligible_problems_per_source": required,
            "maximum_dense_replay_mismatch_rate": max_mismatch,
            "next_step": (
                "Continue workflow."
                if passed
                else "Stop workflow and inspect dense bank collection."
            ),
        },
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
