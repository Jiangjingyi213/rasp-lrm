from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", default="runs/07_stage_aware/06_action_risk_pilot"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.source_root)
    data_summary = read_json(source / "data" / "01_action_risk_data_summary.json")
    analysis = read_json(source / "analysis" / "02_action_risk_pilot_summary.json")
    rows = read_jsonl(source / "data" / "01_action_risk_dataset.jsonl")
    by_source = data_summary["source_full_window_eligible_dense_correct_problems"]
    max_boundaries = {}
    for dataset in ("gsm8k", "math_train"):
        values = [
            int(row["generated_tokens_at_boundary"])
            for row in rows
            if row["dataset"] == dataset
        ]
        max_boundaries[dataset] = max(values) if values else None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checks = {
        "minimum_existing_problem_count_per_source": all(
            int(by_source.get(name, 0)) >= 100 for name in ("gsm8k", "math_train")
        ),
        "existing_action_rows_nonzero": int(analysis["action_rows"]) > 0,
    }
    summary = {
        "schema": "rasp_full_trajectory_workflow_preflight_v1",
        "preflight_valid": all(checks.values()),
        "checks": checks,
        "existing_problem_count": analysis["problem_count"],
        "existing_boundary_count": analysis["boundary_count"],
        "existing_action_rows": analysis["action_rows"],
        "existing_positive_flips": analysis["positive_flips"],
        "existing_dose_response_inversions": analysis["dose_response_inversions"],
        "existing_max_boundary_tokens_by_source": max_boundaries,
        "existing_dense_risk_signal": {
            "action_only": analysis["variants"]["action_only"],
            "causal_context_action": analysis["variants"]["causal_context_action"],
        },
        "limitations": [
            "Existing bank uses uniform_full_trajectory sampling based on future trajectory length.",
            "Existing bank is reused for CPU diagnosis only and is not deployed as an online timing policy.",
            "Existing bank does not contain validated causal soft-stage annotations.",
        ],
    }
    write_json(output / "00_existing_bank_preflight.json", summary)
    write_json(
        output / "phase_gate.json",
        {
            "schema": "rasp_full_trajectory_workflow_preflight_gate_v1",
            "passed": summary["preflight_valid"],
            "checks": checks,
            "next_step": (
                "Continue to the causal-grid smoke."
                if summary["preflight_valid"]
                else "Stop and inspect the existing Action-Risk bank."
            ),
        },
    )
    if not summary["preflight_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
