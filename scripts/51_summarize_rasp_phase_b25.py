from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/rasp_phase_b25")
    args = parser.parse_args()
    root = Path(args.root)
    raw, grouped = [], defaultdict(list)
    for path in sorted(root.glob("seed_*/*/eval.json")):
        result = json.load(path.open())
        for budget, controller in result["budgets"].items():
            row = {
                "seed": result["seed"], "variant": result["variant"], "budget": budget,
                "action_roc_auc": result["action"]["roc_auc"],
                "action_pr_auc": result["action"]["pr_auc"],
                "boundary_roc_auc": result["boundary_any_flip"]["roc_auc"],
                "boundary_pr_auc": result["boundary_any_flip"]["pr_auc"],
                "action_roc_auc_ci_low": result["action"]["problem_bootstrap_95ci"]["roc_auc"][0],
                "action_roc_auc_ci_high": result["action"]["problem_bootstrap_95ci"]["roc_auc"][1],
                "boundary_roc_auc_ci_low": result["boundary_any_flip"]["problem_bootstrap_95ci"]["roc_auc"][0],
                "boundary_roc_auc_ci_high": result["boundary_any_flip"]["problem_bootstrap_95ci"]["roc_auc"][1],
                "checkpoint_selection_split": result["checkpoint_selection_split"],
                "transform_fit_split": result["transform_fit_split"],
                **controller,
            }
            raw.append(row)
            grouped[(result["variant"], budget)].append(row)
    if not raw:
        raise SystemExit(f"No Phase B2.5 eval results found under {root}")
    summary = []
    metrics = [
        "action_roc_auc", "action_pr_auc", "boundary_roc_auc", "boundary_pr_auc",
        "average_selected_ratio", "budget_utilization", "flip_rate",
    ]
    for (variant, budget), rows in sorted(grouped.items()):
        output = {"variant": variant, "budget": budget, "seeds": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output["all_checkpoints_selected_on_validation"] = all(
            row["checkpoint_selection_split"] == "validation" for row in rows
        )
        output["all_transforms_fit_on_train"] = all(row["transform_fit_split"] == "train" for row in rows)
        output["all_calibration_constraints_satisfied"] = all(
            bool(row["calibration_constraints_satisfied"]) for row in rows
        )
        summary.append(output)
    for name, rows in (("comparison_raw.csv", raw), ("comparison_summary.csv", summary)):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (root / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
