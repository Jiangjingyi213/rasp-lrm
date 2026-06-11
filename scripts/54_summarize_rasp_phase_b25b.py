from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/rasp_phase_b25b")
    args = parser.parse_args()
    root = Path(args.root)
    raw = []
    for path in sorted(root.glob("seed_*/frozen_uncertainty_residual/eval.json")):
        result = json.load(path.open())
        row = {
            "seed": result["seed"],
            "alpha": result["alpha"],
            "base_action_roc_auc": result["base"]["action"]["roc_auc"],
            "combined_action_roc_auc": result["combined"]["action"]["roc_auc"],
            "action_roc_delta": result["combined"]["action"]["roc_auc"] - result["base"]["action"]["roc_auc"],
            "action_roc_delta_ci_low": result["paired_delta"]["action"]["roc_auc"]["ci95"][0],
            "action_roc_delta_ci_high": result["paired_delta"]["action"]["roc_auc"]["ci95"][1],
            "action_roc_positive_fraction": result["paired_delta"]["action"]["roc_auc"]["positive_fraction"],
            "action_pr_delta": result["combined"]["action"]["pr_auc"] - result["base"]["action"]["pr_auc"],
            "action_pr_delta_ci_low": result["paired_delta"]["action"]["pr_auc"]["ci95"][0],
            "action_pr_delta_ci_high": result["paired_delta"]["action"]["pr_auc"]["ci95"][1],
            "base_boundary_roc_auc": result["base"]["boundary"]["roc_auc"],
            "combined_boundary_roc_auc": result["combined"]["boundary"]["roc_auc"],
            "boundary_roc_delta": result["combined"]["boundary"]["roc_auc"] - result["base"]["boundary"]["roc_auc"],
            "boundary_roc_delta_ci_low": result["paired_delta"]["boundary"]["roc_auc"]["ci95"][0],
            "boundary_roc_delta_ci_high": result["paired_delta"]["boundary"]["roc_auc"]["ci95"][1],
            "boundary_roc_positive_fraction": result["paired_delta"]["boundary"]["roc_auc"]["positive_fraction"],
            "boundary_pr_delta": result["combined"]["boundary"]["pr_auc"] - result["base"]["boundary"]["pr_auc"],
            "boundary_pr_delta_ci_low": result["paired_delta"]["boundary"]["pr_auc"]["ci95"][0],
            "boundary_pr_delta_ci_high": result["paired_delta"]["boundary"]["pr_auc"]["ci95"][1],
        }
        for budget, metrics in result["budgets"].items():
            row[f"b{budget}_base_ratio"] = metrics["base"]["average_selected_ratio"]
            row[f"b{budget}_combined_ratio"] = metrics["combined"]["average_selected_ratio"]
            row[f"b{budget}_base_flip"] = metrics["base"]["flip_rate"]
            row[f"b{budget}_combined_flip"] = metrics["combined"]["flip_rate"]
        raw.append(row)
    if not raw:
        raise SystemExit(f"No Phase B2.5b eval results found under {root}")
    summary = {
        "seeds": len(raw),
        "alpha_mean": statistics.mean(float(row["alpha"]) for row in raw),
        "all_alphas_positive": all(float(row["alpha"]) > 0 for row in raw),
    }
    for metric in (
        "action_roc_delta", "action_pr_delta", "boundary_roc_delta", "boundary_pr_delta",
        "b0.15_base_ratio", "b0.15_combined_ratio", "b0.15_base_flip", "b0.15_combined_flip",
        "b0.20_base_ratio", "b0.20_combined_ratio", "b0.20_base_flip", "b0.20_combined_flip",
    ):
        values = [float(row[metric]) for row in raw]
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    summary["all_action_deltas_positive"] = all(float(row["action_roc_delta"]) > 0 for row in raw)
    summary["all_boundary_deltas_positive"] = all(float(row["boundary_roc_delta"]) > 0 for row in raw)
    summary["all_action_ci_lows_positive"] = all(float(row["action_roc_delta_ci_low"]) > 0 for row in raw)
    summary["all_boundary_ci_lows_positive"] = all(float(row["boundary_roc_delta_ci_low"]) > 0 for row in raw)
    with (root / "comparison_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    with (root / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (root / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
