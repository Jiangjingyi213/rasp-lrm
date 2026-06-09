from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ["average_selected_ratio", "budget_utilization", "flip_rate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/rasp_phase_b2")
    args = parser.parse_args()
    root = Path(args.root)
    raw, grouped = [], defaultdict(list)
    for path in sorted(root.glob("seed_*/*/eval.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        for budget, metrics in result["budgets"].items():
            row = {
                "seed": result["seed"],
                "variant": result["variant"],
                "budget": budget,
                "risk_roc_auc": result["risk_roc_auc"],
                "risk_pr_auc": result["risk_pr_auc"],
                **metrics,
            }
            raw.append(row)
            grouped[(result["variant"], budget)].append(row)
    if not raw:
        raise SystemExit(f"No Phase B2 eval results found under {root}")
    summary = []
    for (variant, budget), rows in sorted(grouped.items()):
        output = {"variant": variant, "budget": budget, "seeds": len(rows)}
        for metric in ["risk_roc_auc", "risk_pr_auc", *METRICS]:
            values = [float(row[metric]) for row in rows]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output["all_calibration_constraints_satisfied"] = all(
            bool(row["calibration_constraints_satisfied"]) for row in rows
        )
        summary.append(output)
    for name, rows in (("comparison_raw.csv", raw), ("comparison_summary.csv", summary)):
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (root / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {len(raw)} raw rows and {len(summary)} summary rows")


if __name__ == "__main__":
    main()
