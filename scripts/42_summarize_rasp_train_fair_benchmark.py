from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ["average_selected_ratio", "budget_utilization", "flip_rate", "conservative_unsafe_rate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/04_rasp_train/02_fair_benchmark/rasp_train_fair_benchmark")
    args = parser.parse_args()
    root = Path(args.root)
    raw = []
    grouped = defaultdict(list)
    for path in sorted(root.glob("seed_*/*/*/eval.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        for budget, metrics in result["budgets"].items():
            row = {
                "seed": result["seed"],
                "label_type": result["label_type"],
                "variant": result["variant"],
                "budget": budget,
                **metrics,
            }
            raw.append(row)
            grouped[(result["label_type"], result["variant"], budget)].append(row)
    if not raw:
        raise SystemExit(f"No eval.json files found under {root}")

    summary = []
    for (label, variant, budget), rows in sorted(grouped.items()):
        output = {"label_type": label, "variant": variant, "budget": budget, "seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(output)
    for name, rows in [("comparison_raw.csv", raw), ("comparison_summary.csv", summary)]:
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (root / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(raw)} raw rows and {len(summary)} aggregate rows to {root}")


if __name__ == "__main__":
    main()
