#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "generated_rasp_zero_online_conservative" / "manifest.json"
OUTPUT = ROOT / "runs" / "rasp_zero_online_conservative_calibration" / "summary.csv"


def main() -> None:
    manifest = json.load(MANIFEST.open("r", encoding="utf-8"))
    rows = []
    for row in manifest:
        summary_path = ROOT / row["run_dir"] / "00_runtime_summary.json"
        if not summary_path.exists():
            rows.append({**row, "status": "missing"})
            continue
        summary = json.load(summary_path.open("r", encoding="utf-8"))
        rows.append(
            {
                **row,
                "status": "ok",
                "examples": summary.get("examples"),
                "correct": summary.get("correct"),
                "accuracy": summary.get("accuracy"),
                "generated_tokens": summary.get("generated_tokens"),
                "avg_generated_tokens": summary.get("average_generated_tokens"),
                "tokens_per_second": summary.get("decode_tokens_per_second"),
                "average_pruning_ratio": summary.get("average_decode_pruning_ratio"),
                "theoretical_mlp_reduction": summary.get("theoretical_mlp_decode_flops_reduction"),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "policy",
        "risk_threshold",
        "target_average_ratio",
        "early_tokens",
        "early_max_ratio",
        "high_entropy_threshold",
        "high_entropy_max_ratio",
        "low_confidence_threshold",
        "low_confidence_max_ratio",
        "status",
        "examples",
        "correct",
        "accuracy",
        "generated_tokens",
        "avg_generated_tokens",
        "tokens_per_second",
        "average_pruning_ratio",
        "theoretical_mlp_reduction",
        "run_dir",
        "config",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    for row in rows:
        print(
            f"{row['dataset']:7s} {row['policy']:32s} "
            f"acc={row.get('accuracy')} ratio={row.get('average_pruning_ratio')} status={row['status']}"
        )


if __name__ == "__main__":
    main()
