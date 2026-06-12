from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="runs/07_stage_aware/05_s2_stage_sensitivity_v2",
    )
    parser.add_argument("--safe-flip-rate", type=float, default=0.02)
    args = parser.parse_args()
    root = Path(args.root)
    grouped = defaultdict(list)
    selective_reasoning = defaultdict(list)
    validations = []
    for path in sorted(root.glob("*/07_stage_window_bank_validation.json")):
        validation = json.loads(path.read_text(encoding="utf-8"))
        validations.append(validation)
        if validation.get("status") != "ok":
            continue
        rows_path = path.parent / "03_stage_window_counterfactuals.jsonl"
        with rows_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if float(row["ratio"]) > 0:
                    grouped[
                        (
                            str(row.get("dataset") or "unknown"),
                            str(row["operational_stage"]),
                            float(row["ratio"]),
                        )
                    ].append(row)
                    if row["operational_stage"] == "reasoning":
                        selective_reasoning[
                            (
                                str(row.get("dataset") or "unknown"),
                                bool(row.get("reasoning_accepted")),
                                float(row["ratio"]),
                            )
                        ].append(row)
    if not grouped:
        raise SystemExit(f"No validated S2 rows under {root}")
    summary = []
    for (dataset, stage, ratio), rows in sorted(grouped.items()):
        flips = sum(int(bool(row["flipped"])) for row in rows)
        lower, upper = wilson_interval(flips, len(rows))
        summary.append(
            {
                "dataset": dataset,
                "operational_stage": stage,
                "ratio": ratio,
                "boundaries": len(rows),
                "flips": flips,
                "paired_flip_rate": flips / len(rows),
                "paired_flip_rate_wilson_95_low": lower,
                "paired_flip_rate_wilson_95_high": upper,
                "mean_window_token_divergence": sum(
                    float(row["window_token_divergence"]) for row in rows
                )
                / len(rows),
                "safe_by_point_estimate": flips / len(rows) <= args.safe_flip_rate,
            }
        )
    safe_cells = [row for row in summary if row["safe_by_point_estimate"]]
    selective_summary = []
    for (dataset, accepted, ratio), rows in sorted(selective_reasoning.items()):
        flips = sum(int(bool(row["flipped"])) for row in rows)
        lower, upper = wilson_interval(flips, len(rows))
        selective_summary.append(
            {
                "dataset": dataset,
                "reasoning_accepted": accepted,
                "ratio": ratio,
                "boundaries": len(rows),
                "flips": flips,
                "paired_flip_rate": flips / len(rows),
                "paired_flip_rate_wilson_95_low": lower,
                "paired_flip_rate_wilson_95_high": upper,
            }
        )
    result = {
        "schema": "rasp_s2_stage_sensitivity_smoke_v1",
        "validated_shards": sum(v.get("status") == "ok" for v in validations),
        "total_validation_files": len(validations),
        "safe_flip_rate": args.safe_flip_rate,
        "safe_cells_by_point_estimate": safe_cells,
        "warning": "Smoke cells are exploratory; formal admission requires enough paired samples and held-out confirmation.",
        "stage_ratio_summary": summary,
        "selective_reasoning_summary": selective_summary,
    }
    (root / "s2_stage_sensitivity_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    with (root / "s2_stage_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(result)


if __name__ == "__main__":
    main()
