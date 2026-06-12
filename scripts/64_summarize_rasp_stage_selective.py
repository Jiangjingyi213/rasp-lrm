from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/07_stage_aware/03_s1_three_stage_probe")
    parser.add_argument("--max-test-setup-false-accept-rate", type=float, default=0.10)
    parser.add_argument("--min-test-reasoning-coverage", type=float, default=0.10)
    args = parser.parse_args()
    root = Path(args.root)
    grouped = defaultdict(list)
    raw = []
    for path in sorted(root.glob("seed_*/*/selective_eval.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "seed": int(result["seed"]),
            "variant": result["variant"],
            "threshold": result["calibration"]["threshold"],
            "validation_setup_false_accept_rate": result["validation"]["setup_false_accept_rate"],
            "validation_reasoning_coverage": result["validation"]["reasoning_coverage"],
            "test_setup_false_accept_rate": result["test"]["setup_false_accept_rate"],
            "test_reasoning_coverage": result["test"]["reasoning_coverage"],
            "test_accepted_reasoning_precision": result["test"]["accepted_reasoning_precision"],
        }
        raw.append(row)
        grouped[row["variant"]].append(row)
    if not raw:
        raise SystemExit(f"No selective eval results under {root}")
    summary = []
    for variant, rows in sorted(grouped.items()):
        output = {"variant": variant, "seeds": len({row["seed"] for row in rows})}
        for metric in (
            "threshold",
            "validation_setup_false_accept_rate",
            "validation_reasoning_coverage",
            "test_setup_false_accept_rate",
            "test_reasoning_coverage",
            "test_accepted_reasoning_precision",
        ):
            values = [float(row[metric]) for row in rows]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_min"] = min(values)
            output[f"{metric}_max"] = max(values)
        output["validation_eligible"] = (
            output["seeds"] == 3
            and output["validation_setup_false_accept_rate_max"]
            <= args.max_test_setup_false_accept_rate
            and output["validation_reasoning_coverage_min"] >= args.min_test_reasoning_coverage
        )
        output["test_passed"] = (
            output["seeds"] == 3
            and output["test_setup_false_accept_rate_max"]
            <= args.max_test_setup_false_accept_rate
            and output["test_reasoning_coverage_min"] >= args.min_test_reasoning_coverage
        )
        summary.append(output)
    eligible = [row for row in summary if row["validation_eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["validation_reasoning_coverage_mean"],
            -row["validation_setup_false_accept_rate_mean"],
        ),
        default=None,
    )
    gate = {
        "schema": "rasp_stage_probe_s1_5_gate_v1",
        "variant_selection_split": "validation",
        "s1_5_passed": bool(selected and selected["test_passed"]),
        "s2_diagnostic_allowed": selected is not None,
        "s3_controller_allowed": bool(selected and selected["test_passed"]),
        "selected_variant": selected["variant"] if selected else None,
        "max_test_setup_false_accept_rate": args.max_test_setup_false_accept_rate,
        "min_test_reasoning_coverage": args.min_test_reasoning_coverage,
        "selected_summary": selected,
        "s2_checkpoint_seed": 1 if selected else None,
        "s2_checkpoint": (
            str(root / "seed_1" / selected["variant"] / "stage_probe.pt") if selected else None
        ),
        "s2_reasoning_threshold": next(
            (
                row["threshold"]
                for row in raw
                if selected and row["variant"] == selected["variant"] and row["seed"] == 1
            ),
            None,
        ),
    }
    for name, rows in (("selective_raw.csv", raw), ("selective_summary.csv", summary)):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "selective_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "s1_5_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(gate)
    if not gate["s1_5_passed"]:
        print(
            "S1.5 controller gate failed: do not run S3 controller. "
            "A validation-eligible probe may still be used for diagnostic all-stage S2."
        )


if __name__ == "__main__":
    main()
