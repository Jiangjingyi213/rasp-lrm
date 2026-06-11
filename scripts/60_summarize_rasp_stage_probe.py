from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


KEY_RECALLS = ("setup", "reasoning", "verification", "final")
REQUIRED_VARIANTS = (
    "position_only",
    "uncertainty_only",
    "hidden_pca_linear",
    "hidden_pca_nonlinear",
    "hidden_uncertainty",
)
REQUIRED_SEEDS = 3
MAX_MACRO_F1_STD = 0.05
REQUIRED_AUDIT_ROWS = 100
REQUIRED_AUDIT_AGREEMENT = 0.80


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/07_stage_aware/02_s1_operational_stage_probe")
    args = parser.parse_args()
    root = Path(args.root)
    raw, grouped = [], defaultdict(list)
    for path in sorted(root.glob("seed_*/*/eval.json")):
        result = json.loads(path.read_text())
        row = {
            "seed": result["seed"],
            "variant": result["variant"],
            "macro_f1": result["macro_f1"],
            "mean_max_probability": result["mean_max_probability"],
            **{f"recall_{stage}": value for stage, value in result["per_stage_recall"].items()},
        }
        raw.append(row)
        grouped[result["variant"]].append(row)
    if not raw:
        raise SystemExit(f"No stage eval results under {root}")
    summary = []
    for variant, rows in sorted(grouped.items()):
        output = {"variant": variant, "seeds": len({int(row["seed"]) for row in rows})}
        recall_metrics = sorted(key for key in rows[0] if key.startswith("recall_"))
        for metric in ["macro_f1", "mean_max_probability", *recall_metrics]:
            values = [float(value[metric]) for value in rows]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output[f"{metric}_min"] = min(values)
            output[f"{metric}_max"] = max(values)
        summary.append(output)
    by_variant = {row["variant"]: row for row in summary}
    hidden_candidates = [
        by_variant[name]
        for name in ("hidden_pca_linear", "hidden_pca_nonlinear", "hidden_uncertainty")
        if name in by_variant
    ]
    if not hidden_candidates or not all(name in by_variant for name in ("position_only", "uncertainty_only")):
        raise SystemExit("Stage summary requires hidden and both simple baseline variants")
    best_hidden = max(hidden_candidates, key=lambda row: row["macro_f1_mean"])
    simple_best = max(
        by_variant[name]["macro_f1_mean"]
        for name in ("position_only", "uncertainty_only")
        if name in by_variant
    )
    audit_path = root / "data" / "02_stage_manual_audit.csv"
    audit_rows = []
    if audit_path.exists():
        with audit_path.open(newline="", encoding="utf-8") as handle:
            audit_rows = list(csv.DictReader(handle))
    audited = [row for row in audit_rows if row.get("audited_stage", "").strip()]
    valid_audited = [
        row
        for row in audited
        if row["audited_stage"].strip() in KEY_RECALLS
    ]
    audit_agreement = (
        sum(row["audited_stage"].strip() == row["stage"].strip() for row in valid_audited)
        / len(valid_audited)
        if valid_audited
        else 0.0
    )
    gate = {
        "best_hidden_variant": best_hidden["variant"],
        "hidden_macro_f1_advantage_over_simple": best_hidden["macro_f1_mean"] - simple_best,
        "required_macro_f1_advantage": 0.05,
        "required_key_stage_recall": 0.70,
        "required_seeds": REQUIRED_SEEDS,
        "required_macro_f1_std_max": MAX_MACRO_F1_STD,
        "required_manual_audit_rows": REQUIRED_AUDIT_ROWS,
        "required_manual_label_agreement": REQUIRED_AUDIT_AGREEMENT,
        "valid_manual_audit_rows": len(valid_audited),
        "manual_label_agreement": audit_agreement,
        "macro_f1_gate_passed": best_hidden["macro_f1_mean"] - simple_best >= 0.05,
        "key_stage_recall_gate_passed": all(
            best_hidden[f"recall_{stage}_min"] >= 0.70 for stage in KEY_RECALLS
        ),
        "three_seed_stability_gate_passed": (
            all(
                by_variant.get(variant, {}).get("seeds") == REQUIRED_SEEDS
                for variant in REQUIRED_VARIANTS
            )
            and best_hidden["macro_f1_std"] <= MAX_MACRO_F1_STD
        ),
        "manual_audit_gate_passed": (
            len(valid_audited) >= REQUIRED_AUDIT_ROWS
            and audit_agreement >= REQUIRED_AUDIT_AGREEMENT
        ),
    }
    gate["s1_passed"] = all(
        gate[name]
        for name in (
            "macro_f1_gate_passed",
            "key_stage_recall_gate_passed",
            "three_seed_stability_gate_passed",
            "manual_audit_gate_passed",
        )
    )
    for name, rows in (("comparison_raw.csv", raw), ("comparison_summary.csv", summary)):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "s1_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(gate)


if __name__ == "__main__":
    main()
