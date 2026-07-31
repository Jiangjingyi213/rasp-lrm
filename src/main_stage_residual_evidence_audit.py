from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import read_json, write_json


METHODS = (
    "t30_math_safe",
    "dynamic_global_prior_t30",
    "dynamic_shuffled_prior_t30",
)


def _row_map(aggregate: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in aggregate.get("datasets", []):
        name = str(dataset["dataset"])
        output[name] = {str(row["method"]): row for row in dataset.get("rows", [])}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the no-GPU evidence audit for Stage-Residual RASP.")
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    aggregate_path = Path(args.aggregate)
    rows = _row_map(read_json(aggregate_path))
    datasets = [name for name in ("gsm8k", "math500") if name in rows]
    comparisons = []
    for dataset in datasets:
        methods = rows[dataset]
        global_row = methods.get("dynamic_global_prior_t30")
        for method in METHODS:
            row = methods.get(method)
            if row is None or global_row is None:
                continue
            comparisons.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "accuracy": row.get("accuracy"),
                    "actual_pruning": row.get("actual_pruning"),
                    "fallback": row.get("fallback"),
                    "truncation": row.get("truncation"),
                    "accuracy_vs_dynamic_global_pp": 100.0
                    * (float(row["accuracy"]) - float(global_row["accuracy"])),
                }
            )
    audit = {
        "schema": "stage_residual_existing_evidence_audit_v1",
        "source_aggregate": str(aggregate_path),
        "conclusion": (
            "Existing t30 core ablation does not justify treating the raw stage_specific "
            "prior as the main method. Stage-Residual v2 tests bounded residual priors "
            "without collecting a new counterfactual bank."
        ),
        "comparisons": comparisons,
        "new_gpu_data_collection_required": False,
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "evidence_audit.json", audit)
    lines = [
        "# Existing Evidence Audit",
        "",
        "The raw stage-specific prior is a historical control, not the v2 default.",
        "No new motivation, Phase B, action-risk, or counterfactual bank is scheduled.",
        "",
        "| dataset | method | accuracy | delta vs dynamic-global | actual pruning | fallback | truncation |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['dataset']} | `{row['method']}` | {100 * float(row['accuracy']):.2f}% | "
            f"{float(row['accuracy_vs_dynamic_global_pp']):+.2f}pp | "
            f"{100 * float(row['actual_pruning']):.2f}% | {100 * float(row['fallback']):.2f}% | "
            f"{100 * float(row['truncation']):.2f}% |"
        )
    (output_dir / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "manifest.json", audit)
    write_json(output_dir / "status.json", {"status": "completed", "artifact": "evidence_audit.json"})


if __name__ == "__main__":
    main()
