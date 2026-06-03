from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.segmentation.rule_segmenter import FINAL_ANSWER_RE
from src.utils.io import read_jsonl, read_yaml, write_json


def _ratio_key(value: float) -> str:
    return f"{float(value):.8f}"


def _step_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]), int(row["segment_id"]))


def validate_runtime_bank(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    cf_cfg = config.get("counterfactual", {})
    val_cfg = config.get("runtime_bank_validation", {})
    trajectories = read_jsonl(paths["trajectories"])
    segmented = read_jsonl(paths["segments"])
    counterfactuals = read_jsonl(paths["counterfactuals"])
    probe_rows = read_jsonl(paths["probe_dataset"])

    expected_module = str(val_cfg.get("module", "mlp_intermediate_channels"))
    expected_ratios = sorted(float(ratio) for ratio in cf_cfg.get("ratios", []))
    expected_ratio_keys = {_ratio_key(ratio) for ratio in expected_ratios}
    expected_layers = [int(layer) for layer in cf_cfg.get("layers", [])]
    require_all_dense_correct = bool(val_cfg.get("require_all_dense_correct", False))
    max_ratio_zero_flip_rate = float(val_cfg.get("max_ratio_zero_flip_rate", 0.05))
    allow_ratio_zero_filtering = bool(val_cfg.get("allow_ratio_zero_filtering", False))

    errors: list[str] = []
    warnings: list[str] = []
    dense_incorrect = [str(row["id"]) for row in trajectories if not bool(row.get("correct"))]
    if require_all_dense_correct and dense_incorrect:
        errors.append(f"Dense trajectories contain incorrect examples: {dense_incorrect}")
    elif dense_incorrect:
        warnings.append(f"{len(dense_incorrect)} dense trajectories are incorrect and should be filtered before bank collection")

    dense_without_final_marker = [
        str(row["id"]) for row in trajectories if not FINAL_ANSWER_RE.search(str(row.get("completion", "")))
    ]
    if dense_without_final_marker:
        warnings.append(
            "Some dense completions do not contain an explicit Final answer marker; inspect possible truncation: "
            f"{dense_without_final_marker}"
        )

    segments = [segment for row in segmented for segment in row.get("segments", [])]
    if not segments:
        errors.append("No reasoning segments were produced")
    standalone_final_count = sum(
        int(str(segment.get("segment_type")) == "final" and bool(FINAL_ANSWER_RE.match(str(segment.get("text", "")))))
        for segment in segments
    )
    completions_with_final = sum(int(bool(FINAL_ANSWER_RE.search(str(row.get("completion", ""))))) for row in segmented)
    if standalone_final_count < completions_with_final:
        errors.append(
            "Some segmented completions contain Final answer markers without a standalone final segment: "
            f"standalone={standalone_final_count}, completions_with_final={completions_with_final}"
        )

    if len(counterfactuals) != len(probe_rows):
        errors.append(f"Counterfactual rows ({len(counterfactuals)}) and probe rows ({len(probe_rows)}) differ")
    if any(str(row.get("module")) != expected_module for row in counterfactuals):
        errors.append(f"Counterfactual rows contain modules other than {expected_module}")
    if expected_layers and any([int(layer) for layer in row.get("pruned_layers", [])] != expected_layers for row in counterfactuals):
        errors.append("Some counterfactual rows do not target the configured runtime layers")

    by_step: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactuals:
        by_step[_step_key(row)].append(row)
    incomplete_steps = []
    for key, rows in by_step.items():
        ratios = {_ratio_key(float(row.get("ratio", -1.0))) for row in rows}
        if ratios != expected_ratio_keys or len(rows) != len(expected_ratios):
            incomplete_steps.append({"step": key, "ratios": sorted(ratios), "rows": len(rows)})
    if incomplete_steps:
        errors.append(f"{len(incomplete_steps)} reasoning steps do not contain the complete ratio grid")

    hidden_indices = [int(row["hidden_index"]) for row in probe_rows if "hidden_index" in row]
    expected_hidden_indices = list(range(len(probe_rows)))
    if hidden_indices != expected_hidden_indices:
        errors.append("Probe hidden_index values are not contiguous and aligned with probe rows")
    hidden_path = Path(paths["probe_hidden_states"])
    if not hidden_path.exists():
        errors.append(f"Missing hidden-state tensor file: {hidden_path}")

    ratio_counts = Counter(float(row["ratio"]) for row in counterfactuals)
    ratio_flip_rates = {}
    for ratio in expected_ratios:
        rows = [row for row in counterfactuals if _ratio_key(float(row["ratio"])) == _ratio_key(ratio)]
        ratio_flip_rates[f"{ratio:.2f}"] = sum(int(bool(row["flipped"])) for row in rows) / len(rows) if rows else None
    ratio_zero_flip_rate = ratio_flip_rates.get("0.00")
    if ratio_zero_flip_rate is None:
        errors.append("Missing ratio=0 control rows")
    elif ratio_zero_flip_rate > max_ratio_zero_flip_rate:
        message = (
            f"ratio=0 control flip rate is high: {ratio_zero_flip_rate:.4f} > {max_ratio_zero_flip_rate:.4f}; "
            "unstable continuation steps should be filtered before router training"
        )
        if allow_ratio_zero_filtering:
            warnings.append(message)
        else:
            errors.append(message)

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "dense_trajectories": len(trajectories),
        "dense_correct": len(trajectories) - len(dense_incorrect),
        "segmented_trajectories": len(segmented),
        "reasoning_segments": len(segments),
        "standalone_final_segments": standalone_final_count,
        "counterfactual_rows": len(counterfactuals),
        "probe_rows": len(probe_rows),
        "expected_module": expected_module,
        "expected_ratios": expected_ratios,
        "ratio_counts": {f"{ratio:.2f}": ratio_counts[ratio] for ratio in expected_ratios},
        "ratio_flip_rates": ratio_flip_rates,
        "ratio_zero_flip_rate": ratio_zero_flip_rate,
        "reasoning_steps": len(by_step),
        "expected_runtime_layers": expected_layers,
        "hidden_indices_contiguous": hidden_indices == expected_hidden_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    summary = validate_runtime_bank(cfg)
    output = args.output or cfg["paths"].get(
        "runtime_bank_validation",
        str(Path(cfg["paths"]["run_dir"]) / "07_runtime_bank_validation.json"),
    )
    write_json(output, summary)
    print(f"runtime-bank validation: {summary['status']}")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    for error in summary["errors"]:
        print(f"error: {error}")
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
