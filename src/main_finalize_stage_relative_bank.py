from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_json, write_jsonl


STAGES = ("setup", "reasoning", "verify", "final")
RATIOS = (0.0, 0.20, 0.30, 0.40)


def _state_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["dataset"]), str(row["id"]), str(row["stage"]), int(row["action_position"])
    )


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    answer_flipped = bool(row.get("answer_flipped", False))
    protocol_invalid = not bool(row.get("stage_protocol", {}).get("valid", False))
    truncated = bool(row.get("truncated", False))
    collection_error = row.get("collection_error")
    return {
        "ratio": float(row["ratio"]),
        "mask_score_margin": float(row["mask_score_margin"]),
        "unsafe": bool(answer_flipped or protocol_invalid or truncated or collection_error),
        "answer_flipped": answer_flipped,
        "protocol_invalid": protocol_invalid,
        "truncated": truncated,
        "collection_error": str(collection_error) if collection_error else None,
        "actual_pruning": float(row.get("actual_pruning", row["ratio"])),
        "next_token_logit_divergence": float(row.get("next_token_logit_divergence", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize stage-relative action replays into the controller training bank."
    )
    parser.add_argument("--raw-results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-stage-states", type=int, default=100)
    parser.add_argument("--min-final-states", type=int, default=60)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in args.raw_results:
        for row in read_jsonl(path):
            if row.get("schema") != "stage_risk_relative_counterfactual_result_v1":
                raise ValueError(f"Unexpected raw-result schema in {path}")
            if str(row.get("stage")) not in STAGES:
                raise ValueError(f"Unknown stage in {path}: {row.get('stage')!r}")
            grouped[_state_key(row)].append(row)

    normalized = []
    excluded = []
    coverage = Counter()
    for key, rows in sorted(grouped.items()):
        ratios = {float(row["ratio"]) for row in rows}
        if ratios != set(RATIOS):
            excluded.append({"state": key, "reason": "incomplete_ratio_grid", "ratios": sorted(ratios)})
            continue
        first = rows[0]
        candidates = [_candidate(row) for row in sorted(rows, key=lambda value: float(value["ratio"]))]
        # A state is usable even when an action is unsafe. It is excluded only
        # when replay itself failed to provide a complete causal comparison.
        if any(candidate["collection_error"] for candidate in candidates):
            excluded.append({"state": key, "reason": "collection_error"})
            continue
        record = {
            "schema": "stage_risk_controller_dataset_v1",
            "dataset": first["dataset"],
            "id": first["id"],
            "stage": first["stage"],
            "source": first.get("source", first["dataset"]),
            "stage_local_position": first["stage_local_position"],
            "stage_local_tokens": int(first["stage_local_tokens"]),
            "entropy": float(first["entropy"]),
            "confidence": float(first["confidence"]),
            "activation_ood": float(first["activation_ood"]),
            "activation_drift": float(first["activation_drift"]),
            "candidates": candidates,
        }
        normalized.append(record)
        coverage[f"{record['stage']}::{record['source']}::{record['stage_local_position']}"] += 1

    stage_counts = Counter(row["stage"] for row in normalized)
    required = {stage: args.min_final_states if stage == "final" else args.min_stage_states for stage in STAGES}
    missing = {stage: max(0, required[stage] - stage_counts[stage]) for stage in STAGES}
    status = "coverage_passed" if not any(missing.values()) else "needs_more_counterfactuals"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "stage_risk_controller_dataset.jsonl", normalized)
    write_jsonl(output / "excluded_states.jsonl", excluded)
    write_json(
        output / "manifest.json",
        {
            "schema": "stage_risk_controller_dataset_v1",
            "raw_results": args.raw_results,
            "states": len(normalized),
            "excluded_states": len(excluded),
            "ratios": list(RATIOS),
            "unsafe_definition": "answer_flip_or_protocol_invalid_or_truncated",
        },
    )
    write_json(
        output / "status.json",
        {
            "status": status,
            "valid_state_counts": dict(stage_counts),
            "required": required,
            "missing": missing,
            "stage_source_position_counts": dict(coverage),
        },
    )


if __name__ == "__main__":
    main()
