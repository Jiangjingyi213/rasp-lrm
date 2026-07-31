from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_json, write_jsonl


STAGES = ("setup", "reasoning", "verify", "final")


def _positions(span: dict[str, Any]) -> list[tuple[int, int]]:
    """Choose action positions relative to stage content, never future trajectory length."""

    start = int(span["content_start_token"])
    end = int(span["content_end_token"])
    length = end - start
    if length < 2:
        return []
    # These are stage-relative anchors, not absolute decoding boundaries.
    # Retaining the final valid token prevents the typically short final stage
    # from disappearing from the bank merely because it cannot fit 64 tokens.
    offsets = [min(16, length - 1), min(64, length - 1), length - 1]
    return [(start + offset, offset) for offset in sorted(set(offsets))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.20, 0.30, 0.40])
    parser.add_argument("--window-tokens", type=int, default=64)
    parser.add_argument("--min-stage-states", type=int, default=100)
    parser.add_argument("--min-final-states", type=int, default=60)
    args = parser.parse_args()
    ratios = tuple(sorted({float(value) for value in args.ratios}))
    if ratios != (0.0, 0.2, 0.3, 0.4):
        raise ValueError("Stage-risk v1 requires the fixed ratio grid 0/0.2/0.3/0.4")
    tasks = []
    seen = set()
    for path in args.trajectories:
        for row in read_jsonl(path):
            protocol = row.get("stage_protocol", {})
            if not bool(row.get("correct")) or not bool(protocol.get("valid")):
                continue
            identity = (str(row.get("dataset")), str(row.get("id")))
            if identity in seen:
                continue
            spans = {str(item["stage"]): item for item in protocol.get("stage_spans", [])}
            if set(spans) != set(STAGES):
                continue
            generated = row.get("generated_token_ids")
            if not isinstance(generated, list):
                continue
            seen.add(identity)
            for stage in STAGES:
                for position, local_offset in _positions(spans[stage]):
                    remaining = len(generated) - position - 1
                    if remaining < 1:
                        continue
                    tasks.append(
                        {
                            "schema": "stage_risk_relative_counterfactual_task_v1",
                            "dataset": row["dataset"], "id": row["id"], "question": row["question"],
                            "gold": row.get("gold"), "prompt": row["prompt"],
                            "parent_completion": row["completion"],
                            "parent_generated_token_ids": generated,
                            "stage": stage,
                            "source": str(row["dataset"]),
                            "stage_local_tokens": local_offset,
                            "stage_local_position": "early" if local_offset <= 16 else "late",
                            "action_position": position,
                            "forced_prefix_ids": generated[:position],
                            "action_ratios": list(ratios),
                            "window_tokens": int(args.window_tokens),
                            "effective_window_tokens": min(int(args.window_tokens), remaining),
                            "terminal_window": remaining < int(args.window_tokens),
                        }
                    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter(item["stage"] for item in tasks)
    stratified_counts = Counter(
        f"{item['stage']}::{item['source']}::{item['stage_local_position']}" for item in tasks
    )
    required = {stage: args.min_final_states if stage == "final" else args.min_stage_states for stage in STAGES}
    # This is only a parent-derived candidate count. Validity is established
    # after counterfactual replay, so it must not be mistaken for coverage.
    status = "candidate_pool_ready" if all(counts[stage] >= required[stage] for stage in STAGES) else "needs_more_parents"
    write_jsonl(output / "01_counterfactual_tasks.jsonl", tasks)
    write_json(
        output / "manifest.json",
        {
            "schema": "stage_risk_relative_counterfactual_tasks_v1",
            "source_trajectories": args.trajectories,
            "tasks": len(tasks),
            "action_ratios": list(ratios),
            "window_tokens": int(args.window_tokens),
            "parent_filter": "dense_correct_and_protocol_valid",
        },
    )
    write_json(
        output / "status.json",
        {
            "status": status,
            "stage_counts": dict(counts),
            "stage_source_position_counts": dict(stratified_counts),
            "required": required,
        },
    )


if __name__ == "__main__":
    main()
