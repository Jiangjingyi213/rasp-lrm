from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from src.probes.rasp_train_dataset import DEFAULT_RATIOS
from src.rasp.safe_oracle import allocate_causal_budget_aware_safe_oracle, build_safe_oracle_steps, step_key
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def _step_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[step_key(row)].append(row)
    return dict(grouped)


def _position(row: dict[str, Any]) -> float:
    num_segments = max(1, int(row.get("num_segments", 1)))
    segment_index = int(row.get("segment_index", row.get("segment_id", 0)))
    return segment_index / max(1, num_segments - 1)


def _row_for_step(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row.get("ratio", 0.0)))
    for row in ordered:
        if abs(float(row.get("ratio", -1.0))) < 1e-9:
            return row
    return ordered[0]


def _candidate_flipped(rows: list[dict[str, Any]], ratios: list[float]) -> list[bool]:
    by_ratio = {round(float(row.get("ratio", 0.0)), 8): bool(row.get("flipped")) for row in rows}
    return [bool(by_ratio.get(round(float(ratio), 8), True)) for ratio in ratios]


def _candidate_unsafe(
    candidate_flipped: list[bool],
    ratios: list[float],
    monotonic_safe_ratio: float,
) -> list[bool]:
    return [
        bool(flipped) or float(ratio) > float(monotonic_safe_ratio) + 1e-9
        for flipped, ratio in zip(candidate_flipped, ratios)
    ]


def _problem_budget_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(f"{row.get('dataset')}::{row.get('id')}" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dataset", required=True)
    parser.add_argument("--merged-hidden-states", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.15, 0.20])
    parser.add_argument("--ratio-field", default="monotonic_safe_ratio", choices=["monotonic_safe_ratio", "max_safe_ratio"])
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    args = parser.parse_args()

    ratios = sorted({float(value) for value in args.ratios})
    output_dir = ensure_dir(args.output_dir)
    rows = read_jsonl(args.merged_dataset)
    hidden = torch.load(args.merged_hidden_states, map_location="cpu")
    if len(rows) != len(hidden):
        raise ValueError(f"rows ({len(rows)}) and hidden states ({len(hidden)}) differ")

    grouped = _step_rows(rows)
    safe_steps = build_safe_oracle_steps(rows)
    safe_by_key = {step_key(step): step for step in safe_steps}
    selected_by_budget = {}
    for budget in args.budgets:
        selected = allocate_causal_budget_aware_safe_oracle(safe_steps, budget, ratio_field=args.ratio_field)
        selected_by_budget[float(budget)] = {step_key(row): row for row in selected}

    policy_rows: list[dict[str, Any]] = []
    policy_hidden: list[torch.Tensor] = []
    for key, step_rows in sorted(grouped.items()):
        reference = _row_for_step(step_rows)
        hidden_state = hidden[int(reference["hidden_index"])]
        safe_step = safe_by_key[key]
        candidate_flipped = _candidate_flipped(step_rows, ratios)
        candidate_unsafe = _candidate_unsafe(
            candidate_flipped,
            ratios,
            float(safe_step["monotonic_safe_ratio"]),
        )
        for budget in args.budgets:
            selected = selected_by_budget[float(budget)][key]
            policy_rows.append(
                {
                    "dataset": reference.get("dataset"),
                    "id": reference["id"],
                    "segment_id": int(reference["segment_id"]),
                    "segment_index": int(reference.get("segment_index", reference["segment_id"])),
                    "num_segments": int(reference.get("num_segments", 1)),
                    "segment_type": reference.get("segment_type", "unknown"),
                    "entropy": float(reference.get("entropy", 0.0)),
                    "confidence": float(reference.get("confidence", 0.0)),
                    "position": _position(reference),
                    "target_budget": float(budget),
                    "available_budget_before_selection": float(selected["available_budget_before_selection"]),
                    "ratio_field": args.ratio_field,
                    "oracle_ratio": float(selected["selected_ratio"]),
                    "max_safe_ratio": float(safe_step["max_safe_ratio"]),
                    "monotonic_safe_ratio": float(safe_step["monotonic_safe_ratio"]),
                    "non_monotonic": bool(safe_step["non_monotonic"]),
                    "candidate_ratios": ratios,
                    "candidate_flipped": candidate_flipped,
                    "candidate_unsafe": candidate_unsafe,
                }
            )
            policy_hidden.append(hidden_state)

    hidden_out = torch.stack(policy_hidden)
    write_jsonl(output_dir / "11_rasp_train_policy_dataset.jsonl", policy_rows)
    torch.save(hidden_out, output_dir / "11_rasp_train_policy_hidden_states.pt")
    budget_summaries = []
    for budget in args.budgets:
        budget_rows = [row for row in policy_rows if abs(float(row["target_budget"]) - float(budget)) < 1e-9]
        average_ratio = sum(float(row["oracle_ratio"]) for row in budget_rows) / max(1, len(budget_rows))
        budget_summaries.append(
            {
                "target_budget": float(budget),
                "average_oracle_ratio": average_ratio,
                "budget_utilization": average_ratio / float(budget) if budget > 0 else None,
                "oracle_ratio_distribution": dict(
                    sorted(Counter(f"{float(row['oracle_ratio']):.2f}" for row in budget_rows).items())
                ),
            }
        )
    summary = {
        "method": "rasp_train_policy_data",
        "oracle_allocation": "causal_problem_prefix",
        "source_dataset": args.merged_dataset,
        "ratio_field": args.ratio_field,
        "target_budgets": [float(value) for value in args.budgets],
        "candidate_ratios": ratios,
        "problem_count": len({(str(row.get("dataset")), str(row["id"])) for row in policy_rows}),
        "problem_step_count": len(safe_steps),
        "policy_rows": len(policy_rows),
        "problem_budget_rows": _problem_budget_counts(policy_rows),
        "oracle_ratio_distribution": dict(sorted(Counter(f"{float(row['oracle_ratio']):.2f}" for row in policy_rows).items())),
        "target_budget_distribution": dict(sorted(Counter(f"{float(row['target_budget']):.2f}" for row in policy_rows).items())),
        "budget_summaries": budget_summaries,
    }
    write_json(output_dir / "11_rasp_train_policy_data_summary.json", summary)


if __name__ == "__main__":
    main()
