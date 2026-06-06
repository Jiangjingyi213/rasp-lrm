from __future__ import annotations

from collections import defaultdict
from typing import Any


def step_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]), int(row["segment_id"]))


def group_steps(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[step_key(row)].append(row)
    return dict(grouped)


def safe_ratio_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row.get("ratio", 0.0)))
    safe_ratios = [float(row.get("ratio", 0.0)) for row in ordered if not bool(row.get("flipped"))]
    max_safe_ratio = max(safe_ratios, default=0.0)
    monotonic_safe_ratio = 0.0
    for row in ordered:
        ratio = float(row.get("ratio", 0.0))
        if bool(row.get("flipped")):
            break
        monotonic_safe_ratio = ratio
    reference = ordered[0]
    return {
        "dataset": reference.get("dataset"),
        "id": reference["id"],
        "segment_id": int(reference["segment_id"]),
        "segment_index": int(reference.get("segment_index", reference["segment_id"])),
        "num_segments": int(reference.get("num_segments", 1)),
        "segment_type": reference.get("segment_type", "unknown"),
        "tested_ratios": [float(row.get("ratio", 0.0)) for row in ordered],
        "safe_ratios": safe_ratios,
        "max_safe_ratio": max_safe_ratio,
        "monotonic_safe_ratio": monotonic_safe_ratio,
        "non_monotonic": max_safe_ratio > monotonic_safe_ratio,
    }


def build_safe_oracle_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [safe_ratio_summary(step_rows) for step_rows in group_steps(rows).values()]


def allocate_budget_aware_safe_oracle(
    safe_steps: list[dict[str, Any]],
    target_average_ratio: float,
    *,
    ratio_field: str = "max_safe_ratio",
) -> list[dict[str, Any]]:
    selected = [0.0] * len(safe_steps)
    target_total = float(target_average_ratio) * len(safe_steps)
    candidates = []
    for index, step in enumerate(safe_steps):
        cap = float(step[ratio_field])
        for ratio in sorted(float(value) for value in step["tested_ratios"] if 0.0 < float(value) <= cap + 1e-9):
            candidates.append((ratio, index))
    candidates.sort()
    total = 0.0
    for ratio, index in candidates:
        increment = ratio - selected[index]
        if increment <= 1e-9:
            continue
        if total + increment > target_total + 1e-9:
            continue
        selected[index] = ratio
        total += increment
    output = []
    for step, ratio in zip(safe_steps, selected):
        output.append({**step, "selected_ratio": ratio, "ratio_field": ratio_field})
    return output


def available_prefix_budget(
    target_average_ratio: float,
    selected_ratios: list[float],
) -> float:
    """Return the largest next action that keeps the prefix average on budget."""

    return max(
        0.0,
        float(target_average_ratio) * (len(selected_ratios) + 1) - sum(selected_ratios),
    )


def allocate_causal_budget_aware_safe_oracle(
    safe_steps: list[dict[str, Any]],
    target_average_ratio: float,
    *,
    ratio_field: str = "monotonic_safe_ratio",
) -> list[dict[str, Any]]:
    """Allocate safe ratios independently per problem in reasoning-step order.

    Unlike the dataset-level diagnostic oracle above, this allocator never
    borrows budget from another problem or from a future step. Its decisions
    therefore obey the same prefix-average constraint as the runtime policy.
    """

    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for input_index, step in enumerate(safe_steps):
        problem = (str(step.get("dataset") or "unknown"), str(step["id"]))
        grouped[problem].append((input_index, step))

    output: list[dict[str, Any] | None] = [None] * len(safe_steps)
    for problem_steps in grouped.values():
        problem_steps.sort(
            key=lambda item: (
                int(item[1].get("segment_index", item[1].get("segment_id", 0))),
                int(item[1].get("segment_id", 0)),
            )
        )
        selected_history: list[float] = []
        for input_index, step in problem_steps:
            available = available_prefix_budget(target_average_ratio, selected_history)
            cap = float(step[ratio_field])
            feasible = [
                float(value)
                for value in step["tested_ratios"]
                if float(value) <= cap + 1e-9 and float(value) <= available + 1e-9
            ]
            selected_ratio = max(feasible, default=0.0)
            output[input_index] = {
                **step,
                "selected_ratio": selected_ratio,
                "ratio_field": ratio_field,
                "available_budget_before_selection": available,
                "prefix_average_after_selection": (
                    sum(selected_history) + selected_ratio
                )
                / (len(selected_history) + 1),
                "oracle_allocation": "causal_problem_prefix",
            }
            selected_history.append(selected_ratio)

    if any(row is None for row in output):
        raise RuntimeError("Causal oracle failed to allocate every reasoning step")
    return [row for row in output if row is not None]
