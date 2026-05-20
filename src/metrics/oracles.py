from __future__ import annotations

from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_oracles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    by_layer: dict[int, list[dict[str, Any]]] = {}
    by_problem_layer: dict[tuple[str, int], list[dict[str, Any]]] = {}
    by_problem_step_layer: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    problem_layers: dict[str, set[int]] = {}
    problem_steps: dict[tuple[str, int], set[int]] = {}

    for row in rows:
        item_id = str(row["id"])
        segment_id = int(row["segment_id"])
        layer_id = int(row["layer_id"])
        by_layer.setdefault(layer_id, []).append(row)
        by_problem_layer.setdefault((item_id, layer_id), []).append(row)
        by_problem_step_layer.setdefault((item_id, segment_id, layer_id), []).append(row)
        problem_layers.setdefault(item_id, set()).add(layer_id)
        problem_steps.setdefault((item_id, segment_id), set()).add(layer_id)

    static_layer_scores = {
        layer_id: sum(int(row["flipped"]) for row in group) / len(group)
        for layer_id, group in by_layer.items()
    }
    static_best_layer, static_best_rate = max(static_layer_scores.items(), key=lambda x: x[1])

    prompt_num = 0
    prompt_den = 0
    prompt_macro_rates: list[float] = []
    prompt_best_layers: dict[str, int] = {}
    for item_id, layer_ids in problem_layers.items():
        best_layer = max(
            layer_ids,
            key=lambda layer_id: sum(int(row["flipped"]) for row in by_problem_layer[(item_id, layer_id)])
            / len(by_problem_layer[(item_id, layer_id)]),
        )
        best_rows = by_problem_layer[(item_id, best_layer)]
        flips = sum(int(row["flipped"]) for row in best_rows)
        prompt_num += flips
        prompt_den += len(best_rows)
        prompt_macro_rates.append(flips / len(best_rows))
        prompt_best_layers[item_id] = best_layer

    step_num = 0
    step_den = 0
    step_macro_rates: list[float] = []
    for (item_id, segment_id), layer_ids in problem_steps.items():
        best_layer = max(
            layer_ids,
            key=lambda layer_id: sum(int(row["flipped"]) for row in by_problem_step_layer[(item_id, segment_id, layer_id)])
            / len(by_problem_step_layer[(item_id, segment_id, layer_id)]),
        )
        best_rows = by_problem_step_layer[(item_id, segment_id, best_layer)]
        flips = sum(int(row["flipped"]) for row in best_rows)
        step_num += flips
        step_den += len(best_rows)
        step_macro_rates.append(flips / len(best_rows))

    return {
        "static_oracle_best_layer": static_best_layer,
        "static_oracle_flip_rate": static_best_rate,
        "prompt_oracle_flip_rate": prompt_num / prompt_den,
        "step_oracle_flip_rate": step_num / step_den,
        "macro_prompt_oracle_flip_rate": _mean(prompt_macro_rates),
        "macro_step_oracle_flip_rate": _mean(step_macro_rates),
        "layer_flip_rates": {str(k): v for k, v in sorted(static_layer_scores.items())},
        "prompt_oracle_best_layers": dict(sorted(prompt_best_layers.items())),
        "n_counterfactuals": len(rows),
        "n_problem_steps": len(problem_steps),
        "oracle_averaging": "micro rates are comparable over problem-step decisions; macro rates are diagnostic averages",
    }
