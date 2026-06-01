from __future__ import annotations

from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_oracles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    by_choice: dict[str, list[dict[str, Any]]] = {}
    by_problem_choice: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_problem_step_choice: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    problem_choices: dict[str, set[str]] = {}
    problem_steps: dict[tuple[str, int], set[str]] = {}

    for row in rows:
        item_id = str(row["id"])
        segment_id = int(row["segment_id"])
        choice = oracle_choice_key(row)
        by_choice.setdefault(choice, []).append(row)
        by_problem_choice.setdefault((item_id, choice), []).append(row)
        by_problem_step_choice.setdefault((item_id, segment_id, choice), []).append(row)
        problem_choices.setdefault(item_id, set()).add(choice)
        problem_steps.setdefault((item_id, segment_id), set()).add(choice)

    static_choice_scores = {
        choice: sum(int(row["flipped"]) for row in group) / len(group)
        for choice, group in by_choice.items()
    }
    static_best_choice, static_best_rate = max(static_choice_scores.items(), key=lambda x: x[1])
    static_safe_choice, static_safe_rate = min(static_choice_scores.items(), key=lambda x: x[1])

    prompt_num = 0
    prompt_den = 0
    prompt_macro_rates: list[float] = []
    prompt_best_choices: dict[str, str] = {}
    prompt_safe_num = 0
    prompt_safe_den = 0
    prompt_safe_macro_rates: list[float] = []
    prompt_safe_choices: dict[str, str] = {}
    for item_id, choices in problem_choices.items():
        best_choice = max(
            choices,
            key=lambda choice: sum(int(row["flipped"]) for row in by_problem_choice[(item_id, choice)])
            / len(by_problem_choice[(item_id, choice)]),
        )
        best_rows = by_problem_choice[(item_id, best_choice)]
        flips = sum(int(row["flipped"]) for row in best_rows)
        prompt_num += flips
        prompt_den += len(best_rows)
        prompt_macro_rates.append(flips / len(best_rows))
        prompt_best_choices[item_id] = best_choice
        safe_choice = min(
            choices,
            key=lambda choice: sum(int(row["flipped"]) for row in by_problem_choice[(item_id, choice)])
            / len(by_problem_choice[(item_id, choice)]),
        )
        safe_rows = by_problem_choice[(item_id, safe_choice)]
        safe_flips = sum(int(row["flipped"]) for row in safe_rows)
        prompt_safe_num += safe_flips
        prompt_safe_den += len(safe_rows)
        prompt_safe_macro_rates.append(safe_flips / len(safe_rows))
        prompt_safe_choices[item_id] = safe_choice

    step_num = 0
    step_den = 0
    step_macro_rates: list[float] = []
    step_safe_num = 0
    step_safe_den = 0
    step_safe_macro_rates: list[float] = []
    for (item_id, segment_id), choices in problem_steps.items():
        best_choice = max(
            choices,
            key=lambda choice: sum(int(row["flipped"]) for row in by_problem_step_choice[(item_id, segment_id, choice)])
            / len(by_problem_step_choice[(item_id, segment_id, choice)]),
        )
        best_rows = by_problem_step_choice[(item_id, segment_id, best_choice)]
        flips = sum(int(row["flipped"]) for row in best_rows)
        step_num += flips
        step_den += len(best_rows)
        step_macro_rates.append(flips / len(best_rows))
        safe_choice = min(
            choices,
            key=lambda choice: sum(int(row["flipped"]) for row in by_problem_step_choice[(item_id, segment_id, choice)])
            / len(by_problem_step_choice[(item_id, segment_id, choice)]),
        )
        safe_rows = by_problem_step_choice[(item_id, segment_id, safe_choice)]
        safe_flips = sum(int(row["flipped"]) for row in safe_rows)
        step_safe_num += safe_flips
        step_safe_den += len(safe_rows)
        step_safe_macro_rates.append(safe_flips / len(safe_rows))

    return {
        "static_oracle_best_choice": static_best_choice,
        "static_oracle_flip_rate": static_best_rate,
        "static_safe_oracle_best_choice": static_safe_choice,
        "static_safe_oracle_flip_rate": static_safe_rate,
        "prompt_oracle_flip_rate": prompt_num / prompt_den,
        "step_oracle_flip_rate": step_num / step_den,
        "macro_prompt_oracle_flip_rate": _mean(prompt_macro_rates),
        "macro_step_oracle_flip_rate": _mean(step_macro_rates),
        "prompt_safe_oracle_flip_rate": prompt_safe_num / prompt_safe_den,
        "step_safe_oracle_flip_rate": step_safe_num / step_safe_den,
        "macro_prompt_safe_oracle_flip_rate": _mean(prompt_safe_macro_rates),
        "macro_step_safe_oracle_flip_rate": _mean(step_safe_macro_rates),
        "choice_flip_rates": {str(k): v for k, v in sorted(static_choice_scores.items())},
        "prompt_oracle_best_choices": dict(sorted(prompt_best_choices.items())),
        "prompt_safe_oracle_best_choices": dict(sorted(prompt_safe_choices.items())),
        "n_counterfactuals": len(rows),
        "n_problem_steps": len(problem_steps),
        "oracle_averaging": (
            "Legacy oracle fields maximize flip rate for sensitivity diagnostics; "
            "safe oracle fields minimize flip rate for routing upper bounds. "
            "Micro rates are comparable over problem-step decisions; macro rates are diagnostic averages."
        ),
    }


def oracle_choice_key(row: dict[str, Any]) -> str:
    module = row.get("module") or row.get("unit", "layer")
    ratio = row.get("ratio", 1.0)
    layers = row.get("pruned_layers")
    if not layers:
        layers = [row.get("layer_id")]
    return f"{module}|r={float(ratio):.2f}|layers={','.join(str(x) for x in layers)}"
