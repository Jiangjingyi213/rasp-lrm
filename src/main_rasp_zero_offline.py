from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.main_offline_baselines import (
    choice_strength,
    group_steps,
    oracle_choice_key,
    pick_row,
    problem_key,
    row_strength,
    synthetic_dense_row,
)
DEFAULT_BUDGETS = [0.2, 0.4, 0.6]
DEFAULT_RATIOS = [0.2, 0.4, 0.6]
DEFAULT_STAGE_CAPS = {
    "verification": 0.2,
    "final": 0.2,
    "derivation": 0.4,
    "planning": 0.6,
    "understanding": 0.6,
    "unknown": 0.4,
}


def _step_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]), int(row["segment_id"]))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL row in {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_action_rows(configs: list[str], inputs: list[str]) -> list[dict[str, Any]]:
    paths = list(inputs)
    if configs:
        from src.utils.io import read_yaml

        for config in configs:
            paths.append(read_yaml(config)["paths"]["probe_dataset"])
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _normalize(values: dict[tuple[str, str, int], float]) -> dict[tuple[str, str, int], float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _step_scores(steps: dict[tuple[str, str, int], list[dict[str, Any]]], name: str) -> dict[tuple[str, str, int], float]:
    if name == "entropy":
        return _normalize({key: float(rows[0].get("entropy", 0.0)) for key, rows in steps.items()})
    if name == "confidence":
        return _normalize({key: 1.0 - float(rows[0].get("confidence", 0.0)) for key, rows in steps.items()})
    raise ValueError(f"Unsupported built-in score: {name}")


def load_probe_step_scores(path: str | Path) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[_step_key(row)].append(float(row["risk_score"]))
    return {key: mean(values) for key, values in grouped.items()}


def _available_ratios(rows: list[dict[str, Any]], module: str, requested: list[float]) -> list[float]:
    present = {float(row.get("ratio", 0.0)) for row in rows if row.get("module") == module}
    ratios = sorted(ratio for ratio in requested if ratio in present)
    if not ratios:
        raise ValueError(f"No rows for module={module!r} and requested ratios={requested}")
    return ratios


def _ratio_caps(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
    ratios: list[float],
    stage_caps: dict[str, float] | None,
) -> dict[tuple[str, str, int], float]:
    caps = {}
    for key, rows in steps.items():
        if stage_caps is None:
            caps[key] = max(ratios)
            continue
        stage = str(rows[0].get("segment_type", "unknown"))
        cap = float(stage_caps.get(stage, stage_caps.get("unknown", max(ratios))))
        allowed = [ratio for ratio in ratios if ratio <= cap + 1e-9]
        caps[key] = max(allowed, default=0.0)
    return caps


def _allocate_ratios(
    scores: dict[tuple[str, str, int], float],
    caps: dict[tuple[str, str, int], float],
    ratios: list[float],
    target_average_ratio: float,
) -> dict[tuple[str, str, int], float]:
    selected = {key: 0.0 for key in scores}
    target_total = target_average_ratio * len(scores)
    candidates = []
    for key, score in scores.items():
        for ratio in ratios:
            if ratio <= caps[key] + 1e-9:
                candidates.append((float(score), ratio, key))
    # Lower predicted risk receives compute reduction first. Lower ratio levels
    # are selected before larger reductions when risk scores are tied.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    total = 0.0
    for _score, ratio, key in candidates:
        previous = selected[key]
        increment = ratio - previous
        if increment <= 1e-9:
            continue
        if total + increment > target_total + 1e-9:
            continue
        selected[key] = ratio
        total += increment
    return selected


def select_budget_policy(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
    scores: dict[tuple[str, str, int], float],
    module: str,
    ratios: list[float],
    target_average_ratio: float,
    stage_caps: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    missing = set(steps) - set(scores)
    if missing:
        raise ValueError(f"Scores missing for {len(missing)} problem-steps")
    caps = _ratio_caps(steps, ratios, stage_caps)
    selected_ratios = _allocate_ratios(scores, caps, ratios, target_average_ratio)
    selected = []
    for key, rows in steps.items():
        ratio = selected_ratios[key]
        row = synthetic_dense_row(rows) if ratio <= 0 else dict(pick_row(rows, module, ratio))
        row["policy_risk_score"] = float(scores[key])
        row["policy_stage_cap"] = float(caps[key])
        row["policy_target_average_ratio"] = float(target_average_ratio)
        selected.append(row)
    return selected


def select_static_policy(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
    module: str,
    ratio: float,
) -> list[dict[str, Any]]:
    return [dict(pick_row(rows, module, ratio)) for rows in steps.values()]


def select_safe_step_oracle(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    selected = []
    for rows in steps.values():
        candidates = [synthetic_dense_row(rows), *rows]
        selected.append(min(candidates, key=lambda row: (int(bool(row["flipped"])), -row_strength(row), oracle_choice_key(row))))
    return selected


def select_safe_prompt_oracle(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_problem_choice: dict[tuple[tuple[str, str], str], list[dict[str, Any]]] = defaultdict(list)
    choices: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rows in steps.values():
        for row in rows:
            pkey = problem_key(row)
            choice = oracle_choice_key(row)
            by_problem_choice[(pkey, choice)].append(row)
            choices[pkey].add(choice)
    best = {}
    for pkey, problem_choices in choices.items():
        best[pkey] = min(
            problem_choices,
            key=lambda choice: (
                sum(int(bool(row["flipped"])) for row in by_problem_choice[(pkey, choice)]) / len(by_problem_choice[(pkey, choice)]),
                -choice_strength(choice),
                choice,
            ),
        )
    return [dict(next(row for row in rows if oracle_choice_key(row) == best[problem_key(rows[0])])) for rows in steps.values()]


def summarize(name: str, selected: list[dict[str, Any]], target_ratio: float | None = None) -> dict[str, Any]:
    n = len(selected)
    flips = sum(int(bool(row["flipped"])) for row in selected)
    ratios = [float(row.get("ratio", 0.0)) for row in selected]
    strengths = [row_strength(row) for row in selected]
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_stage[str(row.get("segment_type", "unknown"))].append(row)
        by_dataset[str(row.get("dataset", "unknown"))].append(row)
    return {
        "policy": name,
        "target_average_pruning_ratio": target_ratio,
        "n_problem_steps": n,
        "selected_action_flip_rate": flips / n if n else None,
        "average_pruning_ratio": sum(ratios) / n if n else None,
        "average_pruning_strength_proxy": sum(strengths) / n if n else None,
        "module_distribution": dict(sorted(Counter(str(row.get("module", "none")) for row in selected).items())),
        "stage_flip_rates": {
            stage: sum(int(bool(row["flipped"])) for row in rows) / len(rows) for stage, rows in sorted(by_stage.items())
        },
        "dataset_flip_rates": {
            dataset: sum(int(bool(row["flipped"])) for row in rows) / len(rows) for dataset, rows in sorted(by_dataset.items())
        },
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fields = [
        "policy",
        "target_average_pruning_ratio",
        "n_problem_steps",
        "selected_action_flip_rate",
        "average_pruning_ratio",
        "average_pruning_strength_proxy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--probe-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--module", default="mlp_block")
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    args = parser.parse_args()

    rows = load_action_rows(args.configs, args.inputs)
    if not rows:
        raise ValueError("No counterfactual rows found")
    steps = group_steps(rows)
    ratios = _available_ratios(rows, args.module, args.ratios)
    entropy_scores = _step_scores(steps, "entropy")
    confidence_scores = _step_scores(steps, "confidence")
    probe_scores = load_probe_step_scores(args.probe_scores)
    output_dir = ensure_dir(args.output_dir)
    summaries = []
    selected_rows = []

    def add(policy: str, selected: list[dict[str, Any]], target: float | None = None) -> None:
        summaries.append(summarize(policy, selected, target_ratio=target))
        for row in selected:
            out = dict(row)
            out["policy"] = policy
            selected_rows.append(out)

    add("dense_lrm", [synthetic_dense_row(group) for group in steps.values()], 0.0)
    for budget in args.budgets:
        add(f"static_{args.module}_r{budget:.2f}", select_static_policy(steps, args.module, budget), budget)
        add(
            f"entropy_only_budget_{budget:.2f}",
            select_budget_policy(steps, entropy_scores, args.module, ratios, budget),
            budget,
        )
        add(
            f"confidence_only_budget_{budget:.2f}",
            select_budget_policy(steps, confidence_scores, args.module, ratios, budget),
            budget,
        )
        add(
            f"hidden_probe_budget_{budget:.2f}",
            select_budget_policy(steps, probe_scores, args.module, ratios, budget),
            budget,
        )
        add(
            f"rasp_zero_stage_budget_{budget:.2f}",
            select_budget_policy(steps, probe_scores, args.module, ratios, budget, stage_caps=DEFAULT_STAGE_CAPS),
            budget,
        )
    add("prompt_router_safe_oracle", select_safe_prompt_oracle(steps))
    add("step_safe_oracle", select_safe_step_oracle(steps))

    result = {
        "method": "RASP-Zero offline policy evaluation",
        "scope": "Counterfactual policy simulation only; pruning ratios and strength values are proxies, not measured runtime speedups.",
        "n_counterfactual_rows": len(rows),
        "n_problem_steps": len(steps),
        "selection_unit": "one action per dataset/problem/segment",
        "module": args.module,
        "available_ratios": ratios,
        "stage_caps": DEFAULT_STAGE_CAPS,
        "policies": summaries,
    }
    write_json(output_dir / "rasp_zero_offline_summary.json", result)
    write_csv(output_dir / "rasp_zero_offline_summary.csv", summaries)
    write_jsonl(output_dir / "rasp_zero_selected_actions.jsonl", selected_rows)


if __name__ == "__main__":
    main()
