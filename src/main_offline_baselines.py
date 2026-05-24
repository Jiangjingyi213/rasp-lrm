from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.metrics.oracles import oracle_choice_key


MODULE_STRENGTH = {
    "none": 0.0,
    "attention_heads": 0.25,
    "mlp_channels": 0.25,
    "attention_block": 0.5,
    "mlp_block": 0.5,
    "layer": 1.0,
}


def step_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]), int(row["segment_id"]))


def problem_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]))


def pruning_strength(module: str, ratio: float) -> float:
    return MODULE_STRENGTH.get(module, 0.5) * float(ratio)


def choice_strength(choice: str) -> float:
    module = choice.split("|", 1)[0]
    ratio = 0.0
    for part in choice.split("|"):
        if part.startswith("r="):
            ratio = float(part[2:])
            break
    return pruning_strength(module, ratio)


def row_strength(row: dict[str, Any]) -> float:
    return pruning_strength(str(row.get("module", "none")), float(row.get("ratio", 0.0)))


def synthetic_dense_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref = rows[0]
    return {
        "dataset": ref.get("dataset"),
        "id": ref["id"],
        "segment_id": ref["segment_id"],
        "segment_type": ref.get("segment_type", "unknown"),
        "module": "none",
        "ratio": 0.0,
        "pruned_layers": [],
        "flipped": False,
        "policy_note": "dense_no_pruning",
    }


def load_rows(configs: list[str], inputs: list[str]) -> list[dict[str, Any]]:
    paths = list(inputs)
    for config in configs:
        from src.utils.io import read_yaml

        cfg = read_yaml(config)
        paths.append(cfg["paths"]["counterfactuals"])
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl_local(path))
    return rows


def ensure_dir_local(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl_local(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json_local(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir_local(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl_local(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir_local(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_steps(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[step_key(row)].append(row)
    return dict(groups)


def pick_row(rows: list[dict[str, Any]], module: str, ratio: float) -> dict[str, Any]:
    exact = [row for row in rows if row.get("module") == module and float(row.get("ratio", -1)) == float(ratio)]
    if exact:
        return exact[0]
    same_module = [row for row in rows if row.get("module") == module]
    if same_module:
        return min(same_module, key=lambda row: abs(float(row.get("ratio", 0.0)) - float(ratio)))
    return min(rows, key=lambda row: abs(float(row.get("ratio", 0.0)) - float(ratio)))


def safe_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: (int(bool(row["flipped"])), -row_strength(row), oracle_choice_key(row)))


def score_quantiles(step_scores: dict[tuple[str, str, int], float]) -> tuple[float, float]:
    values = sorted(step_scores.values())
    if not values:
        return 0.0, 0.0
    low_index = int(0.33 * (len(values) - 1))
    high_index = int(0.67 * (len(values) - 1))
    return values[low_index], values[high_index]


def score_policy_row(
    rows: list[dict[str, Any]],
    risk_score: float,
    low_threshold: float,
    high_threshold: float,
    module: str,
    allow_no_pruning: bool,
) -> dict[str, Any]:
    if risk_score >= high_threshold and allow_no_pruning:
        return synthetic_dense_row(rows)
    if risk_score >= high_threshold:
        return pick_row(rows, module, 0.2)
    if risk_score >= low_threshold:
        return pick_row(rows, module, 0.4)
    return pick_row(rows, module, 0.6)


def global_choice_scores(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[oracle_choice_key(row)].append(row)
    scores = {}
    for choice, group in grouped.items():
        flips = sum(int(bool(row["flipped"])) for row in group)
        scores[choice] = (flips, len(group), flips / len(group))
    return scores


def select_static_best_safe(rows: list[dict[str, Any]], steps: dict[tuple[str, str, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scores = global_choice_scores(rows)
    best_choice = min(scores, key=lambda choice: (scores[choice][2], -choice_strength(choice), choice))
    return [next(row for row in group if oracle_choice_key(row) == best_choice) for group in steps.values()]


def select_prompt_oracle_safe(steps: dict[tuple[str, str, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_problem_choice: dict[tuple[tuple[str, str], str], list[dict[str, Any]]] = defaultdict(list)
    choices_by_problem: dict[tuple[str, str], set[str]] = defaultdict(set)
    for group in steps.values():
        for row in group:
            pkey = problem_key(row)
            choice = oracle_choice_key(row)
            by_problem_choice[(pkey, choice)].append(row)
            choices_by_problem[pkey].add(choice)

    best_by_problem = {}
    for pkey, choices in choices_by_problem.items():
        best_by_problem[pkey] = min(
            choices,
            key=lambda choice: (
                sum(int(bool(row["flipped"])) for row in by_problem_choice[(pkey, choice)])
                / len(by_problem_choice[(pkey, choice)]),
                -choice_strength(choice),
                choice,
            ),
        )
    return [next(row for row in group if oracle_choice_key(row) == best_by_problem[problem_key(group[0])]) for group in steps.values()]


def select_score_policy(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
    score_name: str,
    module: str,
    allow_no_pruning: bool,
) -> list[dict[str, Any]]:
    if score_name == "entropy":
        scores = {key: float(group[0].get("entropy", 0.0)) for key, group in steps.items()}
    elif score_name == "confidence":
        scores = {key: 1.0 - float(group[0].get("confidence", 0.0)) for key, group in steps.items()}
    else:
        raise ValueError(f"Unsupported score policy: {score_name}")
    low, high = score_quantiles(scores)
    return [
        score_policy_row(group, scores[key], low, high, module=module, allow_no_pruning=allow_no_pruning)
        for key, group in steps.items()
    ]


def summarize_policy(name: str, selected: list[dict[str, Any]], step_oracle_flip_rate: float | None) -> dict[str, Any]:
    n = len(selected)
    flips = sum(int(bool(row["flipped"])) for row in selected)
    strengths = [row_strength(row) for row in selected]
    ratios = [float(row.get("ratio", 0.0)) for row in selected]
    modules = Counter(str(row.get("module", "none")) for row in selected)
    choices = Counter("none|r=0.00|layers=" if row.get("module") == "none" else oracle_choice_key(row) for row in selected)

    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_stage[str(row.get("segment_type", "unknown"))].append(row)
        by_dataset[str(row.get("dataset", "unknown"))].append(row)

    flip_rate = flips / n if n else None
    return {
        "policy": name,
        "n_problem_steps": n,
        "selected_action_flip_rate": flip_rate,
        "average_pruning_ratio": sum(ratios) / n if n else None,
        "average_pruning_strength_proxy": sum(strengths) / n if n else None,
        "gap_to_step_oracle": flip_rate - step_oracle_flip_rate if flip_rate is not None and step_oracle_flip_rate is not None else None,
        "module_distribution": dict(sorted(modules.items())),
        "choice_distribution": dict(sorted(choices.items())),
        "stage_flip_rates": {
            stage: sum(int(bool(row["flipped"])) for row in group) / len(group) for stage, group in sorted(by_stage.items())
        },
        "dataset_flip_rates": {
            dataset: sum(int(bool(row["flipped"])) for row in group) / len(group)
            for dataset, group in sorted(by_dataset.items())
        },
    }


def write_summary_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir_local(path.parent)
    fields = [
        "policy",
        "n_problem_steps",
        "selected_action_flip_rate",
        "average_pruning_ratio",
        "average_pruning_strength_proxy",
        "gap_to_step_oracle",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-output", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--static-module", default="mlp_block")
    parser.add_argument("--static-ratio", type=float, default=0.4)
    parser.add_argument("--score-module", default="mlp_block")
    parser.add_argument("--no-no-pruning", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.configs, args.inputs)
    if not rows:
        raise ValueError("No counterfactual rows found")
    steps = group_steps(rows)

    dense = [synthetic_dense_row(group) for group in steps.values()]
    static_default = [pick_row(group, args.static_module, args.static_ratio) for group in steps.values()]
    static_best = select_static_best_safe(rows, steps)
    entropy_policy = select_score_policy(steps, "entropy", args.score_module, allow_no_pruning=not args.no_no_pruning)
    confidence_policy = select_score_policy(steps, "confidence", args.score_module, allow_no_pruning=not args.no_no_pruning)
    prompt_oracle = select_prompt_oracle_safe(steps)
    step_oracle = [safe_best(group) for group in steps.values()]
    step_oracle_flip = sum(int(bool(row["flipped"])) for row in step_oracle) / len(step_oracle)

    policies = {
        "dense_lrm": dense,
        f"static_{args.static_module}_r{args.static_ratio:.2f}": static_default,
        "static_best_safe_oracle": static_best,
        "entropy_quantile_policy": entropy_policy,
        "confidence_quantile_policy": confidence_policy,
        "prompt_router_safe_oracle": prompt_oracle,
        "step_safe_oracle": step_oracle,
    }
    summaries = [summarize_policy(name, selected, step_oracle_flip) for name, selected in policies.items()]

    result = {
        "n_counterfactual_rows": len(rows),
        "n_problem_steps": len(steps),
        "selection_unit": "one action per dataset/problem/segment",
        "objective": "safety-oriented offline policy simulation; lower selected_action_flip_rate is better",
        "pruning_strength_proxy": MODULE_STRENGTH,
        "policies": summaries,
    }
    write_json_local(args.output, result)
    if args.summary_csv:
        write_summary_csv(args.summary_csv, summaries)
    if args.selected_output:
        selected_rows = []
        for policy, selected in policies.items():
            for row in selected:
                out = dict(row)
                out["policy"] = policy
                out["choice"] = "none|r=0.00|layers=" if out.get("module") == "none" else oracle_choice_key(out)
                selected_rows.append(out)
        write_jsonl_local(args.selected_output, selected_rows)


if __name__ == "__main__":
    main()
