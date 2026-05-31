from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.main_offline_baselines import MODULE_STRENGTH, group_steps, oracle_choice_key, row_strength, synthetic_dense_row
from src.main_rasp_zero_offline import (
    DEFAULT_STAGE_CAPS,
    ensure_dir,
    load_action_rows,
    load_probe_step_scores,
    read_jsonl,
    select_budget_policy,
)


DEFAULT_STRENGTH_BUDGETS = [0.1, 0.2, 0.3]


def action_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("dataset") or "unknown"),
        str(row["id"]),
        int(row["segment_id"]),
        oracle_choice_key(row),
    )


def load_action_scores(path: str | Path) -> dict[tuple[str, str, int, str], float]:
    scores = {}
    for row in read_jsonl(path):
        scores[action_key(row)] = float(row["risk_score"])
    return scores


def select_multimodule_budget_policy(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
    action_scores: dict[tuple[str, str, int, str], float],
    target_average_strength: float,
    score_field: str = "policy_risk_score",
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    candidates = []
    for step, rows in steps.items():
        for row in rows:
            key = action_key(row)
            if key not in action_scores:
                raise ValueError(f"Action score missing for {key}")
            strength = row_strength(row)
            if strength > 0:
                candidates.append((float(action_scores[key]), -strength, oracle_choice_key(row), step, row))
    candidates.sort()
    strength_total = 0.0
    target_total = float(target_average_strength) * len(steps)
    for risk, negative_strength, _choice, step, row in candidates:
        strength = -negative_strength
        previous_strength = row_strength(selected[step]) if step in selected else 0.0
        increment = strength - previous_strength
        if increment <= 1e-9:
            continue
        if strength_total + increment > target_total + 1e-9:
            continue
        chosen = dict(row)
        chosen[score_field] = risk
        chosen["policy_target_average_strength"] = float(target_average_strength)
        selected[step] = chosen
        strength_total += increment
    return [selected.get(step, synthetic_dense_row(rows)) for step, rows in steps.items()]


def select_oracle_budget_policy(
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
    target_average_strength: float,
) -> list[dict[str, Any]]:
    oracle_scores = {}
    for rows in steps.values():
        for row in rows:
            oracle_scores[action_key(row)] = float(int(bool(row["flipped"])))
    return select_multimodule_budget_policy(steps, oracle_scores, target_average_strength, score_field="oracle_flip_label")


def select_static_best_at_strength(
    rows: list[dict[str, Any]],
    steps: dict[tuple[str, str, int], list[dict[str, Any]]],
    target_average_strength: float,
) -> list[dict[str, Any]]:
    by_choice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_choice[oracle_choice_key(row)].append(row)
    choices = []
    for choice, choice_rows in by_choice.items():
        strength = row_strength(choice_rows[0])
        flip_rate = sum(int(bool(row["flipped"])) for row in choice_rows) / len(choice_rows)
        choices.append((abs(strength - target_average_strength), flip_rate, -strength, choice))
    _distance, _risk, _negative_strength, best_choice = min(choices)
    return [dict(next(row for row in step_rows if oracle_choice_key(row) == best_choice)) for step_rows in steps.values()]


def summarize(name: str, selected: list[dict[str, Any]], target_strength: float | None = None) -> dict[str, Any]:
    n = len(selected)
    flips = sum(int(bool(row["flipped"])) for row in selected)
    ratios = [float(row.get("ratio", 0.0)) for row in selected]
    strengths = [row_strength(row) for row in selected]
    predicted_scores = [float(row["policy_risk_score"]) for row in selected if "policy_risk_score" in row]
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_stage[str(row.get("segment_type", "unknown"))].append(row)
        by_dataset[str(row.get("dataset", "unknown"))].append(row)
    return {
        "policy": name,
        "target_average_pruning_strength_proxy": target_strength,
        "n_problem_steps": n,
        "selected_action_flip_rate": flips / n if n else None,
        "average_pruning_ratio": sum(ratios) / n if n else None,
        "average_pruning_strength_proxy": sum(strengths) / n if n else None,
        "budget_utilization": sum(strengths) / (n * target_strength) if n and target_strength else None,
        "average_selected_predicted_risk": sum(predicted_scores) / len(predicted_scores) if predicted_scores else None,
        "module_distribution": dict(sorted(Counter(str(row.get("module", "none")) for row in selected).items())),
        "choice_distribution": dict(
            sorted(
                Counter(
                    "none|r=0.00|layers=" if row.get("module") == "none" else oracle_choice_key(row)
                    for row in selected
                ).items()
            )
        ),
        "stage_flip_rates": {
            stage: sum(int(bool(row["flipped"])) for row in rows) / len(rows) for stage, rows in sorted(by_stage.items())
        },
        "dataset_flip_rates": {
            dataset: sum(int(bool(row["flipped"])) for row in rows) / len(rows) for dataset, rows in sorted(by_dataset.items())
        },
    }


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


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fields = [
        "policy",
        "target_average_pruning_strength_proxy",
        "n_problem_steps",
        "selected_action_flip_rate",
        "average_pruning_ratio",
        "average_pruning_strength_proxy",
        "budget_utilization",
        "average_selected_predicted_risk",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--hidden-step-scores", required=True)
    parser.add_argument("--action-scores", required=True)
    parser.add_argument("--action-stage-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strength-budgets", nargs="+", type=float, default=DEFAULT_STRENGTH_BUDGETS)
    args = parser.parse_args()

    rows = load_action_rows([], args.inputs)
    steps = group_steps(rows)
    mlp_block_ratios = sorted({float(row.get("ratio", 0.0)) for row in rows if row.get("module") == "mlp_block"})
    if not mlp_block_ratios:
        raise ValueError("No mlp_block actions found for v1 ablations")
    hidden_scores = load_probe_step_scores(args.hidden_step_scores)
    action_scores = load_action_scores(args.action_scores)
    action_stage_scores = load_action_scores(args.action_stage_scores)
    output_dir = ensure_dir(args.output_dir)
    summaries = []
    selected_rows = []

    def add(policy: str, selected: list[dict[str, Any]], target: float | None = None) -> None:
        summaries.append(summarize(policy, selected, target_strength=target))
        for row in selected:
            out = dict(row)
            out["policy"] = policy
            selected_rows.append(out)

    add("dense_lrm", [synthetic_dense_row(group) for group in steps.values()], 0.0)
    for strength_budget in args.strength_budgets:
        add(
            f"static_best_strength_{strength_budget:.2f}",
            select_static_best_at_strength(rows, steps, strength_budget),
            strength_budget,
        )
        # Preserve v1 as ablations. MLP-block ratio is 2x the strength proxy.
        ratio_budget = min(0.6, 2 * strength_budget)
        add(
            f"hidden_step_mlp_block_strength_{strength_budget:.2f}",
            select_budget_policy(steps, hidden_scores, "mlp_block", mlp_block_ratios, ratio_budget),
            strength_budget,
        )
        add(
            f"hard_stage_cap_ablation_strength_{strength_budget:.2f}",
            select_budget_policy(
                steps,
                hidden_scores,
                "mlp_block",
                mlp_block_ratios,
                ratio_budget,
                stage_caps=DEFAULT_STAGE_CAPS,
            ),
            strength_budget,
        )
        add(
            f"action_conditioned_strength_{strength_budget:.2f}",
            select_multimodule_budget_policy(steps, action_scores, strength_budget),
            strength_budget,
        )
        add(
            f"rasp_zero_v2_soft_stage_strength_{strength_budget:.2f}",
            select_multimodule_budget_policy(steps, action_stage_scores, strength_budget),
            strength_budget,
        )
        add(
            f"safe_step_oracle_strength_{strength_budget:.2f}",
            select_oracle_budget_policy(steps, strength_budget),
            strength_budget,
        )

    result = {
        "method": "RASP-Zero Offline Evaluation v2",
        "scope": "Action-conditioned, multi-module counterfactual policy simulation. Strength values are proxies, not runtime speedups.",
        "n_counterfactual_rows": len(rows),
        "n_problem_steps": len(steps),
        "action_bank": {
            "modules": sorted(MODULE_STRENGTH),
            "module_strength_weights": MODULE_STRENGTH,
            "ratios": sorted({float(row.get("ratio", 0.0)) for row in rows}),
        },
        "policies": summaries,
    }
    write_json(output_dir / "rasp_zero_offline_v2_summary.json", result)
    write_csv(output_dir / "rasp_zero_offline_v2_summary.csv", summaries)
    write_jsonl(output_dir / "rasp_zero_offline_v2_selected_actions.jsonl", selected_rows)


if __name__ == "__main__":
    main()
