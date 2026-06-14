from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.rasp.action_risk_single_window import context_features
from src.utils.io import read_jsonl, write_json, write_jsonl


VARIANTS = (
    "action_only",
    "causal_context_action",
    "stage_context_action",
    "hidden_stage_context_action",
)
STAGES = ("setup", "reasoning", "verification", "final")


def problem_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["id"])


def problem_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> list[set[tuple[str, str]]]:
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted({problem_key(row) for row in rows}):
        by_dataset[key[0]].append(key)
    rng = random.Random(seed)
    output = [set() for _ in range(folds)]
    offset = 0
    for keys in by_dataset.values():
        rng.shuffle(keys)
        for index, key in enumerate(keys):
            output[(offset + index) % folds].add(key)
        offset += len(keys)
    return output


def flatten(
    rows: list[dict[str, Any]], hidden: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    contexts, stages, hidden_rows, labels, metadata = [], [], [], [], []
    for index, row in enumerate(rows):
        probabilities = row["stage_probabilities"]
        stage_features = [
            float(probabilities["setup"]),
            float(probabilities["reasoning"]),
            float(probabilities["final"]),
            float(row["stage_confidence"]),
            float(bool(row["reasoning_accepted"])),
            *[float(row["operational_stage"] == stage) for stage in STAGES],
        ]
        for ratio, flipped in zip(row["candidate_ratios"][1:], row["candidate_flipped"][1:]):
            contexts.append(
                context_features(
                    entropy=row["entropy"],
                    confidence=row["confidence"],
                    generated_tokens=row["generated_tokens_at_boundary"],
                    max_new_tokens=row["max_new_tokens"],
                    ratio=ratio,
                )
            )
            stages.append(stage_features)
            hidden_rows.append(hidden[index].float().flatten().numpy())
            labels.append(int(bool(flipped)))
            metadata.append(
                {
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "boundary": int(row["generated_tokens_at_boundary"]),
                    "operational_stage": row["operational_stage"],
                    "ratio": float(ratio),
                    "flipped": bool(flipped),
                }
            )
    return (
        np.asarray(contexts),
        np.asarray(stages),
        np.asarray(hidden_rows),
        np.asarray(labels),
        metadata,
    )


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def simulate_first_accepted_policy(
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(metadata):
        grouped[problem_key(row)][int(row["boundary"])].append(index)
    per_dataset: dict[str, dict[str, float]] = defaultdict(
        lambda: {"problems": 0, "selected": 0, "flips": 0, "ratio_sum": 0.0}
    )
    decisions = {}
    for problem, boundaries in grouped.items():
        stats = per_dataset[problem[0]]
        stats["problems"] += 1
        selected = None
        for boundary in (32, 96, 160):
            envelope = 0.0
            accepted = []
            for index in sorted(boundaries[boundary], key=lambda value: float(metadata[value]["ratio"])):
                envelope = max(envelope, float(scores[index]))
                if envelope <= threshold:
                    accepted.append(index)
            if accepted:
                selected = accepted[-1]
                break
        if selected is not None:
            stats["selected"] += 1
            stats["flips"] += int(labels[selected])
            stats["ratio_sum"] += float(metadata[selected]["ratio"])
            decisions[problem] = [int(metadata[selected]["boundary"]), float(metadata[selected]["ratio"])]
        else:
            decisions[problem] = None
    output = {}
    for dataset, stats in sorted(per_dataset.items()):
        problems = int(stats["problems"])
        output[dataset] = {
            "problems": problems,
            "selected": int(stats["selected"]),
            "action_coverage": int(stats["selected"]) / problems,
            "average_action_ratio": float(stats["ratio_sum"]) / problems,
            "problem_flip_rate": int(stats["flips"]) / problems,
        }
    output["combined"] = {
        "problems": sum(value["problems"] for value in output.values()),
        "average_action_ratio": sum(
            value["average_action_ratio"] * value["problems"] for value in output.values()
        )
        / sum(value["problems"] for value in output.values()),
    }
    output["decisions"] = decisions
    return output


def calibrate_policy(
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    target_average_ratio: float = 0.20,
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(float(value) for value in scores)))
    indices = np.linspace(0, len(unique) - 1, min(401, len(unique))).round().astype(int)
    candidates = []
    for threshold in unique[indices]:
        result = simulate_first_accepted_policy(metadata, labels, scores, float(threshold))
        datasets = [value for name, value in result.items() if name not in {"combined", "decisions"}]
        if all(float(value["problem_flip_rate"]) <= 0.05 for value in datasets):
            candidates.append(
                (
                    abs(float(result["combined"]["average_action_ratio"]) - target_average_ratio),
                    -float(result["combined"]["average_action_ratio"]),
                    float(threshold),
                    result,
                )
            )
    if not candidates:
        return {"eligible": False}
    _distance, _negative_ratio, threshold, result = min(candidates)
    return {"eligible": True, "threshold": threshold, **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    hidden = torch.load(args.hidden_states, map_location="cpu")
    if len(rows) != len(hidden):
        raise ValueError("Stage-Action-Risk rows and hidden states differ")
    contexts, stages, hidden_rows, labels, metadata = flatten(rows, hidden)
    folds = problem_folds(rows, args.folds, args.seed)
    predictions = {variant: np.full(len(labels), np.nan) for variant in VARIANTS}
    fold_results = {variant: [] for variant in VARIANTS}
    for fold_index, test_problems in enumerate(folds):
        test = np.asarray(
            [index for index, row in enumerate(metadata) if problem_key(row) in test_problems],
            dtype=int,
        )
        test_set = set(test.tolist())
        train = np.asarray([index for index in range(len(labels)) if index not in test_set], dtype=int)
        context_scaler = StandardScaler().fit(contexts[train])
        stage_scaler = StandardScaler().fit(stages[train])
        context_train, context_test = context_scaler.transform(contexts[train]), context_scaler.transform(contexts[test])
        stage_train, stage_test = stage_scaler.transform(stages[train]), stage_scaler.transform(stages[test])
        hidden_scaler = StandardScaler().fit(hidden_rows[train])
        hidden_train = hidden_scaler.transform(hidden_rows[train])
        hidden_test = hidden_scaler.transform(hidden_rows[test])
        rank = min(args.pca_dim, hidden_train.shape[0] - 1, hidden_train.shape[1])
        pca = PCA(n_components=rank, random_state=args.seed + fold_index).fit(hidden_train)
        inputs = {
            "action_only": (context_train[:, -2:], context_test[:, -2:]),
            "causal_context_action": (context_train, context_test),
            "stage_context_action": (
                np.concatenate([stage_train, context_train], axis=1),
                np.concatenate([stage_test, context_test], axis=1),
            ),
            "hidden_stage_context_action": (
                np.concatenate([pca.transform(hidden_train), stage_train, context_train], axis=1),
                np.concatenate([pca.transform(hidden_test), stage_test, context_test], axis=1),
            ),
        }
        for variant, (train_x, test_x) in inputs.items():
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=args.seed + fold_index,
            )
            model.fit(train_x, labels[train])
            scores = model.predict_proba(test_x)[:, 1]
            predictions[variant][test] = scores
            fold_results[variant].append({"fold": fold_index, **metrics(labels[test], scores)})
    if any(np.isnan(values).any() for values in predictions.values()):
        raise RuntimeError("Some v2 action rows did not receive OOF predictions")

    variants = {}
    for variant, scores in predictions.items():
        variants[variant] = {
            **metrics(labels, scores),
            "folds": fold_results[variant],
            "by_dataset": {
                dataset: metrics(
                    labels[np.asarray([i for i, row in enumerate(metadata) if row["dataset"] == dataset])],
                    scores[np.asarray([i for i, row in enumerate(metadata) if row["dataset"] == dataset])],
                )
                for dataset in sorted({str(row["dataset"]) for row in metadata})
            },
        }
    context = variants["causal_context_action"]
    stage = variants["stage_context_action"]
    fold_wins = sum(
        int(
            (stage_fold["roc_auc"] or -1.0) > (context_fold["roc_auc"] or -1.0)
            and (stage_fold["pr_auc"] or -1.0) > (context_fold["pr_auc"] or -1.0)
        )
        for stage_fold, context_fold in zip(stage["folds"], context["folds"])
    )
    dataset_wins = all(
        (stage["by_dataset"][dataset]["roc_auc"] or -1.0)
        > (context["by_dataset"][dataset]["roc_auc"] or -1.0)
        and (stage["by_dataset"][dataset]["pr_auc"] or -1.0)
        > (context["by_dataset"][dataset]["pr_auc"] or -1.0)
        for dataset in stage["by_dataset"]
    )
    stage_gate_passed = fold_wins >= 4 and dataset_wins
    policy_simulations = {
        variant: calibrate_policy(metadata, labels, scores)
        for variant, scores in predictions.items()
    }
    context_policy = policy_simulations["causal_context_action"]
    stage_policy = policy_simulations["stage_context_action"]
    policy_advantage = bool(
        context_policy.get("eligible")
        and stage_policy.get("eligible")
        and float(stage_policy["combined"]["average_action_ratio"])
        >= 0.8 * float(context_policy["combined"]["average_action_ratio"])
        and all(
            float(stage_policy[dataset]["problem_flip_rate"])
            <= float(context_policy[dataset]["problem_flip_rate"]) + 1e-12
            for dataset in ("gsm8k", "math_train")
        )
        and any(
            float(stage_policy[dataset]["problem_flip_rate"])
            < float(context_policy[dataset]["problem_flip_rate"]) - 1e-12
            for dataset in ("gsm8k", "math_train")
        )
    )
    stage_gate_passed = stage_gate_passed and policy_advantage
    serializable_policies = {}
    for variant, result in policy_simulations.items():
        serializable_policies[variant] = {
            key: value for key, value in result.items() if key != "decisions"
        }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "02_stage_action_risk_oof_predictions.jsonl",
        [
            {
                **row,
                **{f"{variant}_risk": float(predictions[variant][index]) for variant in VARIANTS},
            }
            for index, row in enumerate(metadata)
        ],
    )
    write_json(
        output / "02_stage_action_risk_analysis.json",
        {
            "schema": "rasp_stage_action_risk_exact_boundary_oof_v2",
            "problem_count": len({problem_key(row) for row in rows}),
            "boundary_count": len(rows),
            "action_rows": len(labels),
            "positive_flips": int(labels.sum()),
            "variants": variants,
            "stage_beats_context_on_both_metrics_fold_count": fold_wins,
            "stage_beats_context_on_both_metrics_each_dataset": dataset_wins,
            "oof_first_accepted_policy": serializable_policies,
            "stage_policy_advantage_over_context": policy_advantage,
            "stage_controller_training_allowed": stage_gate_passed,
            "gate": {
                "stage_wins_at_least_four_of_five_folds": fold_wins >= 4,
                "stage_wins_each_dataset": dataset_wins,
                "stage_policy_retains_exposure_and_reduces_risk": policy_advantage,
            },
            "next_step": (
                "Train a stage-gated waiting controller."
                if stage_gate_passed
                else "Do not train a stage-gated controller; revise causal stage representation."
            ),
        },
    )


if __name__ == "__main__":
    main()
