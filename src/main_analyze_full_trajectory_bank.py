from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.rasp.action_risk_single_window import context_features
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


VARIANTS = (
    "action_only",
    "causal_context_action",
    "soft_stage_context_action",
    "hidden_context_action",
    "hidden_soft_stage_context_action",
)


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["id"])


def folds(rows: list[dict[str, Any]], count: int, seed: int) -> list[set[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for value in sorted({key(row) for row in rows}):
        grouped[value[0]].append(value)
    output = [set() for _ in range(count)]
    rng = random.Random(seed)
    offset = 0
    for values in grouped.values():
        rng.shuffle(values)
        for index, value in enumerate(values):
            output[(offset + index) % count].add(value)
        offset += len(values)
    return output


def score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    if len(set(labels.tolist())) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "pr_auc": float(average_precision_score(labels, predictions)),
    }


def risk_bins(
    labels: np.ndarray, predictions: np.ndarray, count: int = 5
) -> list[dict[str, Any]]:
    output = []
    for index, indices in enumerate(np.array_split(np.argsort(predictions), count)):
        if len(indices):
            output.append(
                {
                    "bin": index,
                    "rows": int(len(indices)),
                    "mean_risk": float(predictions[indices].mean()),
                    "flip_rate": float(labels[indices].mean()),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--data-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    hidden = torch.load(args.hidden_states, map_location="cpu").float().flatten(start_dim=1).numpy()
    data_summary = read_json(args.data_summary)
    if len(rows) != len(hidden):
        raise ValueError("Full-trajectory row/hidden mismatch")
    contexts, stages, hidden_rows, labels, metadata = [], [], [], [], []
    for index, row in enumerate(rows):
        probabilities = row.get("stage_probabilities") or {}
        stage = [
            float(probabilities.get("setup") or 0.0),
            float(probabilities.get("reasoning") or 0.0),
            float(probabilities.get("final") or 0.0),
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
            stages.append(stage)
            hidden_rows.append(hidden[index])
            labels.append(int(bool(flipped)))
            metadata.append(
                {
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "generated_tokens_at_boundary": int(row["generated_tokens_at_boundary"]),
                    "trusted_stage": row["trusted_stage"],
                    "ratio": float(ratio),
                    "flipped": bool(flipped),
                }
            )
    contexts = np.asarray(contexts)
    stages = np.asarray(stages)
    hidden_rows = np.asarray(hidden_rows)
    labels = np.asarray(labels)
    predictions = {name: np.full(len(labels), np.nan) for name in VARIANTS}
    fold_results = {name: [] for name in VARIANTS}
    split = folds(rows, args.folds, args.seed)
    for fold_index, test_problems in enumerate(split):
        test = np.asarray(
            [i for i, row in enumerate(metadata) if key(row) in test_problems],
            dtype=int,
        )
        test_set = set(test.tolist())
        train = np.asarray(
            [i for i in range(len(labels)) if i not in test_set],
            dtype=int,
        )
        if len(set(labels[train].tolist())) < 2:
            raise ValueError(f"Fold {fold_index} contains no positive training examples")
        context_scaler = StandardScaler().fit(contexts[train])
        stage_scaler = StandardScaler().fit(stages[train])
        context_train = context_scaler.transform(contexts[train])
        context_test = context_scaler.transform(contexts[test])
        stage_train = stage_scaler.transform(stages[train])
        stage_test = stage_scaler.transform(stages[test])
        hidden_scaler = StandardScaler().fit(hidden_rows[train])
        hidden_train = hidden_scaler.transform(hidden_rows[train])
        hidden_test = hidden_scaler.transform(hidden_rows[test])
        rank = min(args.pca_dim, hidden_train.shape[0] - 1, hidden_train.shape[1])
        pca = PCA(n_components=rank, random_state=args.seed + fold_index).fit(hidden_train)
        inputs = {
            "action_only": (context_train[:, -2:], context_test[:, -2:]),
            "causal_context_action": (context_train, context_test),
            "soft_stage_context_action": (
                np.concatenate([stage_train, context_train], axis=1),
                np.concatenate([stage_test, context_test], axis=1),
            ),
            "hidden_context_action": (
                np.concatenate([pca.transform(hidden_train), context_train], axis=1),
                np.concatenate([pca.transform(hidden_test), context_test], axis=1),
            ),
            "hidden_soft_stage_context_action": (
                np.concatenate([pca.transform(hidden_train), stage_train, context_train], axis=1),
                np.concatenate([pca.transform(hidden_test), stage_test, context_test], axis=1),
            ),
        }
        for variant, (train_x, test_x) in inputs.items():
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=args.seed + fold_index,
            ).fit(train_x, labels[train])
            values = model.predict_proba(test_x)[:, 1]
            predictions[variant][test] = values
            fold_results[variant].append({"fold": fold_index, **score(labels[test], values)})
    if any(np.isnan(values).any() for values in predictions.values()):
        raise RuntimeError("Some full-trajectory rows lack OOF predictions")

    variants = {}
    for variant, values in predictions.items():
        bins = risk_bins(labels, values)
        variants[variant] = {
            **score(labels, values),
            "folds": fold_results[variant],
            "risk_bins": bins,
            "risk_bin_inversions": sum(
                int(right["flip_rate"] + 1e-12 < left["flip_rate"])
                for left, right in zip(bins, bins[1:])
            ),
            "by_dataset": {
                dataset: score(
                    labels[np.asarray([i for i, row in enumerate(metadata) if row["dataset"] == dataset])],
                    values[np.asarray([i for i, row in enumerate(metadata) if row["dataset"] == dataset])],
                )
                for dataset in sorted({str(row["dataset"]) for row in metadata})
            },
        }
    context = variants["causal_context_action"]
    action = variants["action_only"]
    context_fold_wins = sum(
        int(
            (left["roc_auc"] or -1.0) > (right["roc_auc"] or -1.0)
            and (left["pr_auc"] or -1.0) > (right["pr_auc"] or -1.0)
        )
        for left, right in zip(context["folds"], action["folds"])
    )
    soft = variants["soft_stage_context_action"]
    soft_fold_wins = sum(
        int(
            (left["roc_auc"] or -1.0) > (right["roc_auc"] or -1.0)
            and (left["pr_auc"] or -1.0) > (right["pr_auc"] or -1.0)
        )
        for left, right in zip(soft["folds"], context["folds"])
    )
    hidden = variants["hidden_context_action"]
    hidden_fold_wins = sum(
        int(
            (left["roc_auc"] or -1.0) > (right["roc_auc"] or -1.0)
            and (left["pr_auc"] or -1.0) > (right["pr_auc"] or -1.0)
        )
        for left, right in zip(hidden["folds"], context["folds"])
    )
    hidden_soft = variants["hidden_soft_stage_context_action"]
    hidden_soft_fold_wins = sum(
        int(
            (left["roc_auc"] or -1.0) > (right["roc_auc"] or -1.0)
            and (left["pr_auc"] or -1.0) > (right["pr_auc"] or -1.0)
        )
        for left, right in zip(hidden_soft["folds"], hidden["folds"])
    )
    coverage = Counter()
    diagnostic_coverage = Counter()
    for row in rows:
        fraction = int(row["generated_tokens_at_boundary"]) / max(1, int(row["max_new_tokens"]))
        coverage[min(9, int(fraction * 10))] += 1
        diagnostic_fraction = float(
            row.get("diagnostic_only", {}).get("dense_trajectory_relative_position", 0.0)
        )
        diagnostic_coverage[min(9, int(diagnostic_fraction * 10))] += 1
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "02_full_trajectory_oof_predictions.jsonl",
        [
            {
                **row,
                **{f"{variant}_risk": float(predictions[variant][index]) for variant in VARIANTS},
            }
            for index, row in enumerate(metadata)
        ],
    )
    write_json(
        output / "02_full_trajectory_analysis.json",
        {
            "schema": "rasp_full_trajectory_dense_state_oof_v1",
            "problem_count": data_summary["problem_count"],
            "boundary_count": len(rows),
            "action_rows": len(labels),
            "positive_flips": int(labels.sum()),
            "causal_position_decile_counts": {
                str(index): coverage[index] for index in range(10)
            },
            "diagnostic_dense_trajectory_position_decile_counts": {
                str(index): diagnostic_coverage[index] for index in range(10)
            },
            "trusted_stage_counts": data_summary["trusted_stage_counts"],
            "dose_response": data_summary["dose_response"],
            "dose_response_inversions": sum(
                int(right["flip_rate"] + 1e-12 < left["flip_rate"])
                for left, right in zip(
                    data_summary["dose_response"],
                    data_summary["dose_response"][1:],
                )
            ),
            "variants": variants,
            "context_beats_action_both_metrics_fold_count": context_fold_wins,
            "soft_stage_beats_context_both_metrics_fold_count": soft_fold_wins,
            "hidden_beats_context_both_metrics_fold_count": hidden_fold_wins,
            "soft_stage_adds_to_hidden_both_metrics_fold_count": hidden_soft_fold_wins,
            "dense_risk_model_promising": context_fold_wins >= max(2, args.folds - 1),
            "soft_stage_is_optional_diagnostic": True,
            "causal_feature_audit_passed": bool(
                data_summary["causal_features_exclude_dense_trajectory_length"]
                and data_summary["tail_anchor_is_diagnostic_only"]
                and not data_summary["final_test_sources_used"]
            ),
        },
    )


if __name__ == "__main__":
    main()
