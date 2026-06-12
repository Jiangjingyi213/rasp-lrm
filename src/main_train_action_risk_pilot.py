from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.utils.io import read_jsonl, write_json, write_jsonl


VARIANTS = (
    "action_only",
    "causal_context_action",
    "hidden_context_action",
    "hidden_pca_nonlinear",
)


def _problem_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["id"])


def problem_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> list[set[tuple[str, str]]]:
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted({_problem_key(row) for row in rows}):
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


def flatten_actions(
    rows: list[dict[str, Any]], hidden: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    contexts, hidden_rows, labels, metadata = [], [], [], []
    for index, row in enumerate(rows):
        causal_position = int(row["generated_tokens_at_boundary"]) / max(1, int(row["max_new_tokens"]))
        base = [
            float(row["entropy"]),
            float(row["confidence"]),
            causal_position,
            float(np.log1p(int(row["generated_tokens_at_boundary"]))),
        ]
        for ratio, flipped in zip(row["candidate_ratios"][1:], row["candidate_flipped"][1:]):
            contexts.append([*base, float(ratio), float(ratio) ** 2])
            hidden_rows.append(hidden[index].float().flatten().numpy())
            labels.append(int(bool(flipped)))
            metadata.append(
                {
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "boundary_index": int(row["boundary_index"]),
                    "generated_tokens_at_boundary": int(row["generated_tokens_at_boundary"]),
                    "ratio": float(ratio),
                    "flipped": bool(flipped),
                }
            )
    return np.asarray(contexts), np.asarray(hidden_rows), np.asarray(labels), metadata


def safe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def fit_predict(
    variant: str,
    contexts: np.ndarray,
    hidden: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    context_scaler = StandardScaler().fit(contexts[train])
    context_train = context_scaler.transform(contexts[train])
    context_test = context_scaler.transform(contexts[test])
    if variant == "action_only":
        train_x, test_x = context_train[:, -2:], context_test[:, -2:]
    elif variant == "causal_context_action":
        train_x, test_x = context_train, context_test
    else:
        hidden_scaler = StandardScaler().fit(hidden[train])
        train_hidden = hidden_scaler.transform(hidden[train])
        test_hidden = hidden_scaler.transform(hidden[test])
        rank = min(pca_dim, train_hidden.shape[0] - 1, train_hidden.shape[1])
        pca = PCA(n_components=rank, random_state=seed).fit(train_hidden)
        train_x = np.concatenate([pca.transform(train_hidden), context_train], axis=1)
        test_x = np.concatenate([pca.transform(test_hidden), context_test], axis=1)
    if variant == "hidden_pca_nonlinear":
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    if variant == "hidden_pca_nonlinear":
        train_labels = labels[train]
        positives = max(1, int(train_labels.sum()))
        negative_weight = len(train_labels) / max(1, 2 * (len(train_labels) - positives))
        positive_weight = len(train_labels) / (2 * positives)
        sample_weight = np.where(train_labels > 0, positive_weight, negative_weight)
        model.fit(train_x, train_labels, sample_weight=sample_weight)
    else:
        model.fit(train_x, labels[train])
    return model.predict_proba(test_x)[:, 1]


def risk_bins(labels: np.ndarray, scores: np.ndarray, bins: int = 5) -> list[dict[str, Any]]:
    order = np.argsort(scores)
    output = []
    for index, indices in enumerate(np.array_split(order, bins)):
        if len(indices):
            output.append(
                {
                    "bin": index,
                    "count": int(len(indices)),
                    "mean_risk": float(scores[indices].mean()),
                    "flip_rate": float(labels[indices].mean()),
                }
            )
    return output


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
        raise ValueError("Action-risk rows and hidden states differ")
    contexts, hidden_rows, labels, metadata = flatten_actions(rows, hidden)
    folds = problem_folds(rows, args.folds, args.seed)
    predictions = {variant: np.full(len(labels), np.nan) for variant in VARIANTS}
    fold_summaries: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}

    for fold_index, test_problems in enumerate(folds):
        test = np.asarray(
            [index for index, row in enumerate(metadata) if _problem_key(row) in test_problems],
            dtype=int,
        )
        test_set = set(test.tolist())
        train = np.asarray([index for index in range(len(metadata)) if index not in test_set], dtype=int)
        if len(np.unique(labels[train])) < 2:
            raise ValueError(f"Fold {fold_index} train labels contain only one class")
        for variant in VARIANTS:
            scores = fit_predict(
                variant,
                contexts,
                hidden_rows,
                labels,
                train,
                test,
                pca_dim=args.pca_dim,
                seed=args.seed + fold_index,
            )
            predictions[variant][test] = scores
            fold_summaries[variant].append(
                {
                    "fold": fold_index,
                    "problems": len(test_problems),
                    "positives": int(labels[test].sum()),
                    **safe_metrics(labels[test], scores),
                }
            )

    if any(np.isnan(scores).any() for scores in predictions.values()):
        raise RuntimeError("Some action rows did not receive OOF predictions")
    ratios = sorted({float(row["ratio"]) for row in metadata})
    dose_response = []
    for ratio in ratios:
        indices = np.asarray([i for i, row in enumerate(metadata) if float(row["ratio"]) == ratio])
        dose_response.append(
            {
                "ratio": ratio,
                "rows": int(len(indices)),
                "flips": int(labels[indices].sum()),
                "flip_rate": float(labels[indices].mean()),
            }
        )
    positive_ratios = sum(int(row["flips"] >= 5) for row in dose_response)
    dose_inversions = sum(
        int(right["flip_rate"] + 1e-12 < left["flip_rate"])
        for left, right in zip(dose_response, dose_response[1:])
    )

    variants = {}
    for variant, scores in predictions.items():
        bins = risk_bins(labels, scores)
        variants[variant] = {
            **safe_metrics(labels, scores),
            "folds": fold_summaries[variant],
            "risk_bins": bins,
            "risk_bin_inversions": sum(
                int(right["flip_rate"] + 1e-12 < left["flip_rate"])
                for left, right in zip(bins, bins[1:])
            ),
            "by_dataset": {
                dataset: safe_metrics(
                    labels[np.asarray([i for i, row in enumerate(metadata) if row["dataset"] == dataset])],
                    scores[np.asarray([i for i, row in enumerate(metadata) if row["dataset"] == dataset])],
                )
                for dataset in sorted({str(row["dataset"]) for row in metadata})
            },
        }
    hidden_best = max(
        ("hidden_context_action", "hidden_pca_nonlinear"),
        key=lambda name: float(variants[name]["roc_auc"] or -1.0),
    )
    stable_fold_wins = sum(
        int(
            (hidden_fold["roc_auc"] or -1.0)
            > max((context_fold["roc_auc"] or -1.0), (action_fold["roc_auc"] or -1.0))
            and (hidden_fold["pr_auc"] or -1.0)
            > max((context_fold["pr_auc"] or -1.0), (action_fold["pr_auc"] or -1.0))
        )
        for hidden_fold, context_fold, action_fold in zip(
            variants[hidden_best]["folds"],
            variants["causal_context_action"]["folds"],
            variants["action_only"]["folds"],
        )
    )
    dataset_wins = all(
        (variants[hidden_best]["by_dataset"][dataset]["roc_auc"] or -1.0)
        > max(
            variants["causal_context_action"]["by_dataset"][dataset]["roc_auc"] or -1.0,
            variants["action_only"]["by_dataset"][dataset]["roc_auc"] or -1.0,
        )
        for dataset in variants[hidden_best]["by_dataset"]
    )
    passed = (
        positive_ratios >= 2
        and dose_inversions <= 1
        and stable_fold_wins >= 4
        and dataset_wins
        and variants[hidden_best]["risk_bin_inversions"] <= 1
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof_rows = []
    for index, row in enumerate(metadata):
        oof_rows.append({**row, **{f"{variant}_risk": float(predictions[variant][index]) for variant in VARIANTS}})
    write_jsonl(output / "02_action_risk_oof_predictions.jsonl", oof_rows)
    write_json(
        output / "02_action_risk_pilot_summary.json",
        {
            "schema": "rasp_action_risk_pilot_oof_v1",
            "folds": args.folds,
            "problem_count": len({_problem_key(row) for row in rows}),
            "boundary_count": len(rows),
            "action_rows": len(metadata),
            "positive_flips": int(labels.sum()),
            "dose_response": dose_response,
            "dose_response_inversions": dose_inversions,
            "ratios_with_at_least_five_flips": positive_ratios,
            "variants": variants,
            "selected_hidden_variant": hidden_best,
            "hidden_stable_fold_wins_over_simple_baselines": stable_fold_wins,
            "hidden_beats_simple_baselines_on_each_dataset": dataset_wins,
            "pilot_passed": passed,
            "gate": {
                "multiple_positive_ratios": positive_ratios >= 2,
                "dose_response_reasonable": dose_inversions <= 1,
                "hidden_wins_at_least_four_folds_over_action_and_context": stable_fold_wins >= 4,
                "hidden_wins_both_datasets_over_action_and_context": dataset_wins,
                "risk_bins_approximately_ordered": variants[hidden_best]["risk_bin_inversions"] <= 1,
            },
            "note": "Pilot gate selects whether to expand the bank; it is not a <=1% safety claim.",
        },
    )


if __name__ == "__main__":
    main()
