from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.stage_calibration.stage_risk import (
    STAGE_RISK_CHECKPOINT_SCHEMA,
    STAGE_RISK_FEATURE_NAMES,
    stage_risk_features,
)
from src.utils.io import read_jsonl, write_json, write_jsonl


STAGES = ("setup", "reasoning", "verify", "final")
MODEL_VARIANTS = ("action_only", "stage_only", "context_action", "stage_context_action")


def _key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}::{row['id']}"


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def _matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features, labels, groups, metadata = [], [], [], []
    for row in rows:
        stage = str(row["stage"])
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}")
        for candidate in row["candidates"]:
            ratio = float(candidate["ratio"])
            if ratio <= 0.0:
                continue
            features.append(
                stage_risk_features(
                    stage_local_tokens=int(row["stage_local_tokens"]),
                    entropy=float(row["entropy"]),
                    confidence=float(row["confidence"]),
                    activation_ood=float(row["activation_ood"]),
                    activation_drift=float(row["activation_drift"]),
                    mask_score_margin=float(candidate["mask_score_margin"]),
                    candidate_ratio=ratio,
                )
            )
            labels.append(int(bool(candidate["unsafe"])))
            groups.append(_key(row))
            metadata.append(
                {
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "stage": stage,
                    "ratio": ratio,
                    "unsafe": bool(candidate["unsafe"]),
                }
            )
    return np.asarray(features), np.asarray(labels), np.asarray(groups), metadata


def _design(features: np.ndarray, stages: list[str], variant: str) -> np.ndarray:
    if variant == "action_only":
        return features[:, [-1]]
    one_hot = np.zeros((len(features), len(STAGES)), dtype=np.float64)
    for index, stage in enumerate(stages):
        one_hot[index, STAGES.index(stage)] = 1.0
    if variant == "stage_only":
        return one_hot
    if variant == "context_action":
        return features
    if variant == "stage_context_action":
        interactions = np.concatenate(
            [features * one_hot[:, index : index + 1] for index in range(len(STAGES))], axis=1
        )
        return np.concatenate([features, one_hot, interactions], axis=1)
    raise ValueError(f"Unknown model variant {variant!r}")


def _fit_bundle(design: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Fit a regularized risk model and Platt calibrator using training groups only."""

    scaler = StandardScaler().fit(design)
    scaled = scaler.transform(design)
    model = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=1)
    model.fit(scaled, labels)
    group_count = len(set(groups.tolist()))
    inner_folds = min(3, group_count)
    if inner_folds < 2:
        raise ValueError("Need at least two problem groups for training-fold calibration")
    raw_oof = np.full(len(labels), np.nan)
    splitter = GroupKFold(n_splits=inner_folds)
    for train, test in splitter.split(scaled, labels, groups):
        inner = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=1)
        inner.fit(scaled[train], labels[train])
        raw_oof[test] = inner.decision_function(scaled[test])
    if np.isnan(raw_oof).any():
        raise RuntimeError("Incomplete training-fold Platt calibration predictions")
    platt = LogisticRegression(C=1e6, max_iter=1000, random_state=1)
    platt.fit(raw_oof.reshape(-1, 1), labels)
    return {"scaler": scaler, "model": model, "platt_model": platt}


def _predict(bundle: dict[str, Any], design: np.ndarray) -> np.ndarray:
    raw = bundle["model"].decision_function(bundle["scaler"].transform(design))
    return bundle["platt_model"].predict_proba(raw.reshape(-1, 1))[:, 1]


def _oof(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    metadata: list[dict[str, Any]],
    variant: str,
    folds: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(labels), np.nan)
    fold_metrics = []
    stages = [row["stage"] for row in metadata]
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(splitter.split(features, labels, groups)):
        train_x = _design(features[train], [stages[index] for index in train], variant)
        test_x = _design(features[test], [stages[index] for index in test], variant)
        bundle = _fit_bundle(train_x, labels[train], groups[train])
        output[test] = _predict(bundle, test_x)
        fold_metrics.append({"fold": fold, **_metrics(labels[test], output[test])})
    if np.isnan(output).any():
        raise RuntimeError("Incomplete grouped OOF predictions")
    return output, fold_metrics


def _bootstrap_stage_ratio_curves(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metadata: list[dict[str, Any]],
    samples: int = 500,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    rng = np.random.default_rng(1)
    group_values = np.asarray(sorted(set(groups.tolist())))
    cells: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        cells[(row["stage"], float(row["ratio"]))].append(index)
    curves: dict[str, dict[str, dict[str, float]]] = {stage: {} for stage in STAGES}
    for (stage, ratio), indices in cells.items():
        indices_arr = np.asarray(indices)
        point = float(scores[indices_arr].mean())
        draws = []
        for _ in range(samples):
            chosen = set(rng.choice(group_values, size=len(group_values), replace=True).tolist())
            selected = np.asarray([index for index in indices if groups[index] in chosen])
            if len(selected):
                draws.append(float(scores[selected].mean()))
        low, high = np.quantile(draws or [point], [0.025, 0.975])
        curves[stage][f"{ratio:.2f}"] = {
            "n": int(len(indices)), "unsafe_rate": float(labels[indices_arr].mean()),
            "predicted_risk": point, "ci95_low": float(low), "ci95_high": float(high),
        }
    distinguishable = set()
    for left_index, left in enumerate(STAGES):
        for right in STAGES[left_index + 1 :]:
            for ratio in set(curves[left]) & set(curves[right]):
                a, b = curves[left][ratio], curves[right][ratio]
                if a["ci95_high"] < b["ci95_low"] or b["ci95_high"] < a["ci95_low"]:
                    distinguishable.update((left, right))
    return curves, sorted(distinguishable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--required-sources", nargs="*", default=[])
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    features, labels, groups, metadata = _matrix(rows)
    if len(set(groups)) < args.folds:
        raise ValueError("Insufficient problem groups for requested OOF folds")
    if len(np.unique(labels)) < 2:
        raise ValueError("Stage-risk bank needs both safe and unsafe candidate actions")
    scores_by_variant: dict[str, np.ndarray] = {}
    folds_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in MODEL_VARIANTS:
        scores_by_variant[variant], folds_by_variant[variant] = _oof(
            features, labels, groups, metadata, variant, args.folds
        )
    by_source = {}
    observed_sources = sorted({row["dataset"] for row in metadata})
    for source in observed_sources:
        indices = np.asarray([index for index, row in enumerate(metadata) if row["dataset"] == source])
        by_source[source] = {
            variant: _metrics(labels[indices], scores_by_variant[variant]) for variant in MODEL_VARIANTS
        }
    context_scores = scores_by_variant["context_action"]
    stage_scores = scores_by_variant["stage_context_action"]
    fold_wins = sum(
        int(
            context["roc_auc"] is not None and stage["roc_auc"] is not None
            and stage["roc_auc"] > context["roc_auc"] and stage["pr_auc"] > context["pr_auc"]
        )
        for context, stage in zip(
            folds_by_variant["context_action"], folds_by_variant["stage_context_action"]
        )
    )
    required_sources = list(args.required_sources) or observed_sources
    missing_sources = sorted(set(required_sources) - set(observed_sources))
    source_non_degrading = not missing_sources and all(
        by_source[source]["stage_context_action"]["roc_auc"] is not None
        and by_source[source]["stage_context_action"]["pr_auc"] is not None
        and by_source[source]["stage_context_action"]["roc_auc"]
        >= by_source[source]["context_action"]["roc_auc"]
        and by_source[source]["stage_context_action"]["pr_auc"]
        >= by_source[source]["context_action"]["pr_auc"]
        for source in required_sources
    )
    curves, distinguishable_stages = _bootstrap_stage_ratio_curves(labels, stage_scores, groups, metadata)
    stage_gate_passed = (
        fold_wins >= 4 and source_non_degrading and len(distinguishable_stages) >= 2
    )

    full_design = _design(features, [row["stage"] for row in metadata], "stage_context_action")
    bundle = _fit_bundle(full_design, labels, groups)
    action_ratios = tuple(sorted({float(row["ratio"]) for row in metadata}))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": STAGE_RISK_CHECKPOINT_SCHEMA,
            "feature_names": STAGE_RISK_FEATURE_NAMES,
            "stages": STAGES,
            "action_ratios": action_ratios,
            "calibration": "grouped_training_fold_platt",
            "mechanism_gate_passed": stage_gate_passed,
            **bundle,
        },
        output / "stage_risk_controller.joblib",
    )
    summary = {
        "schema": "stage_risk_adaptive_oof_v1",
        "problems": len(set(groups)), "action_rows": len(labels), "positive_unsafe": int(labels.sum()),
        "models": {variant: _metrics(labels, scores_by_variant[variant]) for variant in MODEL_VARIANTS},
        "fold_metrics": folds_by_variant,
        "by_source": by_source,
        "required_sources": required_sources,
        "missing_required_sources": missing_sources,
        "stage_fold_wins": fold_wins,
        "stage_ratio_risk_curves": curves,
        "distinguishable_stages": distinguishable_stages,
        "stage_controller_training_allowed": stage_gate_passed,
        "checkpoint": str(output / "stage_risk_controller.joblib"),
    }
    write_json(output / "stage_risk_oof_summary.json", summary)
    write_jsonl(
        output / "stage_risk_oof_predictions.jsonl",
        [
            {**row, **{f"{variant}_risk": float(scores_by_variant[variant][index]) for variant in MODEL_VARIANTS}}
            for index, row in enumerate(metadata)
        ],
    )


if __name__ == "__main__":
    main()
