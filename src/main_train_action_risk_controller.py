from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.rasp.action_risk_single_window import (
    ACTION_RISK_CHECKPOINT_SCHEMA,
    CONTEXT_FEATURE_NAMES,
    context_features,
)
from src.utils.io import read_jsonl, write_json


TARGETS = {"conservative": 0.10, "balanced": 0.20, "aggressive": 0.30}
ELIGIBLE_BOUNDARIES = (32, 96, 160)


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["id"])


def flatten(rows: list[dict[str, Any]], hidden: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts, hidden_rows, labels = [], [], []
    for index, row in enumerate(rows):
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
            hidden_rows.append(hidden[index].float().flatten().numpy())
            labels.append(int(bool(flipped)))
    return np.asarray(contexts), np.asarray(hidden_rows), np.asarray(labels)


def simulate(
    oof_rows: list[dict[str, Any]],
    *,
    context_threshold: float,
    hidden_threshold: float | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    problems = sorted({key(row) for row in oof_rows})
    for row in oof_rows:
        if int(row["generated_tokens_at_boundary"]) in ELIGIBLE_BOUNDARIES:
            grouped[key(row)][int(row["generated_tokens_at_boundary"])].append(row)
    per_dataset: dict[str, dict[str, float]] = defaultdict(
        lambda: {"problems": 0, "selected": 0, "flips": 0, "ratio_sum": 0.0}
    )
    for problem in problems:
        boundaries = grouped[problem]
        stats = per_dataset[problem[0]]
        stats["problems"] += 1
        selected = None
        for boundary in ELIGIBLE_BOUNDARIES:
            context_envelope = 0.0
            hidden_envelope = 0.0
            accepted = []
            for row in sorted(boundaries.get(boundary, []), key=lambda item: float(item["ratio"])):
                context_envelope = max(context_envelope, float(row["causal_context_action_risk"]))
                hidden_envelope = max(hidden_envelope, float(row["hidden_context_action_risk"]))
                if context_envelope <= context_threshold and (
                    hidden_threshold is None or hidden_envelope <= hidden_threshold
                ):
                    accepted.append(row)
            if accepted:
                selected = accepted[-1]
                break
        if selected is not None:
            stats["selected"] += 1
            stats["flips"] += int(bool(selected["flipped"]))
            stats["ratio_sum"] += float(selected["ratio"])
    output = {}
    for dataset, stats in sorted(per_dataset.items()):
        problems = int(stats["problems"])
        selected = int(stats["selected"])
        output[dataset] = {
            "problems": problems,
            "selected": selected,
            "action_coverage": selected / max(1, problems),
            "average_action_ratio": float(stats["ratio_sum"]) / max(1, problems),
            "selected_average_ratio": float(stats["ratio_sum"]) / max(1, selected),
            "problem_flip_rate": int(stats["flips"]) / max(1, problems),
            "selected_flip_rate": int(stats["flips"]) / max(1, selected),
        }
    output["combined"] = {
        "problems": sum(item["problems"] for item in output.values()),
        "selected": sum(item["selected"] for item in output.values()),
        "average_action_ratio": sum(
            item["average_action_ratio"] * item["problems"] for item in output.values()
        )
        / max(1, sum(item["problems"] for item in output.values())),
    }
    return output


def admissible(metrics: dict[str, Any]) -> bool:
    datasets = [value for name, value in metrics.items() if name != "combined"]
    return bool(datasets) and all(float(item["problem_flip_rate"]) <= 0.05 for item in datasets)


def threshold_grid(values: list[float], maximum_points: int = 401) -> list[float]:
    unique = np.asarray(sorted(set(float(value) for value in values)))
    if len(unique) <= maximum_points:
        return unique.tolist()
    indices = np.linspace(0, len(unique) - 1, maximum_points).round().astype(int)
    return unique[indices].tolist()


def select_context_point(oof_rows: list[dict[str, Any]], target: float) -> tuple[float, dict[str, Any]]:
    thresholds = threshold_grid([float(row["causal_context_action_risk"]) for row in oof_rows])
    candidates = []
    for threshold in thresholds:
        metrics = simulate(oof_rows, context_threshold=threshold)
        if admissible(metrics):
            distance = abs(float(metrics["combined"]["average_action_ratio"]) - target)
            candidates.append((distance, -float(metrics["combined"]["average_action_ratio"]), threshold, metrics))
    if not candidates:
        raise ValueError(f"No admissible context threshold for target {target}")
    _distance, _negative_ratio, threshold, metrics = min(candidates)
    return threshold, metrics


def select_hidden_veto(
    oof_rows: list[dict[str, Any]], context_threshold: float, context_metrics: dict[str, Any]
) -> tuple[float | None, dict[str, Any] | None]:
    minimum_ratio = 0.8 * float(context_metrics["combined"]["average_action_ratio"])
    thresholds = threshold_grid([float(row["hidden_context_action_risk"]) for row in oof_rows])
    candidates = []
    for threshold in thresholds:
        metrics = simulate(oof_rows, context_threshold=context_threshold, hidden_threshold=threshold)
        if float(metrics["combined"]["average_action_ratio"]) + 1e-12 < minimum_ratio:
            continue
        datasets = [name for name in metrics if name != "combined"]
        if all(
            float(metrics[name]["problem_flip_rate"])
            <= float(context_metrics[name]["problem_flip_rate"]) + 1e-12
            for name in datasets
        ):
            candidates.append((-float(metrics["combined"]["average_action_ratio"]), threshold, metrics))
    if not candidates:
        return None, None
    _negative_ratio, threshold, metrics = min(candidates)
    return threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--oof-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    hidden = torch.load(args.hidden_states, map_location="cpu")
    oof_rows = read_jsonl(args.oof_predictions)
    if len(rows) != len(hidden):
        raise ValueError("Action-risk rows and hidden states differ")
    max_new_tokens = {int(row["max_new_tokens"]) for row in rows}
    if len(max_new_tokens) != 1:
        raise ValueError("Action-Risk controller requires one max_new_tokens value")
    contexts, hidden_rows, labels = flatten(rows, hidden)
    context_scaler = StandardScaler().fit(contexts)
    context_scaled = context_scaler.transform(contexts)
    context_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    context_model.fit(context_scaled, labels)
    hidden_scaler = StandardScaler().fit(hidden_rows)
    hidden_scaled = hidden_scaler.transform(hidden_rows)
    rank = min(args.pca_dim, hidden_scaled.shape[0] - 1, hidden_scaled.shape[1])
    hidden_pca = PCA(n_components=rank, random_state=args.seed).fit(hidden_scaled)
    hidden_features = np.concatenate([hidden_pca.transform(hidden_scaled), context_scaled], axis=1)
    hidden_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    hidden_model.fit(hidden_features, labels)

    operating_points = {}
    for name, target in TARGETS.items():
        context_threshold, context_metrics = select_context_point(oof_rows, target)
        hidden_threshold, hidden_metrics = select_hidden_veto(oof_rows, context_threshold, context_metrics)
        operating_points[name] = {
            "eligible": True,
            "target_average_action_ratio": target,
            "context_risk_threshold": context_threshold,
            "context_oof_metrics": context_metrics,
            "hidden_veto_threshold": hidden_threshold,
            "hidden_veto_oof_metrics": hidden_metrics,
            "hidden_veto_eligible": hidden_threshold is not None,
        }
    bundle = {
        "schema": ACTION_RISK_CHECKPOINT_SCHEMA,
        "context_feature_names": CONTEXT_FEATURE_NAMES,
        "max_new_tokens": next(iter(max_new_tokens)),
        "window_tokens": 16,
        "eligible_boundaries": ELIGIBLE_BOUNDARIES,
        "candidate_ratios": tuple(float(value) for value in rows[0]["candidate_ratios"][1:]),
        "context_scaler": context_scaler,
        "context_model": context_model,
        "hidden_scaler": hidden_scaler,
        "hidden_pca": hidden_pca,
        "hidden_model": hidden_model,
        "hidden_veto_eligible": any(
            bool(point["hidden_veto_eligible"]) for point in operating_points.values()
        ),
        "operating_points": operating_points,
        "training_summary": {
            "problems": len({key(row) for row in rows}),
            "boundaries": len(rows),
            "action_rows": len(labels),
            "positive_flips": int(labels.sum()),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "action_risk_single_window.joblib")
    write_json(
        output / "calibration_summary.json",
        {
            key: value
            for key, value in bundle.items()
            if key
            not in {
                "context_scaler",
                "context_model",
                "hidden_scaler",
                "hidden_pca",
                "hidden_model",
            }
        },
    )
    print(
        {
            "checkpoint": str(output / "action_risk_single_window.joblib"),
            "calibration": str(output / "calibration_summary.json"),
            "operating_points": {
                name: {
                    "context_risk_threshold": point["context_risk_threshold"],
                    "context_average_action_ratio": point["context_oof_metrics"]["combined"][
                        "average_action_ratio"
                    ],
                    "hidden_veto_eligible": point["hidden_veto_eligible"],
                }
                for name, point in operating_points.items()
            },
        }
    )


if __name__ == "__main__":
    main()
