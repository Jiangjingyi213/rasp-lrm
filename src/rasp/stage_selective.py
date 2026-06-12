from __future__ import annotations

from typing import Any

import torch

from src.rasp.stage_probe import STAGES, stage_index


def selective_reasoning_metrics(
    labels: list[int],
    reasoning_probabilities: list[float],
    predictions: list[int],
    threshold: float,
) -> dict[str, Any]:
    setup = stage_index("setup")
    reasoning = stage_index("reasoning")
    accepted = [
        prediction == reasoning and probability >= threshold
        for prediction, probability in zip(predictions, reasoning_probabilities)
    ]
    setup_rows = sum(label == setup for label in labels)
    reasoning_rows = sum(label == reasoning for label in labels)
    accepted_rows = sum(accepted)
    true_reasoning_accepted = sum(
        take and label == reasoning for take, label in zip(accepted, labels)
    )
    setup_false_accepts = sum(
        take and label == setup for take, label in zip(accepted, labels)
    )
    return {
        "threshold": float(threshold),
        "rows": len(labels),
        "accepted_rows": accepted_rows,
        "accepted_rate": accepted_rows / max(1, len(labels)),
        "accepted_reasoning_precision": true_reasoning_accepted / max(1, accepted_rows),
        "reasoning_coverage": true_reasoning_accepted / max(1, reasoning_rows),
        "setup_false_accept_rate": setup_false_accepts / max(1, setup_rows),
        "setup_rows": setup_rows,
        "reasoning_rows": reasoning_rows,
    }


def calibrate_reasoning_threshold(
    labels: list[int],
    probabilities: torch.Tensor,
    *,
    max_setup_false_accept_rate: float,
) -> dict[str, Any]:
    reasoning = stage_index("reasoning")
    reasoning_probabilities = probabilities[:, reasoning].tolist()
    predictions = probabilities.argmax(dim=1).tolist()
    candidates = sorted({0.0, 1.0, *reasoning_probabilities})
    feasible = [
        selective_reasoning_metrics(labels, reasoning_probabilities, predictions, threshold)
        for threshold in candidates
    ]
    feasible = [
        row
        for row in feasible
        if row["setup_false_accept_rate"] <= max_setup_false_accept_rate + 1e-12
    ]
    if not feasible:
        raise ValueError("No selective reasoning threshold satisfies the setup safety constraint")
    return max(
        feasible,
        key=lambda row: (
            row["reasoning_coverage"],
            row["accepted_reasoning_precision"],
            row["threshold"],
        ),
    )


def evaluate_reasoning_threshold(
    labels: list[int],
    probabilities: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    reasoning = stage_index("reasoning")
    return selective_reasoning_metrics(
        labels,
        probabilities[:, reasoning].tolist(),
        probabilities.argmax(dim=1).tolist(),
        threshold,
    )

