from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch

from src.probes.rasp_train_dataset import RaspTrainPolicyDataset
from src.probes.train_probe import LinearRiskProbe
from src.probes.action_conditioned_dataset import build_action_features
from src.rasp.safe_oracle import available_prefix_budget
from src.rasp.train_policy import (
    POLICY_FEATURE_SCHEMA,
    ActionRiskPolicyNet,
    causal_action_risk_indices,
)
from src.utils.io import ensure_dir, write_json, write_jsonl


def _load_policy(checkpoint_path: str | Path) -> tuple[ActionRiskPolicyNet, list[float], float, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = dict(checkpoint.get("metadata", {}))
    if metadata.get("feature_schema") != POLICY_FEATURE_SCHEMA:
        raise ValueError("Policy checkpoint feature schema is incompatible with this evaluator")
    model = ActionRiskPolicyNet(
        int(checkpoint["dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        model,
        [float(value) for value in checkpoint["ratios"]],
        float(metadata["calibrated_threshold"]),
        metadata,
    )


def _load_risk_probe(checkpoint_path: str | Path) -> tuple[LinearRiskProbe, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = LinearRiskProbe(int(checkpoint["dim"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, dict(checkpoint.get("metadata", {}))


def _ratio_index(ratio: float, ratios: list[float]) -> int:
    return min(range(len(ratios)), key=lambda index: abs(float(ratios[index]) - float(ratio)))


def _summarize(name: str, selected_indices: list[int], rows: list[dict], ratios: list[float]) -> dict:
    flips = []
    selected_ratios = []
    oracle_ratios = []
    oracle_matches = []
    unsafe_over_oracle = []
    conservative_unsafe = []
    for selected_index, row in zip(selected_indices, rows):
        selected_ratio = float(ratios[selected_index])
        oracle_ratio = float(row["oracle_ratio"])
        flipped = bool(row["candidate_flipped"][selected_index])
        unsafe = bool(row.get("candidate_unsafe", row["candidate_flipped"])[selected_index])
        flips.append(float(flipped))
        selected_ratios.append(selected_ratio)
        oracle_ratios.append(oracle_ratio)
        oracle_matches.append(float(abs(selected_ratio - oracle_ratio) < 1e-8))
        unsafe_over_oracle.append(float(flipped and selected_ratio > oracle_ratio + 1e-8))
        conservative_unsafe.append(float(unsafe))
    target = sum(float(row["target_budget"]) for row in rows) / max(1, len(rows))
    avg_ratio = sum(selected_ratios) / max(1, len(selected_ratios))
    return {
        "policy": name,
        "n": len(rows),
        "target_average_ratio": target,
        "average_selected_ratio": avg_ratio,
        "budget_utilization": avg_ratio / target if target > 0 else None,
        "flip_rate": sum(flips) / max(1, len(flips)),
        "conservative_unsafe_rate": sum(conservative_unsafe) / max(1, len(conservative_unsafe)),
        "oracle_match_rate": sum(oracle_matches) / max(1, len(oracle_matches)),
        "unsafe_over_oracle_rate": sum(unsafe_over_oracle) / max(1, len(unsafe_over_oracle)),
        "average_oracle_ratio": sum(oracle_ratios) / max(1, len(oracle_ratios)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _indices_for_problem_keys(rows: list[dict], keys: list[list[str]]) -> list[int]:
    selected_keys = {(str(dataset), str(problem_id)) for dataset, problem_id in keys}
    return [
        index
        for index, row in enumerate(rows)
        if (str(row.get("dataset") or "unknown"), str(row["id"])) in selected_keys
    ]


def _problem_sequences(rows: list[dict]) -> list[list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row.get("dataset") or "unknown"), str(row["id"]))].append(index)
    sequences = list(grouped.values())
    for indices in sequences:
        indices.sort(
            key=lambda index: (
                int(rows[index].get("segment_index", rows[index].get("segment_id", 0))),
                int(rows[index].get("segment_id", 0)),
            )
        )
    return sequences


def _static_indices(rows: list[dict], ratios: list[float], ratio: float) -> list[int]:
    selected = _ratio_index(ratio, ratios)
    return [selected for _ in rows]


def _oracle_indices(rows: list[dict], ratios: list[float]) -> list[int]:
    return [_ratio_index(float(row["oracle_ratio"]), ratios) for row in rows]


def _score_order_budget_indices(
    rows: list[dict],
    ratios: list[float],
    score_fn: Callable[[dict], float],
) -> list[int]:
    selected = [0] * len(rows)
    target_total = sum(float(row["target_budget"]) for row in rows)
    candidates = []
    for row_index, row in enumerate(rows):
        score = score_fn(row)
        for ratio_index, ratio in enumerate(ratios):
            if ratio > 0.0:
                candidates.append((score, float(ratio), row_index, ratio_index))
    candidates.sort(key=lambda item: (item[0], item[1]))
    total = 0.0
    for _score, ratio, row_index, ratio_index in candidates:
        increment = ratio - float(ratios[selected[row_index]])
        if increment <= 1e-9:
            continue
        if total + increment > target_total + 1e-9:
            continue
        selected[row_index] = ratio_index
        total += increment
    return selected


def _risk_router_indices(
    model: LinearRiskProbe,
    rows: list[dict],
    hidden: torch.Tensor,
    ratios: list[float],
    threshold: float,
) -> list[int]:
    selected = [0] * len(rows)
    runtime_layers = list(range(28))
    with torch.no_grad():
        for sequence in _problem_sequences(rows):
            selected_history: list[float] = []
            for row_index in sequence:
                row = rows[row_index]
                available = available_prefix_budget(float(row["target_budget"]), selected_history)
                selected_index = 0
                for ratio_index in sorted(range(len(ratios)), key=lambda index: ratios[index], reverse=True):
                    ratio = float(ratios[ratio_index])
                    if ratio <= 0.0 or ratio > available + 1e-9:
                        continue
                    features = build_action_features(
                        hidden[row_index],
                        entropy=float(row.get("entropy", 0.0)),
                        confidence=float(row.get("confidence", 0.0)),
                        position=float(row.get("position", 0.0)),
                        ratio=ratio,
                        module="mlp_intermediate_channels",
                        dataset=str(row.get("dataset") or "unknown"),
                        pruned_layers=runtime_layers,
                        layer_dim=len(runtime_layers),
                    )
                    risk = float(torch.sigmoid(model(features.unsqueeze(0))).item())
                    if risk <= threshold:
                        selected_index = ratio_index
                        break
                selected[row_index] = selected_index
                selected_history.append(float(ratios[selected_index]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--risk-router-checkpoint")
    parser.add_argument("--risk-threshold", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Ignored for v2 checkpoints")
    parser.add_argument("--seed", type=int, default=1, help="Ignored for v2 checkpoints")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    dataset = RaspTrainPolicyDataset(args.dataset, args.hidden_states)
    policy, ratios, threshold, metadata = _load_policy(args.policy_checkpoint)
    if ratios != dataset.ratios:
        raise ValueError(f"policy ratios {ratios} differ from dataset ratios {dataset.ratios}")
    test_indices = _indices_for_problem_keys(dataset.rows, metadata["split_problem_keys"]["test"])
    if not test_indices:
        raise ValueError("Policy checkpoint test split does not match this dataset")
    val_rows = [dataset.rows[index] for index in test_indices]
    val_hidden = dataset.hidden[torch.tensor(test_indices, dtype=torch.long)]
    device = torch.device("cpu")
    trained_indices, trained_risks = causal_action_risk_indices(
        policy, val_rows, val_hidden, ratios, threshold, device
    )

    selected_by_policy = {
        "rasp_train_v2_action_risk": trained_indices,
        "safe_oracle": _oracle_indices(val_rows, ratios),
        "offline_noncausal_entropy_budget": _score_order_budget_indices(
            val_rows, ratios, lambda row: float(row.get("entropy", 0.0))
        ),
        "offline_noncausal_confidence_budget": _score_order_budget_indices(
            val_rows, ratios, lambda row: -float(row.get("confidence", 0.0))
        ),
    }
    for ratio in [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]:
        selected_by_policy[f"static_{ratio:.2f}"] = _static_indices(val_rows, ratios, ratio)
    if args.risk_router_checkpoint:
        risk_model, _metadata = _load_risk_probe(args.risk_router_checkpoint)
        selected_by_policy["rasp_zero_risk_budget"] = _risk_router_indices(
            risk_model,
            val_rows,
            val_hidden,
            ratios,
            threshold=float(args.risk_threshold),
        )

    summaries = [_summarize(name, indices, val_rows, ratios) for name, indices in selected_by_policy.items()]
    selected_rows = []
    for name, selected_indices in selected_by_policy.items():
        for row_index, (row, selected_index) in enumerate(zip(val_rows, selected_indices)):
            selected_rows.append(
                {
                    "policy": name,
                    "dataset": row.get("dataset"),
                    "id": row["id"],
                    "segment_id": row["segment_id"],
                    "target_budget": row["target_budget"],
                    "oracle_ratio": row["oracle_ratio"],
                    "selected_ratio": float(ratios[selected_index]),
                    "selected_flipped": bool(row["candidate_flipped"][selected_index]),
                    "selected_conservative_unsafe": bool(
                        row.get("candidate_unsafe", row["candidate_flipped"])[selected_index]
                    ),
                    "candidate_predicted_risks": (
                        trained_risks[row_index]
                        if name == "rasp_train_v2_action_risk"
                        else None
                    ),
                }
            )
    write_json(
        output_dir / "12_rasp_train_offline_summary.json",
        {
            "method": "rasp_train_v2_action_risk",
            "calibrated_threshold": threshold,
            "test_problem_count": len(
                {(str(row.get("dataset") or "unknown"), str(row["id"])) for row in val_rows}
            ),
            "test_rows": len(val_rows),
            "summaries": summaries,
        },
    )
    _write_csv(output_dir / "12_rasp_train_offline_summary.csv", summaries)
    write_jsonl(output_dir / "12_rasp_train_offline_selected_actions.jsonl", selected_rows)


if __name__ == "__main__":
    main()
