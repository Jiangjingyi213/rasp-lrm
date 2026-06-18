from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import assert_metadata_matches
from .protocol import STAGES
from .statistics import keep_mask, stage_balanced_metric


def ratio_key(ratio: float) -> str:
    return f"{float(ratio):.4f}"


def _al_am_masks(layer_metrics: dict[int, torch.Tensor], ratios: list[float]) -> dict[int, dict[str, torch.Tensor]]:
    layer_ids = sorted(layer_metrics)
    stacked = torch.stack([layer_metrics[layer_id].float().cpu() for layer_id in layer_ids])
    standardized = (stacked - stacked.mean(dim=1, keepdim=True)) / stacked.std(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    output = {layer_id: {} for layer_id in layer_ids}
    for ratio in ratios:
        prune = min(
            standardized.numel() - 1,
            max(0, int(round(standardized.numel() * float(ratio)))),
        )
        flat = torch.ones(standardized.numel(), dtype=torch.bool)
        if prune:
            indices = torch.topk(standardized.reshape(-1), k=prune, largest=False).indices
            flat[indices] = False
        matrix = flat.reshape_as(standardized)
        for index, layer_id in enumerate(layer_ids):
            output[layer_id][ratio_key(ratio)] = matrix[index].clone()
    return output


def build_mask_bank(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, dict[int, torch.Tensor]],
    means: dict[str, dict[int, torch.Tensor]],
    ratios: list[float],
) -> dict[str, Any]:
    required = {"c4", "prompt_only", "trajectory", *STAGES}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Missing calibration metrics: {sorted(missing)}")
    layers = sorted(metrics["trajectory"])
    policies: dict[str, Any] = {}

    def global_policy(source: str) -> dict[str, Any]:
        return {
            stage: {
                layer_id: {
                    "metric": metrics[source][layer_id].float().cpu(),
                    "mean": means[source][layer_id].float().cpu(),
                    "masks": {
                        ratio_key(ratio): keep_mask(metrics[source][layer_id], ratio)
                        for ratio in ratios
                    },
                }
                for layer_id in layers
            }
            for stage in STAGES
        }

    policies["c4_global"] = global_policy("c4")
    policies["prompt_only_global"] = global_policy("prompt_only")
    policies["trajectory_global"] = global_policy("trajectory")
    balanced_metrics = {
        layer_id: stage_balanced_metric(
            {stage: metrics[stage][layer_id] for stage in STAGES}
        )
        for layer_id in layers
    }
    balanced_means = {
        layer_id: torch.stack([means[stage][layer_id].float() for stage in STAGES]).mean(dim=0)
        for layer_id in layers
    }
    policies["stage_balanced_global"] = {
        stage: {
            layer_id: {
                "metric": balanced_metrics[layer_id].cpu(),
                "mean": balanced_means[layer_id].cpu(),
                "masks": {
                    ratio_key(ratio): keep_mask(balanced_metrics[layer_id], ratio)
                    for ratio in ratios
                },
            }
            for layer_id in layers
        }
        for stage in STAGES
    }
    policies["stage_specific"] = {
        stage: {
            layer_id: {
                "metric": metrics[stage][layer_id].float().cpu(),
                "mean": means[stage][layer_id].float().cpu(),
                "masks": {
                    ratio_key(ratio): keep_mask(metrics[stage][layer_id], ratio)
                    for ratio in ratios
                },
            }
            for layer_id in layers
        }
        for stage in STAGES
    }
    shuffled = dict(zip(STAGES, (*STAGES[1:], STAGES[0])))
    policies["shuffled_stage"] = {
        stage: policies["stage_specific"][shuffled[stage]] for stage in STAGES
    }
    trajectory_al_am = _al_am_masks(metrics["trajectory"], ratios)
    policies["trajectory_global_al_am"] = {
        stage: {
            layer_id: {
                "metric": metrics["trajectory"][layer_id].float().cpu(),
                "mean": means["trajectory"][layer_id].float().cpu(),
                "masks": trajectory_al_am[layer_id],
            }
            for layer_id in layers
        }
        for stage in STAGES
    }
    policies["stage_specific_al_am"] = {}
    for stage in STAGES:
        stage_masks = _al_am_masks(metrics[stage], ratios)
        policies["stage_specific_al_am"][stage] = {
            layer_id: {
                "metric": metrics[stage][layer_id].float().cpu(),
                "mean": means[stage][layer_id].float().cpu(),
                "masks": stage_masks[layer_id],
            }
            for layer_id in layers
        }
    return {
        "schema": "stage_calibrated_mask_bank_v1",
        "metadata": metadata,
        "ratios": [float(value) for value in ratios],
        "layers": layers,
        "policies": policies,
        "policy_structures": {
            name: ("AL-AM" if name.endswith("_al_am") else "UL-UM")
            for name in policies
        },
        "shuffled_stage_mapping": shuffled,
    }


def save_mask_bank(path: str | Path, bank: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)


def load_mask_bank(path: str | Path, expected_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        bank = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        bank = torch.load(Path(path), map_location="cpu")
    if bank.get("schema") != "stage_calibrated_mask_bank_v1":
        raise ValueError("Unsupported stage-calibrated mask bank schema")
    if expected_metadata:
        assert_metadata_matches(bank.get("metadata", {}), expected_metadata)
    validate_mask_bank(bank)
    return bank


def validate_mask_bank(bank: dict[str, Any]) -> None:
    ratios = [float(value) for value in bank["ratios"]]
    for policy_name, policy in bank["policies"].items():
        structure = bank.get("policy_structures", {}).get(policy_name, "UL-UM")
        if set(policy) != set(STAGES):
            raise ValueError(f"Policy {policy_name} does not contain exactly the four stages")
        for stage in STAGES:
            previous_by_layer = {}
            for layer_id in bank["layers"]:
                entry = policy[stage][layer_id]
                masks = entry["masks"]
                if set(masks) != {ratio_key(value) for value in ratios}:
                    raise ValueError(f"Mask ratio grid mismatch in {policy_name}/{stage}/{layer_id}")
                channels = int(entry["metric"].numel())
                previous = torch.ones(channels, dtype=torch.bool)
                for ratio in sorted(ratios):
                    mask = masks[ratio_key(ratio)].bool()
                    expected_keep = max(1, channels - int(round(channels * ratio)))
                    if structure == "UL-UM" and int(mask.sum()) != expected_keep:
                        raise ValueError(f"Incorrect keep count in {policy_name}/{stage}/{layer_id}/{ratio}")
                    if bool((mask & ~previous).any()):
                        raise ValueError(f"Masks are not nested in {policy_name}/{stage}/{layer_id}")
                    previous = mask
                previous_by_layer[layer_id] = previous
            if structure == "AL-AM":
                total_channels = sum(
                    int(policy[stage][layer_id]["metric"].numel()) for layer_id in bank["layers"]
                )
                for ratio in ratios:
                    actual_keep = sum(
                        int(policy[stage][layer_id]["masks"][ratio_key(ratio)].sum())
                        for layer_id in bank["layers"]
                    )
                    expected_keep = max(1, total_channels - int(round(total_channels * ratio)))
                    if actual_keep != expected_keep:
                        raise ValueError(f"Incorrect AL-AM keep count in {policy_name}/{stage}/{ratio}")
